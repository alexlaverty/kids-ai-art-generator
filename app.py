"""Kid-friendly ComfyUI image chat — single-file FastAPI backend.

Run ComfyUI first (default: http://127.0.0.1:8188), then:
    pip install -r requirements.txt
    python app.py
and open http://127.0.0.1:8777
"""

import asyncio
import json
import random
import re
import sqlite3
import time
from io import BytesIO
from pathlib import Path

import httpx
import websockets
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)
STYLE_IMAGES_DIR = IMAGES_DIR / "styles"
STYLE_IMAGES_DIR.mkdir(exist_ok=True)
DB_PATH = BASE_DIR / "chat.db"

COMFY_URL = "http://127.0.0.1:8188"
GENERATION_TIMEOUT_SECONDS = 600

NEGATIVE_PROMPT = (
    "scary, creepy, violent, gore, blood, nsfw, nude, weapon, "
    "blurry, deformed, watermark, text, signature, low quality"
)

STYLES = json.loads((BASE_DIR / "styles.json").read_text(encoding="utf-8"))

# Fixed prompts used for the Styles learning page. Every style renders the same
# prompt so kids can flip through and compare how each style draws one subject.
EXAMPLE_PROMPTS = [
    "a friendly cat sitting in a sunny garden",
    "a spaceship zooming past colorful planets",
    "a castle on a hill with a rainbow in the sky",
    "a friendly dragon reading a book",
    "a lighthouse on a cliff by the sea",
    "a robot making pancakes",
    "a unicorn in a magical forest",
    "an old sailing ship on the ocean",
    "a cozy treehouse at sunset",
    "a penguin holding a red balloon",
]


def style_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# Base square edge per model family and size choice. Generation time scales
# with pixel count, so "small" is the kid-friendly fast default.
SIZE_PRESETS = {
    "sd15": {"small": 512, "medium": 640, "large": 768},
    "xl": {"small": 768, "medium": 1024, "large": 1216},
}
ORIENTATIONS = ("square", "tall", "wide")


def resolve_dims(base: int, orientation: str) -> tuple[int, int]:
    """Turn a base square edge + orientation into width/height (multiples of 64)."""
    if orientation == "tall":
        w, h = int(base * 0.75), base
    elif orientation == "wide":
        w, h = base, int(base * 0.75)
    else:
        w = h = base
    return max(w // 64 * 64, 64), max(h // 64 * 64, 64)


# Live preview frames and progress for in-flight generations, keyed by the
# frontend-supplied client id. Filled by a websocket watcher per generation.
PREVIEWS: dict[str, dict] = {}

app = FastAPI(title="Art Robot")


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


with get_db() as _conn:
    _conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            style TEXT NOT NULL DEFAULT '',
            filename TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        )
        """
    )
    _cols = [row[1] for row in _conn.execute("PRAGMA table_info(messages)")]
    if "elapsed_seconds" not in _cols:
        _conn.execute("ALTER TABLE messages ADD COLUMN elapsed_seconds REAL")


class GenerateRequest(BaseModel):
    prompt: str
    style: str = ""
    size: str = "small"
    orientation: str = "square"
    client_id: str = ""


def build_checkpoint_workflow(positive: str, ckpt_name: str, width: int, height: int, seed: int) -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE_PROMPT, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": seed,
                "steps": 12,
                "cfg": 7,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1,
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "art-robot"}},
    }


def build_flux2_workflow(positive: str, unet: str, text_encoder: str, vae: str, width: int, height: int, seed: int) -> dict:
    # Same graph as the proven-working new-hero.py script: separate UNET/CLIP/VAE
    # loaders, plain KSampler with cfg 4 and scheduler "simple", real negative prompt.
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": text_encoder, "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE_PROMPT, "clip": ["2", 0]}},
        "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": 12,
                "cfg": 4.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "art-robot"}},
    }


async def list_models(client: httpx.AsyncClient, folder: str) -> list[str]:
    resp = await client.get(f"/models/{folder}")
    resp.raise_for_status()
    return resp.json()


async def pick_workflow(
    client: httpx.AsyncClient, positive: str, seed: int, size: str = "medium", orientation: str = "square"
) -> dict:
    """Choose a workflow based on what's installed: regular checkpoint if present,
    otherwise a Flux 2 diffusion model + text encoder + VAE."""
    checkpoints = await list_models(client, "checkpoints")
    if checkpoints:
        ckpt_name = checkpoints[0]
        family = "xl" if any(tag in ckpt_name.lower() for tag in ("xl", "flux")) else "sd15"
        base = SIZE_PRESETS[family].get(size, SIZE_PRESETS[family]["medium"])
        width, height = resolve_dims(base, orientation)
        return build_checkpoint_workflow(positive, ckpt_name, width, height, seed)

    diffusion_models = await list_models(client, "diffusion_models")
    text_encoders = await list_models(client, "text_encoders")
    vaes = await list_models(client, "vae")
    unet = next((m for m in diffusion_models if "flux" in m.lower()), None)
    vae = next((v for v in vaes if "flux" in v.lower()), None)
    if not (unet and text_encoders and vae):
        raise HTTPException(
            status_code=503,
            detail="ComfyUI is running but I couldn't find a usable model "
            "(need a checkpoint, or a Flux diffusion model + text encoder + VAE).",
        )
    base = SIZE_PRESETS["xl"].get(size, SIZE_PRESETS["xl"]["medium"])
    width, height = resolve_dims(base, orientation)
    return build_flux2_workflow(positive, unet, text_encoders[0], vae, width, height, seed)


async def watch_preview(client_id: str):
    """Relay ComfyUI's per-step progress + latent preview frames into PREVIEWS
    so the frontend can show the picture emerging while it generates."""
    uri = COMFY_URL.replace("http", "ws", 1) + f"/ws?clientId={client_id}"
    try:
        async with websockets.connect(uri, max_size=None) as ws:
            while True:
                msg = await ws.recv()
                slot = PREVIEWS.setdefault(client_id, {})
                if isinstance(msg, bytes):
                    # binary frame: 4-byte event type (1 = preview image),
                    # 4-byte image format, then the JPEG/PNG bytes
                    if len(msg) > 8 and int.from_bytes(msg[:4], "big") == 1:
                        slot["image"] = msg[8:]
                else:
                    data = json.loads(msg)
                    if data.get("type") == "progress":
                        slot["value"] = data["data"].get("value")
                        slot["max"] = data["data"].get("max")
    except (asyncio.CancelledError, Exception):
        pass


async def run_comfy_generation(
    positive: str, seed: int, size: str = "medium", orientation: str = "square", client_id: str = ""
) -> bytes:
    """Submit a text-to-image job to ComfyUI and return the PNG bytes."""
    async with httpx.AsyncClient(base_url=COMFY_URL, timeout=30) as client:
        try:
            workflow = await pick_workflow(client, positive, seed, size, orientation)
        except httpx.HTTPError:
            raise HTTPException(
                status_code=503,
                detail="The art robot isn't awake! Ask a grown-up to start ComfyUI.",
            )
        body = {"prompt": workflow}
        watcher = None
        if client_id:
            body["client_id"] = client_id
            watcher = asyncio.create_task(watch_preview(client_id))
        resp = await client.post("/prompt", json=body)
        if resp.status_code != 200:
            if watcher:
                watcher.cancel()
            raise HTTPException(status_code=502, detail=f"ComfyUI rejected the job: {resp.text[:300]}")
        prompt_id = resp.json()["prompt_id"]

        try:
            deadline = time.monotonic() + GENERATION_TIMEOUT_SECONDS
            outputs = None
            while time.monotonic() < deadline:
                await asyncio.sleep(1)
                resp = await client.get(f"/history/{prompt_id}")
                history = resp.json().get(prompt_id)
                if not history:
                    continue
                if history.get("status", {}).get("status_str") == "error":
                    raise HTTPException(status_code=502, detail="The art robot had a problem making that one. Try again!")
                if history.get("outputs"):
                    outputs = history["outputs"]
                    break
            if outputs is None:
                raise HTTPException(status_code=504, detail="The art robot took too long. Try again!")

            image_ref = None
            for node_output in outputs.values():
                if node_output.get("images"):
                    image_ref = node_output["images"][0]
                    break
            if image_ref is None:
                raise HTTPException(status_code=502, detail="ComfyUI finished but made no image.")

            resp = await client.get(
                "/view",
                params={
                    "filename": image_ref["filename"],
                    "subfolder": image_ref.get("subfolder", ""),
                    "type": image_ref.get("type", "output"),
                },
            )
            resp.raise_for_status()
            return resp.content
        finally:
            if watcher:
                watcher.cancel()
            PREVIEWS.pop(client_id, None)


@app.get("/api/styles")
def list_styles():
    return STYLES


@app.get("/api/example-prompts")
def list_example_prompts():
    return EXAMPLE_PROMPTS


@app.get("/api/style-images")
def list_style_images():
    """For each style, which example images already exist on disk.
    Returns {style name: {prompt index: {path, mtime}}} with paths relative to /images/."""
    result = {}
    for style in STYLES:
        slug = style_slug(style["name"])
        existing = {}
        for i in range(len(EXAMPLE_PROMPTS)):
            filename = f"{slug}_{i}.webp"
            file_path = STYLE_IMAGES_DIR / filename
            if file_path.exists():
                existing[i] = {"path": f"styles/{filename}", "mtime": file_path.stat().st_mtime}
        result[style["name"]] = existing
    return result


@app.get("/api/progress/{client_id}")
def get_progress(client_id: str):
    slot = PREVIEWS.get(client_id)
    if slot is None:
        raise HTTPException(status_code=404, detail="no such generation")
    return {"value": slot.get("value"), "max": slot.get("max"), "preview": "image" in slot}


@app.get("/api/preview/{client_id}")
def get_preview(client_id: str):
    slot = PREVIEWS.get(client_id)
    if slot is None or "image" not in slot:
        raise HTTPException(status_code=404, detail="no preview yet")
    return Response(content=slot["image"], media_type="image/jpeg")


class StyleImageRequest(BaseModel):
    style: str
    prompt_index: int


@app.post("/api/style-images/generate")
async def generate_style_image(req: StyleImageRequest):
    style = next((s for s in STYLES if s["name"] == req.style), None)
    if style is None:
        raise HTTPException(status_code=404, detail=f"Unknown style: {req.style}")
    if not 0 <= req.prompt_index < len(EXAMPLE_PROMPTS):
        raise HTTPException(status_code=400, detail="prompt_index out of range")

    positive = f"{EXAMPLE_PROMPTS[req.prompt_index]}, {style['suffix']}"
    seed = random.randint(0, 2**31)
    image_bytes = await run_comfy_generation(positive, seed)

    # examples ship in the git repo, so store them as WebP (~10x smaller than PNG)
    filename = f"{style_slug(style['name'])}_{req.prompt_index}.webp"
    Image.open(BytesIO(image_bytes)).save(STYLE_IMAGES_DIR / filename, "WEBP", quality=85)
    return {
        "style": style["name"],
        "prompt_index": req.prompt_index,
        "filename": f"styles/{filename}",
        "mtime": time.time(),
    }


@app.get("/api/messages")
def list_messages():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM messages ORDER BY id").fetchall()
    return [dict(row) for row in rows]


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Type what you want to see first!")

    style = next((s for s in STYLES if s["name"] == req.style), None)
    positive = f"{prompt}, {style['suffix']}" if style else prompt
    seed = random.randint(0, 2**31)
    size = req.size if req.size in ("small", "medium", "large") else "small"
    orientation = req.orientation if req.orientation in ORIENTATIONS else "square"
    started = time.monotonic()
    image_bytes = await run_comfy_generation(positive, seed, size, orientation, req.client_id)
    elapsed = round(time.monotonic() - started, 1)

    filename = f"{int(time.time() * 1000)}_{seed}.png"
    (IMAGES_DIR / filename).write_bytes(image_bytes)

    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO messages (prompt, style, filename, elapsed_seconds) VALUES (?, ?, ?, ?)",
            (prompt, req.style if style else "", filename, elapsed),
        )
        row = conn.execute("SELECT * FROM messages WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8777)
