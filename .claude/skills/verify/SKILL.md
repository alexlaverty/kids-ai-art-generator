---
name: verify
description: How to run and verify the Art Robot app (FastAPI + single-file frontend)
---

# Verifying comfy-simple (Art Robot)

## Launch

```bash
python app.py    # serves http://127.0.0.1:8777
```

- Port 8777 may already be in use — the user often has the server running.
  A bind failure ([winerror 10048]) means drive the existing instance instead.
- The frontend is one file: `static/index.html`, served fresh from disk on
  every request — no build step, no cache; just reload the browser tab.
- ComfyUI (http://127.0.0.1:8188) is only needed for actual image
  generation. All UI flows (tabs, composer, Idea Machine, styles page,
  gallery) work without it; generation attempts fail with a friendly
  503 bubble.

## Drive

Tabs are hash-routed: `/#chat`, `/#generator`, `/#gallery`, `/#styles`.
Navigating to the same URL with a different hash does NOT reload the page —
use `location.reload()` to pick up edited HTML.

Flows worth driving:
- Chat composer: prompt box, Surprise!/Create!, style dropdown (+ fun fact),
  Size/Shape accordions.
- Generator: SPIN!, per-reel lock/reroll, style reel, TTS "Read it!" button
  (uses speechSynthesis), sentence box.
- Styles page: card click → golden glow + sticky "Use <style>!" bar;
  image click opens the lightbox (stopPropagation, must not toggle selection).

## Gotchas

- Sandboxed curl/Invoke-WebRequest to 127.0.0.1:8777 can fail while the
  browser reaches it fine; probe from the page (performance resource
  entries) before blaming the change.
- Style example images load with a `?v=Date.now()` cache-buster and are
  re-requested on every styles-page render.
