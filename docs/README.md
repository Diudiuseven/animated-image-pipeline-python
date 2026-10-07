# Animated Image Pipeline (local, in-process)

A self-contained server-side image processing service. It accepts uploaded
images, corrects orientation, normalizes color, resizes on demand, and
transcodes between static images and multi-frame animations. Everything runs
inside this process using in-process samples or local files — no external
service is involved, and results/errors are reproducible locally.

## Supported formats

| Container | Read | Write | Animation | Alpha |
|-----------|------|-------|-----------|-------|
| PNG / APNG | ✅ | ✅ | ✅ (APNG) | ✅ |
| JPEG | ✅ | ✅ | — (static, use `still=true` to take frame 1) | flattened on a configurable background |
| GIF | ✅ | ✅ | ✅ | 1-bit |
| WebP | ✅ | ✅ | ✅ (animated WebP) | ✅ |

Input/output container is identified from magic bytes before decoding.

## Processing stages (in order)

1. **Request + source validation** — parameter and byte-size boundaries are
   checked before any decode work.
2. **Decode pre-walk** — the whole container is walked and every frame is
   decoded once up front, so a truncated file, an undecodable frame, an
   illegal frame duration/size or a capacity violation is rejected early.
3. **Orientation correction** — the single EXIF orientation tag (1..8) is
   applied geometrically to **every** frame (validated against
   `ImageOps.exif_transpose`).
4. **Color normalization** — every frame is converted to straight (unassociated)
   **sRGB RGBA uint8**; embedded ICC profiles are transformed to sRGB via
   littlecms2. The same content from tagged/untagged or differently-encoded
   sources normalizes to identical pixels.
5. **Frame composition** — GIF disposal (0..3) + sub-frame extents, APNG
   dispose/blend and animated-WebP composition are reproduced by the Pillow 12
   decoders; each canonical frame is the fully composited canvas.
6. **Resize** — fit/cover/stretch with premultiplied-alpha LANCZOS resampling
   to avoid black fringes and color bleed at transparent edges.
7. **Encode** — fixed, ordered, deterministic encoder settings; no timestamps
   or filenames enter the byte stream.

## Frame semantics

* **No dropped / duplicated / invented frames.** Frame count, per-frame
  durations and loop count (`0` = infinite) are preserved across transcode and
  resize.
* Container encoders (libwebp, Pillow GIF/APNG) silently merge *pixel-identical*
  adjacent frames and fold their durations together. To keep the strict frame
  count, the encoder inserts a tiny deterministic **frame-preservation
  sentinel** on duplicate runs:
  * WebP/APNG: a 1-level alpha nudge on a corner pixel (visually identical,
    lossless-preserved);
  * GIF: because transparency is 1-bit and the palette quantizes small deltas,
    a single corner pixel is shifted by one palette step (≥32 levels). This is
    the unavoidable GIF tradeoff for strict frame-count fidelity; it affects
    exactly one corner pixel of genuinely duplicated frames.
* Transparency and disposal are honored end-to-end; semi-transparent
  overlapping frames blend without black edges, color spill or residue of a
  previous frame.

## Determinism

Output bytes are a pure function of `(source bytes, request, settings)`:

* pure, per-request, stateless stage functions;
* fixed rounding, fixed resampling filter and fixed encoder parameters;
* identical results at any concurrency, in any processing order and after a
  process restart (covered by a cross-process test).

## Capacity bounds and queue policy

All bounds are configurable in `imagepipeline.config.Limits`; oversize
requests are rejected **before** heavy processing with `limit_exceeded`:

| Setting | Default | Meaning |
|---|---|---|
| `max_source_bytes` | 64 MiB | raw upload size |
| `max_frames` | 500 | frames per image |
| `max_dimension` | 16384 | max canvas edge |
| `max_pixels_per_frame` | 40,000,000 | pixels of one canvas |
| `max_total_pixels` | 500,000,000 | frames × canvas pixels |
| `min_frame_duration_ms` / `max_frame_duration_ms` | 1 / 60000 | per-frame timing |
| `max_total_duration_ms` | 600000 | sum of durations |
| `max_target_dimension` | 8192 | requested output edge |
| `max_concurrent_requests` | 8 | global in-flight admission |
| `max_concurrent_decodes` | 2 | bounded decode thread pool |
| `max_concurrent_encodes` | 2 | bounded encode thread pool |
| `queue_policy` | `reject` | `reject` (429 `queue_rejected`) or `queue` |
| `queue_timeout_s` | 2.0 | max wait when policy is `queue` |
| `processing_timeout_s` | 30.0 | per-request wall-clock budget |

One oversized/overlong animation can only consume its own rejected request and
cannot starve the rest of a batch.

## Failure model

Errors carry a stable machine-readable `code`, a message and
`request_id`/`source_id` for localization. No partial image is ever exposed and
no shared state is left behind:

| code | HTTP | reason |
|---|---|---|
| `invalid_input` | 422 | empty/truncated/undecodable, bad signature, illegal frame timing/size |
| `unsupported_format` | 422 | container/codec not enabled |
| `invalid_request` | 422 | bad parameters; animated → JPEG without `still` |
| `limit_exceeded` | 422 | a configured bound was crossed |
| `queue_rejected` | 429 | at capacity (immediately or after `queue_timeout_s`) |
| `processing_timeout` | 408 | per-request timeout |
| `cancelled` | 499 | client cancellation |
| `internal_processing_error` | 500 | unexpected stage failure |

## HTTP API

* `GET /health` — liveness and active bounds.
* `GET /samples` / `GET /samples/{name}` — list/download in-process samples
  (`animated_transparent`, `animated_webp`, `static_png`, `oriented_jpeg`).
* `POST /process` — multipart `file` plus form fields `target_format`,
  `resize_mode` (`none|fit|cover|stretch`), `width`, `height`, `quality`,
  `lossless`, `still`, `loop`. Returns the encoded image with
  `X-Request-Id`, `X-Frame-Count`, `X-Width`, `X-Height` headers.

Run locally:

```bash
uv run uvicorn imagepipeline.api:app --host 127.0.0.1 --port 8000
# or
uv run python -m uvicorn imagepipeline.api:app --port 8000
```

## Tests

```bash
uv run pytest -q
```

Covered: all 8 EXIF orientations, color normalization, GIF/WebP/APNG frame
composition and transparency, premultiplied-alpha resize, byte determinism
(repeat / order / concurrency / fresh-process restart), oversized animations,
queue/timeout/cancellation and corrupt/truncated input rejection.
