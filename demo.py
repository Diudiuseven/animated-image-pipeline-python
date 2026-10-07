"""Local self-contained demo: generate samples, process, verify determinism.

Run: uv run python demo.py
Writes outputs into ./demo_out/ (created if missing).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from image_pipeline import samples
from image_pipeline.model import ProcessRequest, ResizeMode
from image_pipeline.pipeline import process_or_reject


def main() -> None:
    out_dir = Path("demo_out")
    out_dir.mkdir(exist_ok=True)
    sources = {
        "anim.gif": samples.make_animated_gif(frames=5, size=(64, 64)),
        "anim.webp": samples.make_animated_webp(frames=5, size=(64, 64)),
        "anim.png": samples.make_animated_png(frames=5, size=(64, 64)),
        "static.jpg": samples.make_static_jpeg(orientation=6),
    }
    targets = ("png", "webp", "gif")
    seen: dict[tuple, str] = {}
    for name, data in sources.items():
        for target in targets:
            req = ProcessRequest(
                source=data,
                source_id=name,
                output_format=target,
                width=40,
                resize_mode=ResizeMode.FIT,
                request_id=f"{name}-{target}",
            )
            result = process_or_reject(req)
            if not hasattr(result, "data"):
                print(f"REJECT {name} -> {target}: {result.code} {result.message}")
                continue
            p = out_dir / f"{Path(name).stem}.{target}"
            p.write_bytes(result.data)
            digest = hashlib.sha256(result.data).hexdigest()[:16]
            key = (name, target)
            seen[key] = digest
            print(
                f"OK {name} -> {p.name}  {result.width}x{result.height} "
                f"frames={result.frame_count} sha256={digest}"
            )
    # determinism re-check
    for (name, target), digest in seen.items():
        req = ProcessRequest(
            source=sources[name], source_id=name, output_format=target,
            width=40, request_id="again",
        )
        r = process_or_reject(req)
        assert hasattr(r, "data")
        assert hashlib.sha256(r.data).hexdigest()[:16] == digest
    print("determinism verified across", len(seen), "successful transcodes")


if __name__ == "__main__":
    main()
