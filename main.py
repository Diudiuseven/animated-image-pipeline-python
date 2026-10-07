"""Local entrypoint: `uv run python main.py` serves the pipeline on 127.0.0.1."""

from __future__ import annotations

import uvicorn

from image_pipeline.server import app


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
