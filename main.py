"""Local development entry point: serves the image pipeline HTTP API."""

import uvicorn

from imagepipeline.api import app


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
