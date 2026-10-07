"""Byte-for-byte determinism across repeats, order, concurrency and restart."""

import subprocess
import sys
import textwrap

from imagepipeline.config import Settings
from imagepipeline.models import Format, ResizeMode, TransformRequest
from imagepipeline.pipeline import process
from imagepipeline.samples import generate_sample


def _cases():
    gif, _ = generate_sample("animated_transparent")
    webp, _ = generate_sample("animated_webp")
    png, _ = generate_sample("static_png")
    return [
        (gif, TransformRequest(Format.WEBP, lossless=True)),
        (gif, TransformRequest(Format.GIF)),
        (webp, TransformRequest(Format.GIF)),
        (webp, TransformRequest(Format.WEBP, quality=80)),
        (png, TransformRequest(Format.PNG, ResizeMode.FIT, width=20, height=20)),
        (png, TransformRequest(Format.JPEG, quality=72)),
    ]


def test_repeat_and_order_determinism():
    s = Settings()
    for data, req in _cases():
        first = process(data, req, s, "first").data
        for tag in ("second", "third"):
            assert process(data, req, s, tag).data == first


def test_determinism_across_restart():
    # A fresh interpreter simulates a process restart: encode once here, then
    # have two brand-new Python processes encode the same inputs and compare.
    s = Settings()
    here = [process(d, r, s, "here").data for d, r in _cases()]

    script = textwrap.dedent(
        """
        import base64, pickle, sys
        from imagepipeline.config import Settings
        from imagepipeline.pipeline import process
        from imagepipeline.samples import generate_sample
        data, req = pickle.loads(base64.b64decode(sys.stdin.buffer.read()))
        out = process(data, req, Settings(), "child")
        sys.stdout.buffer.write(base64.b64encode(out.data))
        """
    )
    import base64
    import os
    import pickle

    # Ensure the child interpreter can import the package regardless of cwd.
    env = dict(os.environ)
    import imagepipeline

    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(imagepipeline.__file__)))
    env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")

    for i, (data, req) in enumerate(_cases()):
        payload = base64.b64encode(pickle.dumps((data, req)))
        outs = []
        for _ in range(2):
            proc = subprocess.run(
                [sys.executable, "-c", script],
                input=payload,
                capture_output=True,
                check=True,
                env=env,
            )
            outs.append(base64.b64decode(proc.stdout))
        assert outs[0] == outs[1], f"cross-process mismatch case {i}"
        assert outs[0] == here[i], f"restart mismatch case {i}"
