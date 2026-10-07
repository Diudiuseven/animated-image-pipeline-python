import io

from PIL import Image, ImageDraw

from imagepipeline.config import Settings
from imagepipeline.models import Format
from imagepipeline.stages import compositor as comp
from imagepipeline.stages import decoder as dec


def _anim_with_disposal(disposal):
    W = H = 32
    f0 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(f0).rectangle([0, 0, 15, 31], fill=(255, 0, 0, 255))
    f1 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(f1).rectangle([16, 0, 31, 31], fill=(0, 255, 0, 255))
    buf = io.BytesIO()
    f0.save(
        buf,
        format="GIF",
        save_all=True,
        append_images=[f1],
        duration=[100, 200],
        loop=0,
        disposal=[disposal, disposal],
    )
    return buf.getvalue()


def test_frame_count_durations_loop_preserved():
    data = _anim_with_disposal(2)
    d = dec.decode(data, Format.GIF, Settings())
    doc = comp.composite(d)
    assert doc.frame_count == 2
    assert [f.duration_ms for f in doc.frames] == [100, 200]
    assert doc.loop == 0
    assert doc.size == (32, 32)


def test_dispose_to_background_clears_previous_frame():
    data = _anim_with_disposal(2)  # restore to background
    d = dec.decode(data, Format.GIF, Settings())
    doc = comp.composite(d)
    second = doc.frames[1].pixels
    # Left half was red in frame 0 and must be transparent after disposal=2.
    assert second[16, 0, 3].item() == 0
    # Right half is green and opaque.
    assert tuple(second[16, 24][:3]) == (0, 255, 0)


def test_do_not_dispose_keeps_previous_frame():
    data = _anim_with_disposal(1)  # leave canvas in place
    d = dec.decode(data, Format.GIF, Settings())
    doc = comp.composite(d)
    second = doc.frames[1].pixels
    # Left half red must remain visible in frame 1.
    assert tuple(second[16, 4][:3]) == (255, 0, 0)
    assert second[16, 4, 3].item() == 255


def test_orientation_applied_to_every_frame():
    # Build a 40x20 2-frame gif, tag EXIF orientation=6 (rotate CW -> 20x40)
    f0 = Image.new("RGBA", (40, 20), (10, 20, 30, 255))
    f1 = Image.new("RGBA", (40, 20), (40, 50, 60, 255))
    buf = io.BytesIO()
    f0.save(buf, format="GIF", save_all=True, append_images=[f1], duration=100)
    data = buf.getvalue()
    d = dec.decode(data, Format.GIF, Settings())
    # Simulate orientation metadata on the decoded source.
    d2 = type(d)(**{**d.__dict__, "orientation": 6, "canvas_size": (20, 40)})
    doc = comp.composite(d2)
    assert doc.size == (20, 40)
    for f in doc.frames:
        assert f.size == (20, 40)
