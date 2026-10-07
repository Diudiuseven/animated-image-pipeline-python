import numpy as np
from PIL import Image, ImageOps

from imagepipeline.stages.orientation import apply_orientation, exif_orientation


def _marker(h=2, w=4):
    a = np.zeros((h, w, 4), np.uint8)
    for y in range(h):
        for x in range(w):
            a[y, x] = [x * 40 + 10, y * 100 + 20, (x + y) * 30 + 5, 255]
    return a


def test_all_orientations_match_imageops():
    a = _marker()
    for o in range(1, 9):
        tmp = Image.fromarray(a, "RGBA")
        exif = Image.Exif()
        exif[274] = o
        tmp.info["exif"] = exif.tobytes()
        ref = np.asarray(ImageOps.exif_transpose(tmp))
        got = apply_orientation(a, o)
        assert np.array_equal(got, ref), f"orientation {o} mismatch"


def test_exif_orientation_reader():
    im = Image.new("RGB", (4, 4))
    assert exif_orientation(im) == 1

    im6 = Image.new("RGB", (4, 4))
    exif = Image.Exif()
    exif[274] = 6
    im6.info["exif"] = exif.tobytes()
    assert exif_orientation(im6) == 6

    # invalid value falls back to 1
    im9 = Image.new("RGB", (4, 4))
    exif = Image.Exif()
    exif[274] = 9
    im9.info["exif"] = exif.tobytes()
    assert exif_orientation(im9) == 1
