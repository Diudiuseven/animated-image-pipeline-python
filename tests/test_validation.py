import pytest

from imagepipeline.config import Settings
from imagepipeline.exceptions import InvalidInputError, InvalidRequestError
from imagepipeline.models import Format, ResizeMode, TransformRequest
from imagepipeline.stages import validation as v


def req(**kw):
    kw.setdefault("target_format", Format.PNG)
    return TransformRequest(**kw)


def test_signatures():
    assert v.sniff_signature(b"\x89PNG\r\n\x1a\nrest") is Format.PNG
    assert v.sniff_signature(b"\xff\xd8\xff\xe0rest") is Format.JPEG
    assert v.sniff_signature(b"GIF89a...") is Format.GIF
    assert v.sniff_signature(b"RIFF\x00\x00\x00\x00WEBPrest") is Format.WEBP
    with pytest.raises(InvalidInputError):
        v.sniff_signature(b"not an image at all")
    with pytest.raises(InvalidInputError):
        v.sniff_signature(b"RIFF\x00\x00\x00\x00AVI rest")


def test_empty_and_oversized():
    s = Settings()
    with pytest.raises(InvalidInputError):
        v.validate_source_size(b"", s)
    with pytest.raises(InvalidInputError):
        v.validate_source_size(b"x" * (s.limits.max_source_bytes + 1), s)


def test_resize_request_rules():
    s = Settings()
    v.validate_request(req(resize_mode=ResizeMode.NONE), s)
    with pytest.raises(InvalidRequestError):
        v.validate_request(req(resize_mode=ResizeMode.NONE, width=10), s)
    with pytest.raises(InvalidRequestError):
        v.validate_request(req(resize_mode=ResizeMode.FIT), s)
    v.validate_request(req(resize_mode=ResizeMode.FIT, width=10), s)
    with pytest.raises(InvalidRequestError):
        v.validate_request(req(resize_mode=ResizeMode.COVER, width=10), s)
    v.validate_request(req(resize_mode=ResizeMode.COVER, width=10, height=20), s)
    with pytest.raises(InvalidRequestError):
        v.validate_request(
            req(resize_mode=ResizeMode.FIT, width=s.limits.max_target_dimension + 1),
            s,
        )
