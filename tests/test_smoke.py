def test_skeleton_imports():
    import imagepipeline

    assert imagepipeline.Format.PNG.value == "png"
