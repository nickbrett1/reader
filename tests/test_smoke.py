"""Smoke test: the src-layout package installs and imports cleanly."""


def test_package_imports():
    import reader

    assert reader.__version__
