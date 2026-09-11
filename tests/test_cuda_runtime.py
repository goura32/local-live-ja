from local_live.asr import _cuda_library_paths


def test_cuda_library_path_helper_returns_existing_directories_only():
    paths = _cuda_library_paths()
    assert len(paths) == len(set(paths))
    assert all(path for path in paths)
