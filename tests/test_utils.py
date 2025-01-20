import pytest
from bioneuralnet.utils.path_utils import validate_paths


def test_validate_paths(tmp_path, caplog):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    log_dir = tmp_path / "logs"

    input_dir.mkdir()
    output_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        validate_paths(str(input_dir), str(output_dir), str(log_dir))

    log_dir.mkdir()
    try:
        validate_paths(str(input_dir), str(output_dir), str(log_dir))
    except FileNotFoundError:
        pytest.fail("validate_paths raised FileNotFoundError unexpectedly!")
