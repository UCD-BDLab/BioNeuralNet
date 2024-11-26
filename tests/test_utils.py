import pytest
from bioneuralnet.utils.file_helpers import find_files
from bioneuralnet.utils.path_utils import validate_paths


def test_find_files(tmp_path):
    # Setting temporary directory and files
    temp_dir = tmp_path / "sub"
    temp_dir.mkdir()
    (temp_dir / "file1.csv").write_text("col1,col2\n1,2")
    (temp_dir / "file2.csv").write_text("col1,col2\n3,4")
    (temp_dir / "file3.txt").write_text("Just a text file.")
    
    # Testing finding CSV files
    csv_files = find_files(str(temp_dir), "*.csv")
    assert len(csv_files) == 2, f"Expected 2 CSV files, found {len(csv_files)}"
    assert all(file.endswith(".csv") for file in csv_files), "All files should have .csv extension"
    
    # Testing finding TXT files
    txt_files = find_files(str(temp_dir), "*.txt")
    assert len(txt_files) == 1, f"Expected 1 TXT file, found {len(txt_files)}"
    assert txt_files[0].endswith(".txt"), "File should have .txt extension"


def test_validate_paths(tmp_path, caplog):
    # Creating temporary directories
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    log_dir = tmp_path / "logs"
    
    input_dir.mkdir()
    output_dir.mkdir()

    # log_dir is intentionally not created to test validation
    with pytest.raises(FileNotFoundError):
        validate_paths(str(input_dir), str(output_dir), str(log_dir))
    
    # Create log_dir and validate again
    log_dir.mkdir()
    try:
        validate_paths(str(input_dir), str(output_dir), str(log_dir))
    except FileNotFoundError:
        pytest.fail("validate_paths raised FileNotFoundError unexpectedly!")
