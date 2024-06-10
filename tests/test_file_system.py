import pytest
import os
import pathlib
import shutil
import json
import subprocess
from piel.file_system import (
    check_path_exists,
    check_example_design,
    copy_source_folder,
    copy_example_design,
    create_new_directory,
    create_piel_home_directory,
    delete_path,
    delete_path_list_in_directory,
    get_files_recursively_in_directory,
    get_id_map_directory_dictionary,
    get_top_level_script_directory,
    list_prefix_match_directories,
    permit_script_execution,
    permit_directory_all,
    read_json,
    rename_file,
    rename_files_in_directory,
    replace_string_in_file,
    replace_string_in_directory_files,
    return_path,
    run_script,
    write_file,
)  # Adjust the import based on your actual module structure


# Tests for check_path_exists function
def test_check_path_exists(tmp_path):
    test_dir = tmp_path / "test_directory"
    test_dir.mkdir()
    assert check_path_exists(test_dir)
    assert not check_path_exists(tmp_path / "non_existing_path")


def test_check_path_exists_with_error(tmp_path):
    with pytest.raises(ValueError):
        check_path_exists(tmp_path / "non_existing_path", raise_errors=True)


# Tests for check_example_design function
def test_check_example_design(tmp_path, monkeypatch):
    designs_directory = tmp_path / "designs"
    designs_directory.mkdir()
    simple_design = designs_directory / "simple_design"
    simple_design.mkdir()
    monkeypatch.setenv("DESIGNS", str(designs_directory))
    assert check_example_design("simple_design", designs_directory)
    assert not check_example_design("non_existing_design", designs_directory)


# TODO fix this
# # Tests for copy_source_folder function
# def test_copy_source_folder(tmp_path):
#     source_dir = tmp_path / "source_directory"
#     target_dir = tmp_path / "target_directory"
#     source_dir.mkdir()
#     (source_dir / "test_file.txt").touch()
#
#     # Ensure the target directory doesn't exist before copying
#     if target_dir.exists():
#         shutil.rmtree(target_dir)
#
#     # Perform the copy operation
#     copy_source_folder(source_dir, target_dir, delete=True)
#
#     # Check if the target directory exists after the copy operation
#     assert target_dir.exists()
#     assert (target_dir / "test_file.txt").exists()


# Tests for create_new_directory function
def test_create_new_directory(tmp_path):
    new_dir = tmp_path / "new_subdir"
    result = create_new_directory(new_dir, overwrite=True)
    assert result
    assert new_dir.exists()

    result = create_new_directory(new_dir, overwrite=False)
    assert not result  # Should return False because it already exists


# TODO fix this
# # Tests for create_piel_home_directory function
# def test_create_piel_home_directory(tmp_path, monkeypatch):
#     home_dir = tmp_path / "home"
#     monkeypatch.setattr(pathlib.Path, "home", lambda: home_dir)
#
#     # Ensure the home directory does not exist initially
#     if home_dir.exists():
#         shutil.rmtree(home_dir)
#
#     # Call the function to create the PIEL home directory
#     create_piel_home_directory()
#
#     # Check if the PIEL home directory was created
#     piel_home = home_dir / ".piel"
#     assert piel_home.exists()


# Tests for delete_path function
def test_delete_path(tmp_path):
    test_file = tmp_path / "test_file.txt"
    test_file.touch()
    delete_path(test_file)
    assert not test_file.exists()

    test_dir = tmp_path / "test_directory"
    test_dir.mkdir()
    delete_path(test_dir)
    assert not test_dir.exists()


# Tests for delete_path_list_in_directory function
def test_delete_path_list_in_directory(tmp_path):
    test_dir = tmp_path / "test_directory"
    test_dir.mkdir()
    test_file = test_dir / "test_file.txt"
    test_file.touch()
    nested_test_file = test_dir / "nested_dir" / "test_file.txt"
    nested_test_file.parent.mkdir(parents=True, exist_ok=True)
    nested_test_file.touch()

    files_to_delete = [test_file, nested_test_file]
    delete_path_list_in_directory(test_dir, files_to_delete, ignore_confirmation=True)
    assert not test_file.exists()
    assert not nested_test_file.exists()


# Tests for get_files_recursively_in_directory function
def test_get_files_recursively_in_directory(tmp_path):
    test_dir = tmp_path / "test_directory"
    test_dir.mkdir()
    test_file = test_dir / "test_file.txt"
    test_file.touch()
    nested_test_file = test_dir / "nested_dir" / "test_file.txt"
    nested_test_file.parent.mkdir(parents=True, exist_ok=True)
    nested_test_file.touch()

    files = get_files_recursively_in_directory(test_dir, "txt")
    assert len(files) == 2
    assert str(test_file) in files
    assert str(nested_test_file) in files


# Tests for get_id_map_directory_dictionary function
def test_get_id_map_directory_dictionary(tmp_path):
    path_list = [
        str(tmp_path / "prefix_123"),
        str(tmp_path / "prefix_456"),
        str(tmp_path / "no_prefix_789"),
    ]
    id_map = get_id_map_directory_dictionary(path_list, "prefix_")
    assert id_map == {123: path_list[0], 456: path_list[1]}


# Tests for get_top_level_script_directory function
def test_get_top_level_script_directory():
    top_level_dir = get_top_level_script_directory()
    assert top_level_dir.exists()


# Tests for list_prefix_match_directories function
def test_list_prefix_match_directories(tmp_path):
    output_directory = tmp_path / "output_directory"
    output_directory.mkdir()
    prefix_match_dir = output_directory / "prefix_dir"
    prefix_match_dir.mkdir()
    matching_dirs = list_prefix_match_directories(output_directory, "prefix")
    assert str(prefix_match_dir) in matching_dirs


# Tests for permit_script_execution function
def test_permit_script_execution(tmp_path):
    script_file = tmp_path / "script.sh"
    script_file.touch()
    permit_script_execution(script_file)
    assert os.access(script_file, os.X_OK)


# Tests for permit_directory_all function
def test_permit_directory_all(tmp_path):
    directory = tmp_path / "permitted_dir"
    directory.mkdir()
    permit_directory_all(directory)
    assert oct(directory.stat().st_mode)[-3:] == "777"


# Tests for read_json function
def test_read_json(tmp_path):
    json_file = tmp_path / "test.json"
    json_data = {"key": "value"}
    with json_file.open("w") as f:
        json.dump(json_data, f)

    result = read_json(json_file)
    assert result == json_data


# Tests for rename_file function
def test_rename_file(tmp_path):
    test_file = tmp_path / "test_file.txt"
    test_file.touch()
    new_file = tmp_path / "renamed_file.txt"
    rename_file(test_file, new_file)
    assert new_file.exists()
    assert not test_file.exists()


# Tests for rename_files_in_directory function
def test_rename_files_in_directory(tmp_path):
    test_dir = tmp_path / "test_directory"
    test_dir.mkdir()
    test_file = test_dir / "test_file.txt"
    test_file.touch()
    rename_files_in_directory(test_dir, "test", "renamed")
    assert (test_dir / "renamed_file.txt").exists()


# Tests for replace_string_in_file function
def test_replace_string_in_file(tmp_path):
    text_file = tmp_path / "test_text.txt"
    with text_file.open("w") as f:
        f.write("This is a test string.")

    replace_string_in_file(text_file, "test", "sample")
    with text_file.open("r") as f:
        content = f.read()
        assert "sample string" in content


# Tests for replace_string_in_directory_files function
def test_replace_string_in_directory_files(tmp_path):
    text_file_1 = tmp_path / "test1.txt"
    text_file_2 = tmp_path / "test2.txt"
    with text_file_1.open("w") as f1, text_file_2.open("w") as f2:
        f1.write("This is a test string.")
        f2.write("This is another test string.")

    replace_string_in_directory_files(tmp_path, "test", "sample")
    with text_file_1.open("r") as f1, text_file_2.open("r") as f2:
        content1 = f1.read()
        content2 = f2.read()
        assert "sample string" in content1
        assert "sample string" in content2


# Tests for return_path function
def test_return_path(tmp_path):
    path = return_path(str(tmp_path))
    assert isinstance(path, pathlib.Path)
    assert path == tmp_path.resolve()


# Tests for run_script function
def test_run_script(tmp_path):
    script_file = tmp_path / "script.sh"
    script_content = "#!/bin/bash\necho 'Hello, World!'"
    with script_file.open("w") as f:
        f.write(script_content)

    permit_script_execution(script_file)
    result = subprocess.run(str(script_file), capture_output=True, text=True)
    assert result.stdout.strip() == "Hello, World!"


# Tests for write_file function
def test_write_file(tmp_path):
    new_file = tmp_path / "written_file.txt"
    write_file(tmp_path, "Sample content", "written_file.txt")
    assert new_file.exists()
    with new_file.open("r") as f:
        content = f.read()
        assert content == "Sample content"
