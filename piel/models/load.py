from typing import Any
from ..file_system import read_json
from ..types import PielBaseModel, PathTypes


def load_from_dict(model_dictionary: dict, type: Any) -> Any:
    """
    See limitations in https://github.com/pydantic/pydantic/issues/8084
    :param model_dictionary:
    :param model:
    :return:
    """
    # Validate this is a PielPydanticModel
    assert issubclass(type, PielBaseModel)

    # Validate the model
    type_instance = type.model_construct(**model_dictionary)

    return type_instance


def load_from_json(
    json_file: PathTypes,
    type: Any,
) -> Any:
    """
    This function will load the model from the given model instance.
    """
    # Read the json file
    model_dictionary = read_json(json_file)
    type_instance = load_from_dict(model_dictionary, type)
    return type_instance
