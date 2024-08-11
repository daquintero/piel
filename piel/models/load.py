from typing import Any
from ..file_system import read_json
from ..types import PielBaseModel, PathTypes


def load_from_dict(model_dictionary: dict, model: Any) -> Any:
    """
    See limitations in https://github.com/pydantic/pydantic/issues/8084
    :param model_dictionary:
    :param model:
    :return:
    """
    # Validate this is a PielPydanticModel
    assert issubclass(model, PielBaseModel)

    # Validate the model
    model_instance = model.model_construct(**model_dictionary)

    return model_instance


def load_from_json(
    json_file: PathTypes,
    model: Any,
) -> Any:
    """
    This function will load the model from the given model instance.
    """
    # Read the json file
    model_dictionary = read_json(json_file)
    model_instance = load_from_dict(model_dictionary, model)
    return model_instance
