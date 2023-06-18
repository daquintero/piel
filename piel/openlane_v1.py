from .defaults import example_open_lane_configuration


def write_openlane_configuration(project_directory=None, configuration=dict()):
    with open(target_directory + "config.json", "w") as write_file:
        json.dump(configuration, write_file, indent=4)


__all__ = ["write_openlane_configuration"]
