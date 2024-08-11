from ...file_system import return_path


def save(fig, **kwargs):
    """
    This function is a generic implementation of the savefig functionality in matplotlib,
    and is used as a wrapper on all
    :return:
    """
    # Save the figure if 'path' is provided in kwargs
    if "path" in kwargs and kwargs["path"]:
        # TODO implement verification to guarantee path functionality always available
        path = kwargs["path"]
        path = return_path(path)
        fig.savefig(path)
        assert path.exists()
        print(f"Figure saved at: {str(path)}")

    if "paths" in kwargs and kwargs["paths"]:
        # TODO implement verification to guarantee path functionality always available
        path_list = kwargs["paths"]
        for path_i in path_list:
            path_i = return_path(path)
            fig.savefig(path_i)
            assert path_i.exists()
            print(f"Figure saved at: {str(path_i)}")

    return None
