from ...file_system import return_path


def save(fig, **kwargs):
    """
    This function is a generic implementation of the savefig functionality in matplotlib,
    and is used as a wrapper on all
    :return:
    """
    # TODO implement verification to guarantee path functionality always available
    path = kwargs["path"]
    path = return_path(path)
    fig.savefig(path)
