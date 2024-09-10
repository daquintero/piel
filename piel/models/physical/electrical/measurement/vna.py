import piel.experimental as pe


def E8364A(**kwargs) -> pe.types.VNA:
    return pe.types.VNA(name="E8364A", manufacturer="Agilent", model="E8364A", **kwargs)
