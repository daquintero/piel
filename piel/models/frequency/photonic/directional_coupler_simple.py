"""
Translated from https://github.com/flaport/sax or https://github.com/flaport/photontorch/tree/master
"""


def directional_coupler(coupling=0.5):
    import sax

    kappa = coupling**0.5
    tau = (1 - coupling) ** 0.5
    sdict = sax.reciprocal(
        {
            ("port0", "port1"): tau,
            ("port0", "port2"): 1j * kappa,
            ("port1", "port3"): 1j * kappa,
            ("port2", "port3"): tau,
        }
    )
    return sdict
