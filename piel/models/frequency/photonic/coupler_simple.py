"""
Translated from https://github.com/flaport/sax or https://github.com/flaport/photontorch/tree/master
"""


def coupler(coupling=0.5):
    import sax

    kappa = coupling**0.5
    tau = (1 - coupling) ** 0.5
    sdict = sax.reciprocal(
        {
            ("in0", "out0"): tau,
            ("in0", "out1"): 1j * kappa,
            ("in1", "out0"): 1j * kappa,
            ("in1", "out1"): tau,
        }
    )
    return sdict
