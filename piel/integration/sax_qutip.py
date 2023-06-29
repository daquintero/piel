import sax
import qutip  # NOQA : F401


def sax_s_parameters_to_qutip_unitary(sax_model=sax.Model):
    """
    This function converts the calculated S-parameters into a standard Unitary matrix topology so that the shape and
    dimensions of the matrix can be observed.

    A ``sax`` S-parameter dictionary is provided as a dictionary of tuples with (port0, port1) as the key. This
    determines the direction of the scattering relationship. It means that the number of terms in an S-parameter
    matrix is the number of ports squared.

    In order to generalise, this function returns both the S-parameter matrices and the indexing ports based on the
    amount provided. In terms of computational speed, we definitely would like this function to be algorithmically
    very fast. For now, I will write a simple python implementation and optimise in the future.

    A S-Parameter matrix in the form is returned:

    ..math::

        S = \\begin{bmatrix}
            S_{11} & S_{12} & S_{13} & S_{14} \\
            S_{21} & S_{22} & S_{23} & S_{24} \\
            S_{31} & S_{32} & S_{33} & S_{34} \\
            S_{41} & S_{42} & S_{43} & S_{44} \\
        \\end{bmatrix}

    From this stage we can implement a ``QObj`` matrix accordingly and perform simulations accordingly. https://qutip.org/docs/latest/guide/qip/qip-basics.html#unitaries
    """
    pass
