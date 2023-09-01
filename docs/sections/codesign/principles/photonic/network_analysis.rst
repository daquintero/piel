Network Analysis
------------------

Impedance Matrix
^^^^^^^^^^^^^^^^^^^

Derivation for a two-conductor TEM transmission line
'''''''''''''''''''''''''''''''''''''''''''''''''''''

.. math::

    \begin{equation}
        V = \int_{+}^{-} E \dot dl
    \end{equation}


.. math::

    \begin{equation}
        I = \oint_{C+} H \dot dl
    \end{equation}

.. math::

    \begin{equation}
        Z_0 = \frac{V}{I}
    \end{equation}

When

.. math::

    \begin{align}
        V_n = V_n^+ + V_n^- \\
        I_n = I_n^+ - I_n^-
    \end{align}


.. math::

    \begin{equation}
        [V] = [Z][I]
    \end{equation}

.. math::

    \begin{equation}
        \begin{bmatrix}
            V_0 \\
            V_1 \\
            \vdots \\
            V_N
        \end{bmatrix} =
        \begin{bmatrix}
            Z_{00} & Z_{01} & \ldots & Z_{0N} \\
            Z_{10} & Z_{11} &  & Z_{1N} \\
            \vdots & & & \vdots \\
            Z_{N0} & Z_{N1} & \ldots & Z_{NN}
        \end{bmatrix}
        \begin{bmatrix}
            I_0 \\
            I_1 \\
            \vdots \\
            I_N
        \end{bmatrix}
    \end{equation}


.. math::

    \begin{equation}
        [I] = [Y][V]
    \end{equation}

.. math::

    \begin{equation}
        \begin{bmatrix}
            I_0 \\
            I_1 \\
            \vdots \\
            I_N
        \end{bmatrix} =
        \begin{bmatrix}
            Y_{00} & Y_{01} & \ldots & Y_{0N} \\
            Y_{10} & Y_{11} &  & Y_{1N} \\
            \vdots & & & \vdots \\
            Y_{N0} & Y_{N1} & \ldots & Y_{NN}
        \end{bmatrix}
        \begin{bmatrix}
            V_0 \\
            V_1 \\
            \vdots \\
            V_N
        \end{bmatrix}
    \end{equation}


.. math::

    \begin{equation}
        [Y] = [Z]^{-1}
    \end{equation}


Scattering Matrix
^^^^^^^^^^^^^^^^^^^^^

.. math::

    \begin{equation}
        [V^-] = [S][V^+]
    \end{equation}

.. math::

    \begin{equation}
        \begin{bmatrix}
            V_0^- \\
            V_1^- \\
            \vdots \\
            V_N^-
        \end{bmatrix} =
        \begin{bmatrix}
            S_{00} & S_{01} & \ldots & S_{0N} \\
            S_{10} & S_{11} &  & S_{1N} \\
            \vdots & & & \vdots \\
            S_{N0} & S_{N1} & \ldots & S_{NN}
        \end{bmatrix}
        \begin{bmatrix}
            V_0^+ \\
            V_1^+ \\
            \vdots \\
            V_N^+
        \end{bmatrix}
    \end{equation}

.. math::

    \begin{equation}
        S_{ij} = \frac{V_i^-}{V_j^+} |_
    \end{equation}
