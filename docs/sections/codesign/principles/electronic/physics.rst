Device Physics
--------------------------

Ideal 1D `pn` junction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A `pn` junction is a building block for most electronic components, such as diodes and transistors.

Let's discuss what doping is. We have some bulk silicon. In a perfect pure silicon crystal lattice, the six valence electrons of each silicon atom form covalent bonds with the valence electrons of its four nearest neighbors. This is the intrinsic state of silicon. The intrinsic electron carrier concentration of this semiconductor is defined by :math:`n_i`. However, when we dope silicon, normally at high temperatures with some other material, the lattice properties change and we change the number of free electrons in the lattice.

When we `p` dope some silicon, we tend to do this with materials that have one less valence electron than silicon, such as boron. This means that when we dope silicon with boron, we create a hole in the lattice.

When we `n` dope some silicon, we tend to do this with materials that have one more valence electron than silicon, such as phosphorus or arsenic. This means that when we `n` dope silicon, we create an extra electron in the lattice.

If we put a `p` doped silicon crystal next to an `n` doped silicon crystal, we create a `pn` junction. The electron concentration gradient is very large in between these regions. We describe the free electron concentration, the donor concentration, as :math:`N_D`. We describe the hole concentration, the acceptor concentration, as :math:`N_A`. Under zero voltage bias, the depletion region voltage defined by :math:`\phi_0` is a function of the temperature, the intrinsic carrier concentration, and the donor and acceptor concentrations.

.. math::

    \begin{equation}
    \phi_0 = \phi_T \ln \left( \frac{N_A N_D}{n_i^2} \right)
    \end{equation}

The thermal voltage is defined by :math:`\phi_T` and is a function of the temperature :math:`T`, the charge of an electron :math:`q`, and the Boltzmann constant :math:`k`.

.. math::

    \begin{equation}
    \phi_T = \frac{kT}{q}
    \end{equation}


Depletion Region vs Voltage Bias
''''''''''''''''''''''''''''''''''

.. math::

    \begin{equation}
    Q_j = A_D \sqrt{\left ( 2 \epsilon_{si} q \frac{N_A N_D}{N_A + N_D}\right ) (\phi_0 - V_D)}
    \end{equation}


.. math::

    \begin{equation}
    W_j = W_2 - W_1 = \sqrt{\left( \frac{2 \epsilon_si}{q} \frac{N_A + N_D}{N_A N_D} \right) (\phi_0 - V_D) }
    \end{equation}


.. math::

    \begin{equation}
    E_j = \sqrt{\left( \frac{2q}{\epsilon_{si}} \frac{N_A N_D}{N_A + N_D} \right) (\phi_0 - V_D)}
    \end{equation}

.. math::

    \begin{equation}
    C_j = \frac{dQ_j}{dV_D} = A_D \sqrt{\left( \frac{\epsilon_{si}q}{2} \frac{N_A N_D}{N_A + N_D} \right) (\phi_0 - V_D)^{-1}} = \frac{C_{j0}}{\sqrt{1 - \frac{V_D}{\phi_0}}}
    \end{equation}


Circuit Models
'''''''''''''''

.. math::

    \begin{equation}
    I_D = I_S (e^{\frac{V_D}{n \phi_T}} - 1)
    \end{equation}


.. math::

    \begin{equation}
    C_D = \frac{C_{j0}}{(1 - \frac{V_D}{\phi_0}) ^ m} + \frac{\tau_T I_S}{\phi_T} e^{\frac{V_D}{n \phi_T}}
    \end{equation}


Static MOS Transistor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

    \begin{equation}
    W_D = \frac{2 \epsilon_{si} \phi}{q N_A}
    \end{equation}


.. math::

    \begin{equation}
    Q_D = \sqrt{2 q N_A \epsilon_{si} \phi}
    \end{equation}


.. math::

    \begin{equation}
    \phi_F = - \phi_T ln(\frac{N_A}{n_i})
    \end{equation}


.. math::

    \begin{equation}
    V_T = V_{T0} + \gamma \left ( \sqrt{|-2 \phi_F + V_{SB}|} - \sqrt{|-2 \phi_F|} \right )
    \end{equation}


.. math::

    \begin{equation}
    C_{ox} = \frac{\epsilon_{ox}}{t_{ox}}
    \end{equation}


.. math::

    \begin{equation}
    I_D = -\mathcal{v}_n(x) Q_i(x) W
    \end{equation}

.. math::

    \begin{equation}
    mathcal{v}_n(x) = - \mu_n \zeta(x) = \mu_n \frac{dV}{dx}
    \end{equation}

.. math::

    \begin{align}
    I_D dx = \mu_n C_{ox} W (V_{GS} - V - V_T) dV \\
    I_D = k_n^' \frac{W}{L} \left [ (V_{GS} - V_T) V_{DS} - \frac{V_{DS}^2}{2} \right ]
    \end{align}

.. math::

    \begin{align}
    k_n^' = \mu_n C_{ox} = \frac{\mu_n \epsilon_{ox}}{t_{ox}}
    \end{align}

NMOS Switch Model
'''''''''''''''''''

.. math::

    \begin{equation}
    R_{eq} = \frac{1}{t_2 - t_1} \int_{t_1}^{t_2} R_{on}(t) dt = \frac{1}{t_2 - t_1} \int_{t_1}^{t_2} \frac{V_{DS}(t)}{I_D(t)} dt \approx \frac{1}{2}(R_{on}(t_1) + R_{on}(t_2))
    \end{equation}


.. math::

    \begin{equation}
    R_{S,D} = \frac{L_{S,D}}{W} R_{\square} + R_C
    \end{equation}

TODO finish modelling equations
