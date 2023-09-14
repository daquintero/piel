Signal Analysis Basics
-----------------------

It is important that we understand fundamental signal analysis principles of how we commonly speak of electromagnetic waves.

Some references about this are *On-wafer microwave measurements and de-embedding* by Errikos Lourandakis.

When we discuss radio-frequency and photonic signal testing and analysis, we commonly talk in terms of power ratio.

.. math::

    \begin{equation}
        V(t) = V_0 e^{j \omega t} = V_0 [cos(\omega t) + j sin(\omega t)]
    \end{equation}

We commonly discuss our photonic circuit component loss in terms of how many :math:`dB` our optical signal is losing across a component. This is because, unlike low-frequency waves, the dynamics of energy transfer a bit more complicated in the radio-frequency regime.

.. math::

    \begin{equation}
        \text{Power Ratio}(dB) = 10 log_{10} \left( \frac{P_2}{P_1} \right)
    \end{equation}


We can put this in terms of an electrical voltage for a RF signal in relation to the impedance across the network component.
