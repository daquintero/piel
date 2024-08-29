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

Converting dBm to Watts
''''''''''''''''''''''''

When working with power in dBm, converting it to Watts is a common requirement. The equation to perform this conversion is:

.. math::

   P_{\text{Watt}} = 10^{\left(\frac{P_{\text{dBm}}}{10}\right)} \times 10^{-3}


Example Usage::

    >>> # Convert 30 dBm to Watts
    >>> piel.units.dBm2watt(30)

Converting Watts to dBm
''''''''''''''''''''''''

The inverse operation, converting Watts to dBm, uses the following equation:

.. math::

   P_{\text{dBm}} = 10 \times \log_{10}\left(\frac{P_{\text{Watt}}}{10^{-3}}\right)


Example Usage::

    >>> # Convert 1 Watt to dBm
    >>> piel.units.watt2dBm(1)

Considering Network Impedance
''''''''''''''''''''''''''''''

In RF systems, network impedance plays a critical role. It is common to calculate voltage levels like Vrms (root mean square voltage) and Vpp (peak-to-peak voltage) for a specific impedance, typically 50 ohms. This is particularly important for matching loads in waveguides and other transmission lines.

Converting dBm to Peak-to-Peak Voltage (Vpp)
''''''''''''''''''''''''''''''''''''''''''''

To find the peak-to-peak voltage for a given power in dBm and a specified impedance (e.g., 50 Ω), we use the following conversions:

1. **dBm to Watts:**

   .. math::

      P_{\text{Watt}} = 10^{\left(\frac{P_{\text{dBm}}}{10}\right)} \times 10^{-3}

2. **Watts to Vrms:**

   .. math::

      V_{\text{rms}} = \sqrt{P_{\text{Watt}} \times Z}

3. **Vrms to Vpp:**

   .. math::

      V_{\text{pp}} = V_{\text{rms}} \times \sqrt{2} \times 2

Example Usage::

    >>> # Convert 1 dBm to Vpp with a 50 Ω impedance
    >>> piel.units.dBm2vpp(1, impedance=50)

    >>> # Convert 10 dBm to Vpp with default impedance (50 Ω)
    >>> piel.units.dBm2vpp(10)

Converting Vpp to dBm
''''''''''''''''''''''

Conversely, if you know the peak-to-peak voltage, you may want to determine the corresponding power in dBm. The steps involved are:

1. **Vpp to Vrms:**

   .. math::

      V_{\text{rms}} = \frac{V_{\text{pp}}}{\sqrt{2} \times 2}

2. **Vrms to Watts:**

   .. math::

      P_{\text{Watt}} = \frac{V_{\text{rms}}^2}{Z}

3. **Watts to dBm:**

   .. math::

      P_{\text{dBm}} = 10 \times \log_{10}\left(\frac{P_{\text{Watt}}}{10^{-3}}\right)

Example Usage::

    >>> # Convert 2 Vpp to dBm with a 50 Ω impedance
    >>> piel.units.vpp2dBm(2)


