Modelling Techniques
---------------------

One of the main complexities of designing a device is understanding how to use a device model technique in order to understand a meaningful application of such device.

Let's explore how this relates to the tools and methodology of our device models:

.. list-table::
   :header-rows: 1

   * - Technique
     - Tools
     - Limitations
     - Usage
   * - Transient
     - SPICE
     - Signal discretization limited by period observed. Large systems modelling may involve a long computation.
     - Switching, dynamics, device changes.
   * - Network Transmission
     - S-Parameter / Transfer-Matrix-Model
     - If we want to model the transmission as a function of frequency. Fast computation as matrix multiplication.
     - Linear network-transmission. Nonlinear can be implemented but more involved.
     - Frequency-Domain

Converting between time-domain and frequency-domain signals requires some understanding of the type of modelling implemented. One way to understand translating between a time-domain signal is through Fourier transforms of course.

We can create an analogy starting from basic time-domain models. What is a transient SPICE simulation?
Let's assume it's for a non-periodic input such as a single switch state. This could be a signal that goes from 0V to 1V, 10%-90% final state with a duration of :math:`\tau` seconds. Let's also assume the time it takes to go from 90%-10% final state is :math:`\tau`

Let's consider a few logical situations here:

* We cannot apply a turn-on-off signal slower than :math:`\tau` if we want the full switch state to complete switching.
* We could apply the turn off signal immediately after turn-on, with the off-on-off transition lasting :math:`2\tau` seconds. We could repeat this periodically with a period of :math:`\frac{1}{2\tau}` Hertz. We can use this to model the power transmitted through the signal. With this case, the medium must be able to transfer the rising edge of a signal with a frequency of :math:`\frac{1}{2\tau}` . This means that the medium must support this frequency.
* We need to understand the time-period and the frequency of interest in order to determine how to correctly apply the Fourier Transform in a meaningful way. This may not exactly match to the way we talk about bandwidths in a transmission line context, for example.

.. math::

    \begin{equation}
        \mathcal{F}\{f(t)\} = F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-j \omega t} \, dt
    \end{equation}

.. math::

    \begin{equation}
        \mathcal{F}^{-1}\{F(\omega)\} = f(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} F(\omega) e^{j \omega t} \, d\omega
    \end{equation}


Resonant systems will have a minimum bandwidth of operation that is non-zero, unlike non-resonant DC-wideband systems. As such, when we are modelling frequency-domain transmission models, we are looking at it in terms of a specific frequency or specific frequency range.

.. TODO do time-domain to SPICE conversion. Finish understanding this.

1. Smith, Steven W. The Scientist and Engineer's Guide to Digital Signal Processing. 2nd ed., California Technical Publishing, 1999. www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch10.pdf. Accessed 2 July 2024.
2. Rose-Hulman Institute of Technology. "Fourier Transform Tables." Class Handouts, www.rose-hulman.edu/class/ee/yoder/ece380/Handouts/Fourier%20Transform%20Tables%20w.pdf. Accessed 2 July 2024.
