Digital Design Metrics
--------------------------

Read further in *Digital Integrated Electronics* by Jan Rabaey.

Say we have a digital switch gate. We apply a transition signal from low to high. The time in between 50% of the input signal and 50% of the output signal is called the *propagation delay*. However, rising (:math:`t_{pLH}`) and falling (:math:`t_{pHL}`) propagation delays tend to be different in physical components.

We define the propagation delay of the gate as the average of the two propagation delays:

.. math::

    \begin{equation}
    t_p = \frac{t_{pLH} + t_{pHL}}{2}
    \end{equation}

TODO add figure.

The rise and fall times are a function of the strength of the driving gate, the load gate capacitance which is related to the fanout, and the resistance of the interconnect. In CMOS, there is a direct relationship between the gate drive strength and gate capacitance.
