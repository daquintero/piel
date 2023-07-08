``sax``
=======

Implementation Principle
------------------------

The methodology of interconnection between photonics and electronics
design can be done in the time and frequency domain. However, one of the
most basic simulation implementations is determining how an electronic
system implements a photonic operation. This means, for a given mapping
between an electronic signal to a photonic one, how does the full
photonic system change?

This is where frequency domain solver tools like ``sax`` come into play
for photonics.

One pseudo electronic-photonic simulation currently available has been
demonstrated in `PhotonTorch
09_XOR_task_with_MZI <https://docs.photontorch.com/examples/09_XOR_task_with_MZI.html>`__.
We want to extend this type of functionality into a co-design between
electronic and photonic tools.

Pseudo-Static Active Models
-------------------------------
One of the main principles of ``sax`` is that we can create models that allow us to simulate the frequency-domain response of the input and output. In our case, as we want to simulate the interface of electronics and photonics, we want models that we can see how the input-output varies based on the electrical signal control provided. However, as ``sax`` is a frequency-domain solver, we can not just implement time-dependent active control.

The first implementation proposed in ``piel`` is a pseudo-static implementation of ``sax`` models, whose model evaluation is dependent on a set of controlled-phase parameters for a physical circuit. This can be then interconnected with the asynchronous electrical models in ``cocotb`` through some analog or digital signal-to-phase mapping.

Another aspect that we want to make sure we are modelling is the variability of each of our interconnected photonic and electronic components. Not all photonic electro-optic phase shifters are the same, neither are the electronic drivers, neither are the bends, or multi-mode waveguides, or so on. Each can be different.

GDSFactory is a functional implementation of circuit design and physical parameterised devices can be easily created.

Implementation
~~~~~~~~~~~~~~~
A ``sax`` circuit is a "component model function" such as the ones we define for our instances. This means that we can functionally evaluate variations in our parameters by just evaluating the ``sax`` circuit function with our component parameters. This evaluation tends to be in the order of `ms` whereas the circuit composition tends to be in a larger order of magnitude depending on complexity. In ``sax`` documentation words:

.. epigraph::
    The ``circuit`` function just creates a similar function as we created for the waveguide and the coupler, but instead of taking parameters directly it takes parameter dictionaries for each of the instances in the circuit. The keys in these parameter dictionaries should correspond to the keyword arguments of each individual subcomponent.




API
------

.. toctree::

    ../../../autoapi/piel/tools/sax/index.rst
