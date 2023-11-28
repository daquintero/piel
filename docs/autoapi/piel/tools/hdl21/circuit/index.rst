:py:mod:`piel.tools.hdl21.circuit`
==================================

.. py:module:: piel.tools.hdl21.circuit

.. autoapi-nested-parse::

   The way the construction of the SPICE models is implemented in ``piel`` is also microservice-esque. Larger
   circuits are constructed out of smaller building blocks. This means that simulation configurations and so on are
   constructed upon a provided initial circuit, and the SPICE directives are appended accordingly.

   As mentioned previously, ``piel`` creates circuits based on fundamental SPICE because this opens the possibility to
   large scale integration of these circuit models on different SPICE solvers, including proprietary ones as long as the
   SPICE circuit can be written to particular directories. However, due to the ease of circuit integration,
   and translation that ``hdl21`` provides, it's API can also be used to implement parametrised functionality,
   although it is easy to export the circuit as a raw SPICE directive after composition.

   This composition tends to be implemented in a `SubCircuit` hierarchical implementation, as this allows for more modularisation of the netlisted devices.

   Let's assume that we can get an extracted SPICE netlist of our circuit, that includes all nodes, and component
   circuit definitions. This could then be simulated accordingly for the whole circuit between inputs and outputs. This
   would have to be constructed out of component models and a provided netlist in a similar fashion to ``SAX``.



