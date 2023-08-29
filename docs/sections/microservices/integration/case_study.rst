Case Study
==========

`GDSFactory <https://github.com/gdsfactory/gdsfactory>`__ is an
incredible tool for photonic, layout, simulation and design. However,
for digital design, the
`OpenROAD <https://github.com/The-OpenROAD-Project>`__ design process is
far more mature in terms of chip designs. We want to leverage the power
of both. It is possible to design any GDS-based layout in ``GDSFactory``
and it provides a much more amenable direct-editing tool for users.

What ``piel`` is interested in is enabling the co-design between
electronics and photonics. One aspect that is currently a large design
challenge is exploring the effect of parametric variations on one
component on a total photonic-electronic system performance. It is these
design questions that ``piel`` aims to answer and simulate.

There are multiple levels of design integration of the tools. A basic
set of interconnection functions is just a function that implements
reading and importing the layout of the OpenROAD process directly into a
GDSFactory environment.

The next level of simulation analysis is to understand the effect of the
total requirements of a particular system implementation. This is
already achieved through `TODO link example <>`__. Each individual
toolset interconnection mechanism is described on its corresponding
``piel/integration/<"tool1-tool2">.py`` source file.
