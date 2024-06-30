It might be desired to use a given electronic component experimental measurement to model how this would perform when used in an integrated photonic-electronic system. There's a few references to be aware of:

.. list-table::
   :header-rows: 1

   * - Tool
     - Usage
     - `scikit-rf <https://scikit-rf.readthedocs.io/en/latest/index.html>`_
     - Conversion to a standard RF format directly from toolset through `IO methods <https://scikit-rf.readthedocs.io/en/latest/api/io/index.html#io>`.
     - TouchStone File Format
     - Industry standard for RF vector network analysis `formats <https://en.wikipedia.org/wiki/Touchstone_file>`.

Hence, it makes sense to use any of these tools to integrate with the experimental results. We can translate directly from them in order to model our photonic systems accordingly.
