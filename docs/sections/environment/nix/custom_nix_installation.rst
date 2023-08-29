Crating your own ``nix`` Installation
-------------------------------------

You might have your own set of tools that you are using alongside ``piel``. One common complexity in using multiple open-source tools is distributing them and guaranteeing that they run consistently between different machines.

We follow some of the principles in the `Setting up the OpenLane Nix Cache page <https://openlane2.readthedocs.io/en/latest/contributors/updating_tools.html#setting-up-the-openlane-nix-cache>`__ .

Relevant tools:

-  `cachix <https://docs.cachix.org/getting-started>`__ Binary Cache platform for open source and business.

You might want to clone only the maintained ``nix`` configuration of the ``openlane2`` design flow.

Find out `what is a git sparse-checkout <https://stackoverflow.com/questions/47541033/sparse-checkouts-how-does-it-works>`__

What we would like to do is have a version controlled installation that tracks the version controlled `nix` configuration in ``openlane 2``.

.. code-block::

    git clone https://github.com/efabless/openlane2.git
    cd openlane2/nix/
    # TODO We want to track this directory only, how?
