FSIC 2024 Talk Outline
-----------------------

This page involves the outline and resources related to of the presentation: *Integrating Mixed-Signal Microelectronics and Photonics: A Co-Design Approach with piel* for the `2024 Free Silicon Conference (FSiC) <https://wiki.f-si.org/index.php/FSiC2024>`_.


Outline
^^^^^^^^

- Motivation
    - Most people will have an electronics background.
    - Why co-design?
        - Basic applications of photonics and electronics TODO add ref to codesign flows
            - Opto-electronic control
            - Electro-optic control
        - Modelling concurrent photonic-electronic systems
            - Diagram picture here.
            - This is what we want to model.
- Introduction to ``piel``
    - Python library, container of a collection of examples of co-design flows, and useful functions that streamline interaction between toolsets specifically for co-design, proper dependency management for reproducibility*
    - Uses existing functionality from toolsets, aims to proper dependency management functions and interactions so you don't get horrible conflicts.
    - Why?
        - Applications of electronic-photonic systems. Massive pain of proprietary toolsets, not easy to interact with.
        - Power of open source PDKs in this context.
        - Was working on quantum photonic systems and it was a massive pain to simulate how the system behaved.
    - TODO possibly talk about design flows if need be
- Let's go through an example.
    - Finish 07_full_flow_electronic_photonic_demo.
    - Say, we have an optical function we want to demonstrate or implement, extract the logic for that.
    - Go through the flow of implementing that logic via amaranth, syntheziging that logic via openlane, extracting performance parameters accordingly, modelling the analog amplifier accordingly and implementing it via gdsfactory with sky130
    - Model the component with analog, digital circuits simulators accordingly. In the future, add the mixed-signal with full-concurrency for photonic time-domain signals.
    - This is a good example.
    - We can explore say, a cocotb-signal being used to model the transmission of a photonic network.
