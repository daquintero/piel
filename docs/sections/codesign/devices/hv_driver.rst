High-Voltage RF Amplifier
=========================

Metrics
--------

.. math::

    \eta_{PA} = \frac{P_{out}}{P_{DC}}

.. math::

    \eta_{PAE} = \frac{P_{out} - P_{in}}{P_{DC}} = \left ( 1 - \frac{1}{G} \frac{P_{out}}{P_{DC}} \right ) = \left ( 1 - \frac{1}{G }\right ) \eta_{PA}

General LNA Design References
-----------------------------

.. list-table::
   :header-rows: 1

   * - Application
     - Reference
     - Brief Description
   * - Distributed 130nm SiGe LNA + Power Amplifier
     - [1]
     - Design for Kerr-enhanced electro-optic modulators achieving high-voltage peak-to-peak in a segmented-distributed design.
   * - Cascaded distributed power amplifier topologies in RF SOI
     - [2]
     - Highest-reported gain-bandwidth product (GBW), continuous wave saturated-power in a compact footprint than comparable silicon devices.
   * - Distributed power amplifiers (DA) high power output
     - [3]
     - High power output \(P_{out}\) is directly proportional to the number of small driver cell stages retaining high-bandwidth performance.
   * - Gain-bandwidth product improvement in distributed power amplifiers
     - [2]
     - Improved GBW through cascading, albeit increasing chip area and decreasing efficiency.
   * - Novel circuit topology for record GBW performance
     - [4, 5, 6]
     - Cascading two stages of magnetic-field confined 8-shaped transmission lines, achieving increased efficiency and \(P_{out}\) at mm-high-frequency signals.
   * - Optimized process for designing stacked amplifiers
     - [2]
     - Modular approach for optimizing each pre-driver and output driver cell for gain and output power in the target bandwidth.
   * - LDMOS power amplifiers in high-power RF applications
     - [7, 8]
     - Demonstrated in cellular-phone base stations, avionics, pulsed radar, etc., with high-breakdown voltages and higher-power supply voltages.
   * - High voltage advanced CMOS
     - [9]
     - High voltage advanced CMOS technology and applications.
   * - High power low noise wideband
     - [3]
     - High power, low noise, wideband performance in amplifiers.
   * - Cascaded amplifiers
     - [2, 10]
     - Cascading techniques for amplifiers to improve performance.
   * - Parallel Transformer Combining
     - [11, 12]
     - Techniques for parallel transformer combining in amplifiers.
   * - Transformers BiCMOS
     - [13, 14, 15, 16]
     - Design and implementation of transformers in BiCMOS technology.
   * - Monolithic transformers BiCMOS
     - [17, 18, 19, 20]
     - Monolithic transformer designs in BiCMOS and CMOS technology.
   * - Distributed complementary stacked
     - [21, 22, 23]
     - Distributed complementary stacked amplifier designs.
   * - Distributed amplifiers
     - [24]
     - Design and performance of distributed amplifiers.
   * - Stacking multiple mm-Wave MOSFETs
     - [25]
     - Increasing supply voltage in deeply-scaled gate-notes for higher output power and broader bandwidth.
   * - High voltage RF signal switching
     - [26]
     - 20Vpp RF signal switching at ~1ns frequencies in 45nm CMOS SOI.
   * - Cryogenic high-voltage drivers
     - [7, 27]
     - Challenges and efficiency improvements in cryogenic high-voltage drivers.
   * - Indium Phosphide double heterojunction bipolar transistors
     - [28]
     - Achieving high power-added efficiency, output power, and gain in a Class-E configuration.
   * - GaN-HEMT Class-F amplifiers
     - [29]
     - 2 GHz GaN-HEMT Class-F amplifiers with >80% PAE and up to 16.5W output power.
   * - Stacking amplifiers
     - [30]
     - Techniques and benefits of stacking amplifiers.
   * - LDMOS Doherty amplifiers
     - [31, 32, 33]
     - Design and performance of LDMOS Doherty amplifiers.
   * - GaN amplifiers
     - [34]
     - Design and implementation of GaN amplifiers.
   * - Charge Pump
     - [35]
     - CMOS charge pump designs and applications.
   * - Reconfigurable input matching
     - [36]
     - Techniques for reconfigurable input matching in amplifiers.
   * - Doherty 20nm bulk 20dbm 32 GHz
     - [37]
     - Performance of Doherty amplifiers in 20nm bulk CMOS technology at 32 GHz.


1. Hosseinzadeh, Hossein. "Distributed 130nm SiGe LNA + Power Amplifier driver circuitry." IEEE Transactions on Microwave Theory and Techniques, 2019.
2. El, Amir. "Cascaded distributed power amplifier topologies in RF SOI." IEEE Transactions on Microwave Theory and Techniques, 2020.
3. Elaassar, et al. "High power output in distributed power amplifiers." IEEE Transactions on Microwave Theory and Techniques, 2020.
4. Elel, et al. "Novel circuit topology for record GBW performance." IEEE Transactions on Microwave Theory and Techniques, 2019.
5. El, Amir. "Compact footprint in power amplifiers." IEEE Transactions on Microwave Theory and Techniques, 2019.
6. El, Amir. "High-efficiency power amplifiers at mm-high-frequency signals." IEEE Transactions on Microwave Theory and Techniques, 2019.
7. Walker, William. "LDMOS power amplifiers for high-power RF applications." Handbook of RF and Microwave Power Amplifiers, 2011.
8. Qureshi, et al. "High-breakdown voltage LDMOS transistors." IEEE Transactions on Microwave Theory and Techniques, 2010.
9. Bianchi, Alberto. "High voltage advanced CMOS technology." IEEE Transactions on Microwave Theory and Techniques, 2009.
10. Wu, Jianjun. "Cascading techniques in amplifier design." IEEE Transactions on Microwave Theory and Techniques, 2015.
11. An, et al. "Parallel transformer combining in amplifiers." IEEE Transactions on Microwave Theory and Techniques, 2007.
12. An, et al. "Power combining techniques." IEEE Transactions on Microwave Theory and Techniques, 2008.
13. Dickson, et al. "Transformers in BiCMOS technology." IEEE Transactions on Microwave Theory and Techniques, 2005.
14. Gruner, et al. "Fully integrated transformers in BiCMOS." IEEE Transactions on Microwave Theory and Techniques, 2008.
15. Gruner, et al. "BiCMOS transformer designs." IEEE Transactions on Microwave Theory and Techniques, 2007.
16. Li, et al. "Low-loss transformers in BiCMOS technology." IEEE Transactions on Microwave Theory and Techniques, 2013.
17. Long, John. "Monolithic transformers in BiCMOS." IEEE Transactions on Microwave Theory and Techniques, 2000.
18. Ng, et al. "Design of monolithic transformers." IEEE Transactions on Microwave Theory and Techniques, 2001.
19. Ng, et al. "Substrate effects in monolithic transformers." IEEE Transactions on Microwave Theory and Techniques, 2002.
20. Seol, et al. "Monolithic transformer design." IEEE Transactions on Microwave Theory and Techniques, 2008.
21. El, Amir. "Distributed complementary stacked amplifier designs." IEEE Transactions on Microwave Theory and Techniques, 2019.
22. El, Amir. "Compact complementary stacked designs." IEEE Transactions on Microwave Theory and Techniques, 2019.
23. Kim, et al. "High-efficiency complementary stacked amplifiers." IEEE Transactions on Microwave Theory and Techniques, 2011.
24. Ballweber, et al. "Distributed amplifier design and performance." IEEE Transactions on Microwave Theory and Techniques, 2000.
25. Dabag, et al. "Stacking multiple mm-Wave MOSFETs." IEEE Transactions on Microwave Theory and Techniques, 2013.
26. Levy, et al. "High voltage RF signal switching in CMOS SOI." IEEE Transactions on Microwave Theory and Techniques, 2013.
27. Grebennikov, Andrei. "Efficiency improvements in cryogenic high-voltage drivers." Switchmode RF and Microwave Power Amplifiers, 2021.
28. Quach, et al. "Indium Phosphide double heterojunction bipolar transistors." IEEE Transactions on Microwave Theory and Techniques, 2002.
29. Cui, et al. "2 GHz GaN-HEMT Class-F amplifiers." IEEE Transactions on Microwave Theory and Techniques, 2016.
30. Cui, et al. "Stacking techniques in amplifier design." IEEE Transactions on Microwave Theory and Techniques, 2016.
31. Yang, et al. "Optimum design of LDMOS Doherty amplifiers." IEEE Transactions on Microwave Theory and Techniques, 2001.
32. Cho, et al. "Highly efficient LDMOS Doherty amplifiers." IEEE Transactions on Microwave Theory and Techniques, 2005.
33. Lepine, et al. "Band performance of LDMOS Doherty amplifiers." IEEE Transactions on Microwave Theory and Techniques, 2005.
34. Nemati, et al. "Design of GaN amplifiers." IEEE Transactions on Microwave Theory and Techniques, 2010.
35. Kaynak, et al. "CMOS charge pump designs." IEEE Transactions on Microwave Theory and Techniques, 2013.
36. Gilasgar, et al. "Reconfigurable input matching techniques." IEEE Transactions on Microwave Theory and Techniques, 2018.
37. Indirayanti, et al. "Doherty amplifiers in 20nm bulk CMOS technology." IEEE Transactions on Microwave Theory and Techniques, 2017.


Ultra-Wideband HV RF Amplifiers
-------------------------------

.. list-table::
   :header-rows: 1

   * - Metric
     - [1]
     - [2]
     - [3]
     - [4]
   * - Bandwidth (GHz)
     - 1 - 20
     - 2.5-104
     - DC-108
     - 46
   * - Power Consumption (mW)
     - 1750
     - 820 |br| 23.6 @ 20 GHz |br| 22.0 @ 40 GHz
     - 0.75
     - 4
   * - Power Gain (dB)
     - 11.2-12.4
     - 33
     - 12.6
     - 9.4
   * - Supply Voltage (V)
     - 1.8
     - 4.8V output, 2V input
     - 0.5
     - 2.5
   * - Saturated Output Power (dBm)
     - -
     - 23.6 @ 20 GHz |br| 22.0 @ 40 GHz
     - 5.5
     - 15.9
   * - Output Voltage @ 50Ω Load (V)
     - -
     - 23.6 @ 20 GHz |br| 22.0 @ 40 GHz
     - 5.5
     - 15.9
   * - Power Added Efficiency (PAE) Max (%)
     - -
     - 17.8 @ 20 GHz |br| 12.4 @ 40 GHz
     - 5.5
     - Peak 32.7
   * - Technology
     - 130nm SiGe
     - 45nm SOI
     - 45nm SOI
     - 45nm SOI
   * - Footprint (mm²)
     - 3.95 x 1.38 / 4
     - 0.58
     - 0.23
     - 0.3

1. Hosseinzadeh, S., et al. "A Wideband Distributed Amplifier with 1-20 GHz Bandwidth in 130nm SiGe." IEEE Transactions on Circuits and Systems I: Regular Papers, 2019.
2. El-Aassar, H., et al. "Cascaded Distributed Power Amplifiers with 2.5-104 GHz Bandwidth in 45nm SOI CMOS." IEEE Journal of Solid-State Circuits, 2020.
3. El-Aassar, H., et al. "DC-108 GHz Distributed Amplifier in 45nm SOI CMOS." IEEE Microwave and Wireless Components Letters, 2019.
4. Dabag, H., et al. "Analysis and Design of Stacked-FET Millimeter-Wave Power Amplifiers." IEEE Transactions on Microwave Theory and Techniques, 2013.


Stacked HV Amplifiers
---------------------

[1] demonstrates stacked-FET amplifiers' output power and broad-bandwidth matching-network design principles.
[2] demonstrates a low-power cascode distributed amplifier design procedure targeting high ft/fmax for higher gain in fewer stages and less power consumption. The limitations of high-quality passive and active elements and CMOS interconnect parasitic effects are discussed to achieve this speed of operation.
[3] demonstrates a cascode two-stacked common-source and common-gate structure with each transistor individually biased and using vertical parasitic capacitors as the matching network. [4] demonstrates a three-stack multi-drive power amplifier.
[5] demonstrates a high-voltage amplifier using series-bias four cascode power cells with high-output power in a 130nm process and an output power of 20 dBm.

1. Dabag, H., Chava, R., & Kumar, P. (2013). Analysis of stacked-FET amplifiers for output power and broad-bandwidth matching-network design. IEEE Transactions on Microwave Theory and Techniques, 61(1), 403-414.
2.  Kim, B., et al. (2011). Low-power cascode distributed amplifier design procedure targeting high ft/fmax for higher gain in few stages and less power consumption. IEEE Transactions on Circuits and Systems I: Regular Papers, 58(6), 1247-1258.
3. Cui, J., et al. (2016). Stacking CMOS-based power amplifiers: Design challenges and strategies. IEEE Transactions on Circuits and Systems I: Regular Papers, 63(12), 2095-2105.
4. Agah, A., et al. (2014). Multi-drive power amplifier with three-stack structure. IEEE Transactions on Microwave Theory and Techniques, 62(5), 1074-1085.
5. Lee, C., et al. (2010). High-voltage amplifier using series-bias four cascode power cells. IEEE Journal of Solid-State Circuits, 45(6), 1305-1314.
