It might be necessary to do propagation delay measurements through an electronic component.
Let's consider a few experimental strategies to implement this depending on the equipment available.

.. list-table::
   :header-rows: 1

   * - Tool Type
     - Version
     - Functionality
   * - Time-Tagger
     - `Swabian Time Tagger Ultra <https://www.swabianinstruments.com/static/downloads/TimeTaggerSeries.pdf>`_  & `Picoharp 300 <https://www.picoquant.com/products/category/tcspc-and-time-tagging-modules/picoharp-300-stand-alone-tcspc-module-with-usb-interface-succ>`_
     -
   * - RF Oscilloscope
     - `Tektronix <https://download.tek.com/manual/DPO70000SX-Series-Real-Time-Oscilloscopes-User-Manual-EN-071335707.pdf>`_
     -
   * - RF Signal Generator
     - `Tektronix AWG70001A <https://download.tek.com/manual/AWG70000A-Installation-Safety-Instructions-071311004_New.pdf?_gl=1*1bqtdvz*_gcl_au*NTkzODIyNjEyLjE3MjAwODc3MTg.*_ga*MTE3MzUwMjgzOC4xNzIwMDg3NzE4*_ga_1HMYS1JH9M*MTcyMDA4NzcxNy4xLjAuMTcyMDA4NzcxNy42MC4wLjA.>`_
     -

.. list-table::
   :header-rows: 1

   * - Method
     - Evaluation
     - Requirements
     - References
   * - Time-Domain Reflectometry
     - Technically more-accurate, can require more complex equipment.
     - Oscilloscope with TDR functionality, RF Signal Generator
     - `Propagation Delay Measurements Using TDR (Time-Domain Reflectometry) <https://www.analog.com/en/resources/technical-articles/propagation-delay-measurements-using-tdr-timedomain-reflectometry.html>`_
   * - Time-Domain Reflectometry
     - Technically more-accurate
     - Oscilloscope with TDR functionality, RF Signal Generator
     - `Propagation Delay Measurements Using TDR (Time-Domain Reflectometry) <https://www.analog.com/en/resources/technical-articles/propagation-delay-measurements-using-tdr-timedomain-reflectometry.html>`_


Let's consider some important general concepts to understand of such systems:

* The first thing to understand when setting up a timing experiment is to make sure that the bandwidth of the system can catch the signal resolution of the pulse we want to image.
* Are we putting in each device the corresponding electrical signal compatible with the corresponding hardware rating? Is there a risk of breaking it?
* We also want to consider what are the frequencies that we are interested in a given pulse, and how we're able to discretize them from a given measurement. Is our network impedance compatible between devices?
* We want to understand how we've removed experimental artifacts from the system.

Using a Time-Tagger for Propagation Delay Measurements
-------------------------------------------------------

Using the devices as detailed above, we can understand the electrical characteristics of the `Swabian Time Taggers <https://www.swabianinstruments.com/static/documentation/TimeTagger/sections/hardware.html#electrical-characteristics>`_.



