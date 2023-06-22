# Parametric Functionality

One of the main benefits of open-source electronics is the ability to explore parametric designs closely interfaced with existing design flows in common languages, far more than writing old archaic tool-specific scripting languages.

## OpenLane Parametric Parameters

A list of interesting `OpenLane` design parameters over which you might want to parametrise your design could be:

| Parameter         | Description                                             |
|-------------------|---------------------------------------------------------|
| CLOCK_PERIOD      | in nanoseconds                                          |
| PL_TARGET_DENSITY |                                                         |
| SYNTH_PARAMETERS  | `yosys` synthesis parameters for parametric designs     |
| SYNTH_CAP_LOAD    | The capacitive load on the output ports in femtofarads. |
| SYNTH_MAX_FANOUT  |                                                         |
