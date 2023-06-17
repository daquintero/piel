# Project Structure Usage

It is very simple and follows convention. You need to structure your logic design files in a directory with this structure:

Pre-generation file required configuration
```
design_folder_name
    config.json # Optional, Required only for OpenLane V1
    io/
        pin_order.cfg # Required: OpenLane
    model/
        design_model.py # Optional: cocotb
    sdc/
        design.sdc # Required: OpenLane
    src/
        source_files.v # Required by all
    tb/
        test_design.py # Required cocotb
```

If you run the full flow, the design folder will look like:

```
design_folder_name
    config.json # Optional, Required only for OpenLane V1
    io/
        pin_order.cfg # Required: OpenLane
    model/
        design_model.py # Optional: cocotb
    runs/
        openlane_run_folder # Required OpenLane
    sdc/
        design.sdc # Required: OpenLane
    src/
        source_files.v # Required by all
    tb/
        test_design.py # Required cocotb
```
