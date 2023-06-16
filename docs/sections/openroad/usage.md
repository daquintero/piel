# Usage

It is very simple. You need to structure your logic design files in a directory with this structure:

Pre-generation file required configuration
```
    design_folder_name
        io/
            pin_order.cfg # Required: OpenRoad
        model/
            design_model.py # Optional: cocotb
        sdc/
            design.sdc # Required: OpenRoad
        src/
            source_files.v # Required by all
        tb/
            test_design.py # Required cocotb

```

