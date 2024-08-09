Experiment Metadata Management
------------------------------

When performing experimental testing and acquiring data, one major issue is both knowing what configuration to apply if you have multiple tests and then relating this to a given plotting configuration. It would be nice if we could reuse all the existing functionality from the simulation data to analyse the experimental data. As such, it is necessary to create a common directory and device test structure that can be reused.

``piel`` develops an approach to perform this set of relationships and configuration of testing whilst sharing the same data structures as the simulation configurations.

The goal of the defined classes is to provide a serializable interface to describe an experimental setup. Based on this, translation methods between experimental and simulation designs can be implemented accordingly.

Ultimately, the way this works is that we have a data directory in this structure:

.. raw::

    measurements/ # Contains all the measurements, for all devices, interconnection and operating conditions
        1/ # A specific setup configuration identified via an id
            instance.json # metadata for this setup
        experiment.json # mapping between ids and all the information for the experimental setup

This means that a unique `1/` directory is created for each measurement set that contains a custom configuration, specific device test, corresponding bias configuration, and device setup. This can be mapped, and saved, through a `experiment_id` which can be mapped within `experiment.json`.

It could be argued that if there are more than one files per operating setup, then it requires a directory in which this is saved.

Note that the information within an operating setup as defined in the `experiment.json` can involve:

.. raw::

    experiment_type # frequency, time, dc
    measurement_information
        configuration_information
        ...
    device_under_test_information
        environment_information
            temperature
    wiring_information
    signal_setting_information
        dc_bias

This all needs to be serialised in a flat format where this can be edited and accessed through a csv file for example.
It can be defined that we have components with different environments, and interconnect with different sets of
information that we then want to compose together. The goal is really to serialise all this relevant information.

Hence, we want to compose this information in an easy format that we can then compose ultimately.
