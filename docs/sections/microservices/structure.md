### Package Structure

A general overview of the piel python package structure:

```raw
piel/
    cli/ # command-line scripts
    experimental/ # selected set of functions useful when interacting with relevant equipment
    flows/ # High-level user-specific functionality to automate common design tasks
    integration/ # Deterministic explicit functionality between toolsets
    materials/ # Self-contained or referenced material models
    models/ # Specific instances, or target application usages, operate on other directories
    tools/ # Relevant extended functionality onto supported tools
    visual/ # Utilities to plot and visualise relationships
```
