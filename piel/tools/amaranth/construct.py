"""
This module provides a function to construct an Amaranth module from a truth table. It converts a truth table
into a digital logic module using the Amaranth framework.

The supported implementation measurement are:
- "combinatorial"
- "sequential"
- "memory"
"""

from piel.types.digital import TruthTable, LogicImplementationType


def construct_amaranth_module_from_truth_table(
    truth_table: TruthTable,
    logic_implementation_type: LogicImplementationType = "combinatorial",
):
    """
    Constructs an Amaranth module based on the provided truth table.
    # TODO implementation type

    Args:
        truth_table (TruthTable): The truth table to be implemented as a TruthTable object.
        logic_implementation_type (Literal["combinatorial", "sequential", "memory"], optional): The type of implementation.
            - "combinatorial": Implements the truth table as combinational logic.
            - "sequential": Implements the truth table as sequential logic.
            - "memory": Implements the truth table using memory elements.
            Defaults to "combinatorial".

    Returns:
        am.Module: An Amaranth module implementing the given truth table.

    Examples:
        >>> detector_phase_truth_table = {
        >>>     "detector_in": ["00", "01", "10", "11"],
        >>>     "phase_map_out": ["00", "10", "11", "11"],
        >>> }
        >>> my_truth_table = TruthTable(
        >>>     input_ports=["detector_in"],
        >>>     output_ports=["phase_map_out"],
        >>>     **detector_phase_truth_table
        >>> )
        >>> am_module = construct_amaranth_module_from_truth_table(my_truth_table)
    """
    import amaranth as am

    # Extract inputs and outputs from the truth table
    inputs = truth_table.input_ports
    outputs = truth_table.output_ports
    truth_table_dict = truth_table.implementation_dictionary

    if logic_implementation_type == "combinatorial":

        class TruthTableModule(am.Elaboratable):
            """
            A class representing an Amaranth module generated from a truth table.

            Attributes:
                input_signal (am.Signal): The signal corresponding to the input port of the truth table.
                output_signals (dict): A dictionary mapping output port names to their corresponding signals.
                inputs_names (list): A list of input port names.
                outputs_names (list): A list of output port names.
                truth_table (dict): The truth table dictionary with inputs and outputs.
            """

            def __init__(self, truth_table_dict: dict, inputs: list, outputs: list):
                """
                Initializes the TruthTableModule with the given truth table, inputs, and outputs.

                Args:
                    truth_table_dict (dict): The dictionary representing the truth table.
                    inputs (list): A list of input port names.
                    outputs (list): A list of output port names.
                """
                super(TruthTableModule, self).__init__()

                # Ensure that the truth table has entries
                if len(truth_table_dict[inputs[0]]) == 0:
                    raise ValueError("No truth table inputs provided: " + str(inputs))

                # Initialize signals for input and output ports and assign them as attributes
                self.input_signal = am.Signal(
                    len(truth_table_dict[inputs[0]][0]), name=inputs[0]
                )
                self.output_signals = {
                    output: am.Signal(len(truth_table_dict[output][0]), name=output)
                    for output in outputs
                }

                # Assign input and output signals as class attributes for external access
                setattr(self, inputs[0], self.input_signal)
                for output in outputs:
                    setattr(self, output, self.output_signals[output])

                self.inputs_names = inputs
                self.outputs_names = outputs
                self.truth_table = truth_table_dict

            def elaborate(self, platform):
                """
                Elaborates the Amaranth module to implement the logic based on the truth table.

                Args:
                    platform: The platform on which the module is to be implemented. TODO

                Returns:
                    am.Module: The elaborated Amaranth module.
                """
                m = am.Module()

                # Assume the truth table entries are consistent and iterate over them
                with m.Switch(self.input_signal):
                    for i in range(len(self.truth_table[self.inputs_names[0]])):
                        input_case = str(self.truth_table[self.inputs_names[0]][i])
                        with m.Case(input_case):
                            # Assign values to each output signal for the current case
                            for output in self.outputs_names:
                                output_signal_value = self.output_signals[output].eq
                                m.d.comb += output_signal_value(
                                    int(self.truth_table[output][i], 2)
                                )

                    # Default case: set all outputs to 0
                    with m.Case():
                        for output in self.outputs_names:
                            m.d.comb += self.output_signals[output].eq(0)

                return m

    elif logic_implementation_type == "sequential":

        class TruthTableModule(am.Elaboratable):
            def __init__(self, truth_table_dict: dict, inputs: list, outputs: list):
                super(TruthTableModule, self).__init__()

                if len(truth_table_dict[inputs[0]]) == 0:
                    raise ValueError("No truth table inputs provided: " + str(inputs))

                self.input_signal = am.Signal(
                    len(truth_table_dict[inputs[0]][0]), name=inputs[0]
                )
                self.output_signals = {
                    output: am.Signal(len(truth_table_dict[output][0]), name=output)
                    for output in outputs
                }

                setattr(self, inputs[0], self.input_signal)
                for output in outputs:
                    setattr(self, output, self.output_signals[output])

                self.inputs_names = inputs
                self.outputs_names = outputs
                self.truth_table = truth_table_dict

                # State register for sequential implementation
                self.state = am.Signal(
                    len(truth_table_dict[inputs[0]][0]), name="state"
                )

            def elaborate(self, platform):
                m = am.Module()

                # Register to hold the state
                next_state = am.Signal.like(self.state)

                m.d.sync += self.state.eq(next_state)

                # State transition logic based on input signal
                with m.Switch(self.input_signal):
                    for i in range(len(self.truth_table[self.inputs_names[0]])):
                        input_case = int(self.truth_table[self.inputs_names[0]][i], 2)
                        with m.Case(input_case):
                            m.d.sync += next_state.eq(input_case)
                            for output in self.outputs_names:
                                output_value = int(self.truth_table[output][i], 2)
                                m.d.sync += self.output_signals[output].eq(output_value)

                    # Default case: retain current state
                    with m.Case():
                        m.d.sync += next_state.eq(self.state)

                return m

    return TruthTableModule(truth_table_dict, inputs, outputs)
