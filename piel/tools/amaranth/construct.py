import amaranth as am
from typing import Literal

__all__ = ["construct_amaranth_module_from_truth_table"]


def construct_amaranth_module_from_truth_table(
    truth_table: dict,
    inputs: list[str],
    outputs: list[str],
    implementation_type: Literal[
        "combinatorial", "sequential", "memory"
    ] = "combinatorial",
):
    """
    This function implements a truth table as a module in amaranth,
    Note that in some form in amaranth each statement is a form of construction.

    The truth table is in the form of:

        detector_phase_truth_table = {
            "detector_in": ["00", "01", "10", "11"],
            "phase_map_out": ["00", "10", "11", "11"],
        }

    Args:
        truth_table (dict): The truth table in the form of a dictionary.
        inputs (list[str]): The inputs to the truth table.
        outputs (list[str]): The outputs to the truth table.
        implementation_type (Literal["combinatorial", "sequential", "memory"], optional): The type of implementation. Defaults to "combinatorial".

    Returns:
        Generated amaranth module.
    """

    class TruthTable(am.Elaboratable):
        def __init__(self, truth_table: dict, inputs: list, outputs: list):
            super(TruthTable, self).__init__()
            # Raise error if no truth table inputs are provided:
            if len(truth_table[inputs[0]]) == 0:
                raise ValueError("No truth table inputs provided." + str(inputs))

            # Initialise all the signals accordingly.
            for key, value in truth_table.items():
                # TODO Determine signal type or largest width from the values.
                max_length = max((len(s) for s in value), default=0)
                setattr(self, key, am.Signal(shape=max_length, name=key))

            self.inputs_names = inputs
            self.outputs_names = outputs

        def elaborate(self, platform):
            m = am.Module()
            # We need to iterate over the length of the truth table arrays for the input and output keys.
            # TODO implement multiinput.
            # TODO implement some verification that the arrays are of the same length.

            # Implements a particular output.
            output_signal_value_i = getattr(self, outputs[0]).eq

            with m.Switch(getattr(self, inputs[0])):
                for i in range(len(truth_table[self.inputs_names[0]])):
                    # We iterate over the truth table values
                    with m.Case(str(truth_table[self.inputs_names[0]][i])):
                        m.d.comb += output_signal_value_i(
                            int(truth_table[self.outputs_names[0]][i], 2)
                        )
                with m.Case():
                    m.d.comb += output_signal_value_i(0)

            return m

    return TruthTable(truth_table, inputs, outputs)
