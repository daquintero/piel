# # Digital Simulation & Layout Design Space Exploration

# ### Deployment and Timing of Large-Scale `openlane 2`
#
# One common and powerful aspect of this tool is the large-scale deployment of `openlane 2` designs so you can explore, say, the effect of different variations on your digital designs. You might want to multi-thread running your designs. Let's explore how to do this.
#
# We will use the built-in `multiprocessing` module in `python`:

import multiprocessing
import time

# We will go through the whole process of using `amaranth` for digital simulation and design later. For now, let's assume we have a random truth table we want to implement multiple times with different `id`. We will time both sequential and parallel implementations of this layout flow, and determine which is faster.

from piel.integration.amaranth_openlane import layout_openlane_from_truth_table

truth_table = {
    "input": [
        "0000",
        "0001",
        "0010",
        "0011",
        "0100",
        "0101",
        "0110",
        "0111",
        "1000",
        "1001",
        "1010",
        "1011",
        "1100",
        "1101",
        "1110",
        "1111",
    ],
    "output": [
        "0101",
        "1100",
        "0101",
        "0110",
        "0010",
        "1101",
        "0110",
        "0011",
        "1001",
        "1110",
        "0100",
        "1000",
        "0001",
        "1011",
        "1111",
        "1010",
    ],
}
input_ports_list = ["input"]
output_ports_list = ["output"]


def sequential_implementations(amount_of_implementations: int):
    implementations = list()

    for i in range(amount_of_implementations):
        implementation_i = layout_openlane_from_truth_table(
            truth_table=truth_table,
            inputs=input_ports_list,
            outputs=output_ports_list,
            parent_directory="sequential",
            target_directory_name="sequential_" + str(i),
        )
        implementations.append(implementation_i)


# +
def parallel_implementations(amount_of_implementations: int):
    processes = []

    for i in range(amount_of_implementations):
        # Create all processes
        process_i = multiprocessing.Process(
            target=layout_openlane_from_truth_table,
            kwargs={
                "truth_table": truth_table,
                "inputs": input_ports_list,
                "outputs": output_ports_list,
                "parent_directory": "parallel",
                "target_directory_name": "parallel_" + str(i),
            },
        )
        processes.append(process_i)

    for p in processes:
        p.start()

    # This starts them in parallel
    for p in processes:
        p.join()


# -

# Let's time this:

# +
start_parallel = time.time()
parallel_implementations(amount_of_implementations=4)
end_parallel = time.time()

print("Parallel")
print(end_parallel - start_parallel)

# +
start_sequential = time.time()
sequential_implementations(amount_of_implementations=4)
end_sequential = time.time()

print("Sequential")
print(end_sequential - start_sequential)
# -
