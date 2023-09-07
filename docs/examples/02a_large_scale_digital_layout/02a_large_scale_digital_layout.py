# # Digital Simulation & Layout Design Space Exploration

# ### Deployment and Timing of Large-Scale `openlane 2`
#
# One common and powerful aspect of this tool is the large-scale deployment of `openlane 2` designs so you can explore, say, the effect of different variations on your digital designs. You might want to multi-thread running your designs. Let's explore how to do this.
#
# We will use the built-in `multiprocessing` module in `python`:

import multiprocessing
import time

# We will go through the whole process of using `amaranth` for digital simulation and design later. For now, let's assume we have a XOR gate truth table we want to implement multiple times with different `id`. We will time both sequential and parallel implementations of this layout flow, and determine which is faster.

from piel.integration.amaranth_openlane import layout_openlane_from_truth_table

xor_truth_table = {
    "input": ["00", "01", "10", "11"],
    "output": ["0", "1", "1", "0"],
}
input_ports_list = ["input"]
output_ports_list = ["output"]


def sequential_implementations(amount_of_implementations: int):
    implementations = list()

    for i in range(amount_of_implementations):
        implementation_i = layout_openlane_from_truth_table(
            truth_table=xor_truth_table,
            inputs=input_ports_list,
            outputs=output_ports_list,
            parent_directory="sequential",
            target_directory_name="xor_" + str(i),
        )
        implementations.append(implementation_i)


def parallel_implementations(amount_of_implementations: int):
    processes = []

    for i in range(amount_of_implementations):
        # Create all processes
        process_i = multiprocessing.Process(
            target=layout_openlane_from_truth_table,
            kwargs={
                "truth_table": xor_truth_table,
                "inputs": input_ports_list,
                "outputs": output_ports_list,
                "parent_directory": "parallel",
                "target_directory_name": "xor_" + +str(i),
            },
        )
        processes.append(process_i)

    # This starts them in parallel
    for p in processes:
        p.join()

    for p in processes:
        p.start()


# Let's time this:

# +
start_parallel = time.time()
parallel_implementations(amount_of_implementations=4)
end_parallel = time.time()

print("Parallel")
print(end_parallel - start_parallel)

# +
start_sequential = time.time()
parallel_implementations(amount_of_implementations=4)
end_sequential = time.time()

print("Sequential")
print(end_sequential - start_sequential)
