import piel
import matplotlib.pyplot as plt


def test_create_axes_parameters_table_overlay():
    # Create a figure and axis
    fig, axs = plt.subplots(2, 1)  # Create a 2x1 grid of subplots

    # Plot some lines with different styles and colors
    x = [0, 1, 2, 3, 4]
    y1 = [0, 1, 4, 9, 16]
    y2 = [0, -1, -4, -9, -16]

    axs[0].plot(x, y1, color="blue", linestyle="-", label="Line 1")
    axs[1].plot(x, y2, color="red", linestyle="--", label="Line 2")

    # Parameters list corresponding to each line
    parameters_list = [{"Parameter": "Line 1"}, {"Parameter": "Line 2"}]

    # Call the function to create the table
    piel.visual.create_axes_parameters_table_overlay(
        fig,
        axs,
        parameters_list,
        font_family="Roboto",
        header_font_weight="bold",
        cell_font_size=10,
    )

    # Show the plot with the table
    plt.show()


# TODO fixme
# def test_create_axes_parameters_table_separate():
#     # Sample tables_list, where each element corresponds to a table
#     tables_list = [
#         [
#             ["A", "B", "C"],  # Headers for Table 1
#             [1, 4, 7],  # Row 1 for Table 1
#             [2, 5, 8],  # Row 2 for Table 1
#             [3, 6, 9],  # Row 3 for Table 1
#         ],
#         [
#             ["D", "E", "F"],  # Headers for Table 2
#             [10, 40, 70],  # Row 1 for Table 2
#             [20, 50, 80],  # Row 2 for Table 2
#             [30, 60, 90],  # Row 3 for Table 2
#         ],
#         [
#             ["G", "H", "I"],  # Headers for Table 3
#             [100, 400, 700],  # Row 1 for Table 3
#             [200, 500, 800],  # Row 2 for Table 3
#             [300, 600, 900],  # Row 3 for Table 3
#         ],
#     ]
#
#     # Create subplots
#     fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 10))
#
#     # Plot data in the subplots
#     axs[0].plot([1, 2, 3], [4, 5, 6], label="Plot 1")
#     axs[0].set_title("Plot 1")
#     axs[0].legend()
#
#     axs[1].plot([1, 2, 3], [7, 8, 9], label="Plot 2")
#     axs[1].set_title("Plot 2")
#     axs[1].legend()
#
#     axs[2].plot([1, 2, 3], [10, 11, 12], label="Plot 3")
#     axs[2].set_title("Plot 3")
#     axs[2].legend()
#
#     # Insert tables and adjust subplot positions
#     piel.visual.create_axes_parameters_tables_separate(
#         fig, axs, tables_list, table_height=0.15, spacing=0.005
#     )
#
#     # Show the final plot
#     plt.show()
