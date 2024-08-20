import piel
import matplotlib.pyplot as plt


def test_create_axes_parameters_table():
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
    piel.visual.create_axes_parameters_table(
        fig,
        axs,
        parameters_list,
        font_family="Roboto",
        cell_font_size=10,
        header_font_weight="bold",
    )

    # Show the plot with the table
    plt.show()
