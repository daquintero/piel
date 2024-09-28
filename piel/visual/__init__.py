from .plot.core import save
from .plot.basic import plot_simple, plot_simple_multi_row, plot_simple_multi_row_list
from .plot.position import (
    create_axes_per_figure,
    create_plot_containers,
    list_to_overlayed_plots,
    list_to_separate_plots,
)
from .plot.table import (
    create_axes_parameters_table_overlay,
    create_axes_parameters_tables_separate,
)
from .plot import signals, metrics
from .data_conversion import append_row_to_dict, points_to_lines_fixed_transient
from .style import (
    activate_piel_styles,
    primary_cycler,
    secondary_cycler,
    primary_color_palette,
    secondary_color_palette,
)
from .signals import *
from .types import AxesPlottingTypes, ExtensiblePlotsDirectionPerElement
from .json_to_markdown import dictionary_to_markdown_str
from .experimental import *

from . import table

activate_piel_styles()
