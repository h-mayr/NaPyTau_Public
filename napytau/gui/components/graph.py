from tkinter import Canvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Any, TYPE_CHECKING

from matplotlib.axes import Axes
import customtkinter
import numpy as np
import scipy

from napytau.gui.components.toolbar import Toolbar
from napytau.gui.model.color import Color
from napytau.gui.model.marker_factory import generate_marker
from napytau.gui.model.marker_factory import generate_error_marker_path

from napytau.import_export.model.datapoint_collection import DatapointCollection


if TYPE_CHECKING:
    from napytau.gui.app import App  # Import only for the type checking.


class Graph:
    def __init__(self, parent: "App") -> None:
        self.parent = parent
        self._knot_mode: bool = False
        self._knot_distances: list[float] = []
        self.graph_frame = self.plot(customtkinter.get_appearance_mode())
        self.graph_frame.grid(
            row=1, column=0, rowspan=2, padx=(10, 10), pady=(10, 0), sticky="nsew"
        )
        self.graph_frame.grid_propagate(False)

    def plot(self, appearance: str) -> Canvas:
        # the figure that will contain the plot
        fig = Figure(
            figsize=(3, 4), dpi=100, facecolor=Color.WHITE, edgecolor=Color.BLACK
        )

        # adding two stacked subplots sharing the x-axis
        axes_1 = fig.add_subplot(211)
        axes_2 = fig.add_subplot(212, sharex=axes_1)
        fig.subplots_adjust(left=0.1, bottom=0.07, right=0.9, top=0.95, hspace=0.08)

        # set colors according to appearance mode
        self.set_colors(appearance)

        # apply colors onto figure and both axes
        self.apply_coloring(fig, axes_1)
        self.apply_coloring(fig, axes_2)

        # add grid style to both axes
        for ax in (axes_1, axes_2):
            ax.grid(
                True,
                which="both",
                color=self.secondary_color,
                linestyle="--",
                linewidth=0.3,
            )
            ax.set_xscale("log")

        # hide x-tick labels on top axes (shared; bottom carries them)
        axes_1.tick_params(labelbottom=False)

        # draw shifted markers + fitting curve on top axes
        self.plot_shifted_markers(self.parent.datapoints_for_fitting, axes_1)
        # draw unshifted markers + derivative curve on bottom axes
        self.plot_unshifted_markers(self.parent.datapoints_for_fitting, axes_2)

        if len(self.parent.datapoints_for_fitting.get_active_datapoints()) > 0:
            self.plot_fitting_curve(self.parent.datapoints_for_fitting, axes_1)
            self.plot_derivative_curve(self.parent.datapoints_for_fitting, axes_2)

        # draw knot markers on both axes
        self._plot_knot_lines(axes_1)
        self._plot_knot_lines(axes_2)

        # creating the Tkinter canvas containing the Matplotlib figure
        self.canvas = FigureCanvasTkAgg(fig, master=self.parent)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.draw()

        return self.canvas.get_tk_widget()

    def update_plot(self) -> None:
        """
        Is called whenever the graph needs to be re-rendered.
        """
        self.graph_frame = self.plot(customtkinter.get_appearance_mode())
        self.graph_frame.grid(
            row=1, column=0, rowspan=2, padx=(10, 10), pady=(10, 0), sticky="nsew"
        )
        self.graph_frame.grid_propagate(False)
        self.parent.toolbar = Toolbar(self.parent, self.canvas)

    def toggle_knot_mode(self) -> None:
        """Toggle interactive knot placement mode on or off."""
        self._knot_mode = not self._knot_mode

    def clear_knots(self) -> None:
        """Remove all user-placed knots and clear dataset sampling_points."""
        self._knot_distances = []
        self.parent.dataset[0].set_sampling_points([])

    def _plot_knot_lines(self, axes: Axes) -> None:
        """Draw a vertical dashed orange line for each knot distance."""
        for d in self._knot_distances:
            axes.axvline(
                x=d,
                color="#ff8c00",
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
            )

    def _on_click(self, event: Any) -> None:
        """
        Handle a matplotlib click event.

        In knot mode, left-click adds a knot at the clicked distance;
        clicking within 10 % of an existing knot removes it instead.
        Right-click removes the nearest knot.
        """
        if not self._knot_mode or event.inaxes is None:
            return

        x = event.xdata
        if x is None or x <= 0:
            return

        # need at least some data loaded
        if len(self.parent.datapoints_for_fitting.get_active_datapoints()) == 0:
            return

        # Check if click is within 10 % of an existing knot → remove it
        for existing in list(self._knot_distances):
            if abs(x - existing) / existing < 0.10:
                self._knot_distances.remove(existing)
                self._update_sampling_points()
                self.parent.after(0, self._redraw_and_recalculate)
                return

        # Otherwise add a new knot at the click position (keep list sorted)
        if event.button == 1:  # left click only
            self._knot_distances.append(x)
            self._knot_distances.sort()
            self._update_sampling_points()
            self.parent.after(0, self._redraw_and_recalculate)

    def _update_sampling_points(self) -> None:
        """
        Convert knot distances to time units and store them in the dataset.
        time = distance / (velocity * c)
        """
        dataset = self.parent.dataset[0]
        velocity = dataset.get_relative_velocity().value.get_velocity()
        c = scipy.constants.speed_of_light
        if velocity == 0:
            return
        sampling_points = [d / (velocity * c) for d in self._knot_distances]
        dataset.set_sampling_points(sampling_points)

    def _redraw_and_recalculate(self) -> None:
        """Redraw the graph and trigger a τ recalculation."""
        self.update_plot()
        self.parent.control_panel.recalculate()

    def set_colors(self, appearance: str) -> None:
        if appearance == "Light":
            self.main_color = Color.WHITE
            self.secondary_color = Color.BLACK
            self.main_marker_color = Color.DARK_GREEN
            self.secondary_marker_color = Color.DARK_RED

        else:
            self.main_color = Color.DARK_GRAY
            self.secondary_color = Color.WHITE
            self.main_marker_color = Color.LIGHT_GREEN
            self.secondary_marker_color = Color.Light_RED

    def apply_coloring(self, figure: Figure, axes: Axes) -> None:
        """
        setting color in dependence of appearance mode
        :param figure: the figure to be recolored
        :param axes: the axes to be recolored
        :return: nothing
        """

        figure.patch.set_facecolor(self.main_color)

        # set color of background
        axes.set_facecolor(self.main_color)

        # set color of ticks
        axes.tick_params(axis="x", colors=self.secondary_color)
        axes.tick_params(axis="y", colors=self.secondary_color)

    def plot_shifted_markers(
        self, datapoints: DatapointCollection, axes: Axes
    ) -> None:
        """Plot shifted-intensity markers (green) on the given axes."""
        checked_datapoints: DatapointCollection = datapoints.get_active_datapoints()

        for index, datapoint in enumerate(checked_datapoints):
            marker = generate_marker(
                generate_error_marker_path(datapoint.get_intensity()[0].error)
            )
            size = datapoint.get_intensity()[0].error
            axes.plot(
                datapoint.get_distance().value,
                datapoint.get_intensity()[0].value,
                marker=marker,
                linestyle="None",
                markersize=size,
                label=f"Point {index + 1}",
                color=self.main_marker_color,
            )

    def plot_unshifted_markers(
        self, datapoints: DatapointCollection, axes: Axes
    ) -> None:
        """Plot unshifted-intensity markers (red) on the given axes."""
        checked_datapoints: DatapointCollection = datapoints.get_active_datapoints()

        for index, datapoint in enumerate(checked_datapoints):
            marker = generate_marker(
                generate_error_marker_path(datapoint.get_intensity()[1].error)
            )
            size = datapoint.get_intensity()[1].error
            axes.plot(
                datapoint.get_distance().value,
                datapoint.get_intensity()[1].value,
                marker=marker,
                linestyle="None",
                markersize=size,
                label=f"Point {index + 1}",
                color=self.secondary_marker_color,
            )

    def plot_fitting_curve(self, datapoints: DatapointCollection, axes: Axes) -> None:
        """
         plotting fitting curve of datapoints
        :param x_data: x coordinates
        :param y_data: y coordinates
        :param axes: the axes on which to draw the fitting curve
        :return: nothing
        """

        # Extracting distance values / intensities of checked datapoints
        checked_datapoints: DatapointCollection = datapoints.get_active_datapoints()

        checked_distances: list[float] = [
            valueErrorPair.value
            for valueErrorPair in checked_datapoints.get_distances()
        ]

        checked_shifted_intensities: list[float] = [
            valueErrorPair.value
            for valueErrorPair in checked_datapoints.get_shifted_intensities()
        ]

        # Calculating coefficients
        coeffs = np.polyfit(
            checked_distances,
            checked_shifted_intensities,
            int(self.parent.menu_bar.number_of_polynomials.get()),
        )

        poly = np.poly1d(coeffs)  # Creating polynomial with given coefficients

        x_fit = np.linspace(min(checked_distances), max(checked_distances), 100)
        y_fit = poly(x_fit)

        # plot the curve
        axes.plot(x_fit, y_fit, color="red", linestyle="--", linewidth="0.6")

    def plot_derivative_curve(
        self, datapoints: DatapointCollection, axes: Axes
    ) -> None:
        """
         plotting derivative curve of datapoints
        :param x_data: x coordinates
        :param y_data: y coordinates
        :param axes: the axes on which to draw the fitting curve
        :return: nothing
        """

        # Extracting distance values / intensities of checked datapoints
        checked_datapoints: DatapointCollection = datapoints.get_active_datapoints()

        checked_distances = checked_datapoints.get_distances().get_values()

        checked_unshifted_intensities = (
            checked_datapoints.get_unshifted_intensities().get_values()
        )

        # Calculating coefficients
        coeffs = np.polyfit(
            checked_distances,
            checked_unshifted_intensities,
            int(self.parent.menu_bar.number_of_polynomials.get()),
        )

        poly = np.poly1d(coeffs)  # Creating polynomial with given coefficients

        x_fit = np.linspace(min(checked_distances), max(checked_distances), 100)
        y_fit = poly(x_fit)

        # plot the curve
        axes.plot(x_fit, y_fit, color="blue", linestyle="-", linewidth="0.6")
