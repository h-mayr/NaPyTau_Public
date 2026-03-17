from tkinter import Canvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Any, TYPE_CHECKING

from matplotlib.axes import Axes
import customtkinter
import numpy as np
import scipy
from scipy.interpolate import make_lsq_spline, make_splrep

from napytau.gui.components.toolbar import Toolbar
from napytau.gui.model.color import Color
from napytau.gui.model.marker_factory import generate_marker
from napytau.gui.model.marker_factory import generate_error_marker_path

from napytau.core.polynomials import (
    calculate_polynomial_coefficients_for_coupled_fit,
    calculate_polynomial_coefficients_for_fit,
)
from napytau.core.tau import calculate_tau_i_values
from napytau.core.delta_tau import calculate_error_propagation_terms
from napytau.import_export.model.datapoint_collection import DatapointCollection


if TYPE_CHECKING:
    from napytau.gui.app import App  # Import only for the type checking.


class Graph:
    def __init__(self, parent: "App") -> None:
        self.parent = parent
        self._knot_mode: bool = False
        self._knot_distances: list[float] = []
        self._axes_tau_yscale: str = "linear"
        self._axes1_yscale: str = "linear"
        self._axes2_yscale: str = "linear"
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

        # three stacked subplots sharing the x-axis: τᵢ on top, shifted, unshifted
        axes_tau = fig.add_subplot(311)
        axes_1 = fig.add_subplot(312, sharex=axes_tau)
        axes_2 = fig.add_subplot(313, sharex=axes_tau)
        fig.subplots_adjust(left=0.1, bottom=0.07, right=0.9, top=0.95, hspace=0.08)

        # set colors according to appearance mode
        self.set_colors(appearance)

        # apply colors onto all three axes
        self.apply_coloring(fig, axes_tau)
        self.apply_coloring(fig, axes_1)
        self.apply_coloring(fig, axes_2)

        # add grid style to all three axes
        for ax in (axes_tau, axes_1, axes_2):
            ax.grid(
                True,
                which="both",
                color=self.secondary_color,
                linestyle="--",
                linewidth=0.3,
            )
            ax.set_xscale("log")

        axes_tau.set_yscale(self._axes_tau_yscale)
        axes_1.set_yscale(self._axes1_yscale)
        axes_2.set_yscale(self._axes2_yscale)

        # hide x-tick labels on upper two axes (shared; bottom carries them)
        axes_tau.tick_params(labelbottom=False)
        axes_1.tick_params(labelbottom=False)

        # draw tau markers on top axes
        self.plot_tau_values(axes_tau)
        # draw shifted markers + fitting curve on middle axes
        self.plot_shifted_markers(self.parent.datapoints_for_fitting, axes_1)
        # draw unshifted markers + derivative curve on bottom axes
        self.plot_unshifted_markers(self.parent.datapoints_for_fitting, axes_2)

        active = self.parent.datapoints_for_fitting.get_active_datapoints()
        if len(active) > 0:
            self._set_ylim_with_errors(axes_1, active, intensity_index=0)
            self._set_ylim_with_errors(axes_2, active, intensity_index=1)
            self.plot_fitting_curve(self.parent.datapoints_for_fitting, axes_1)
            self.plot_derivative_curve(self.parent.datapoints_for_fitting, axes_2)

        # draw knot markers on all three axes
        self._plot_knot_lines(axes_tau)
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

    def toggle_axes1_yscale(self) -> None:
        """Switch top subplot y-axis between linear and log scale."""
        self._axes1_yscale = "log" if self._axes1_yscale == "linear" else "linear"
        self.parent.after(0, self.update_plot)

    def toggle_axes2_yscale(self) -> None:
        """Switch bottom subplot y-axis between linear and log scale."""
        self._axes2_yscale = "log" if self._axes2_yscale == "linear" else "linear"
        self.parent.after(0, self.update_plot)

    def clear_knots(self) -> None:
        """Remove all user-placed knots and clear dataset sampling_points."""
        self._knot_distances = []
        self.parent.dataset[0].set_sampling_points([])

    def _set_ylim_with_errors(
        self,
        axes: Axes,
        datapoints: DatapointCollection,
        intensity_index: int,
    ) -> None:
        """Set y-axis limits to cover value ± error for all active datapoints."""
        pairs = [dp.get_intensity()[intensity_index] for dp in datapoints]
        y_lo = min(p.value - p.error for p in pairs)
        y_hi = max(p.value + p.error for p in pairs)
        pad = (y_hi - y_lo) * 0.1 if y_hi != y_lo else abs(y_hi) * 0.1 + 1.0
        lo = y_lo - pad
        if axes.get_yscale() == "log":
            lo = max(lo, 1e-10)
        axes.set_ylim(lo, y_hi + pad)

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

    def toggle_axes_tau_yscale(self) -> None:
        """Switch top (τ) subplot y-axis between linear and log scale."""
        self._axes_tau_yscale = "log" if self._axes_tau_yscale == "linear" else "linear"
        self.parent.after(0, self.update_plot)

    def plot_tau_values(self, axes: Axes) -> None:
        """Plot per-datapoint τᵢ ± Δτᵢ values vs distances on the given axes."""
        dataset = self.parent.active_dataset
        try:
            tau_factor = (
                self.parent.control_panel.timescale.get()
                if hasattr(self.parent, "control_panel")
                else 0.0
            )
            fit_mode = getattr(self.parent, "fit_mode", "lsq")
            if fit_mode == "coupled":
                coefficients, knots = calculate_polynomial_coefficients_for_coupled_fit(
                    dataset,
                    tau_factor if tau_factor > 0 else 1.0,
                    self.parent.polynomial_degree,
                )
            else:
                coefficients, knots = calculate_polynomial_coefficients_for_fit(
                    dataset,
                    self.parent.polynomial_degree,
                    self.parent.smoothing_factor,
                )
            tau_i = calculate_tau_i_values(
                dataset, coefficients, knots, self.parent.polynomial_degree
            )
            delta_tau_i = calculate_error_propagation_terms(
                dataset, coefficients, tau_factor, knots, self.parent.polynomial_degree
            )
            distances = np.array(dataset.get_datapoints().get_distances().get_values())
            axes.errorbar(
                distances,
                tau_i,
                yerr=delta_tau_i,
                fmt="o",
                markersize=4,
                color=self.main_marker_color,
                ecolor=self.main_marker_color,
                capsize=2,
                linewidth=0.8,
            )
        except Exception:
            pass

    def _fit_spline_for_display(
        self,
        distances: np.ndarray,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Fit a B-spline through (distances, values) using the current fit settings.

        Returns (d_fit, y_fit) arrays for plotting, or None if fitting fails.
        """
        dataset = self.parent.dataset[0]
        velocity = dataset.get_relative_velocity().value.get_velocity()
        if velocity == 0:
            return None

        c = scipy.constants.speed_of_light
        times = distances / (velocity * c)

        sort_idx = np.argsort(times)
        times_s = times[sort_idx]
        values_s = values[sort_idx]
        distances_s = distances[sort_idx]

        degree = self.parent.polynomial_degree
        fit_mode = getattr(self.parent, "fit_mode", "lsq")
        # For coupled mode, display a regular LSQ spline (coupled fit is for τ calc)
        smoothing_factor = (
            None if fit_mode == "coupled" else self.parent.smoothing_factor
        )
        sampling_points = dataset.get_sampling_points()

        t_min, t_max = times_s[0], times_s[-1]
        interior_knots = (
            np.array(sorted(t for t in sampling_points if t_min < t < t_max))
            if sampling_points
            else np.array([])
        )

        try:
            if smoothing_factor is not None:
                spline = make_splrep(times_s, values_s, k=degree, s=smoothing_factor)
            elif len(interior_knots) > 0:
                t_full = np.concatenate(
                    [
                        [times_s[0]] * (degree + 1),
                        interior_knots,
                        [times_s[-1]] * (degree + 1),
                    ]
                )
                spline = make_lsq_spline(times_s, values_s, t=t_full, k=degree)
            else:
                spline = make_splrep(times_s, values_s, k=degree)
        except Exception:
            return None

        d_fit = np.linspace(distances_s[0], distances_s[-1], 300)
        t_fit = d_fit / (velocity * c)
        return d_fit, np.asarray(spline(t_fit))

    def plot_fitting_curve(self, datapoints: DatapointCollection, axes: Axes) -> None:
        """Plot the B-spline fit through shifted intensities on the given axes."""
        active = datapoints.get_active_datapoints()
        distances = np.array(active.get_distances().get_values())
        shifted = np.array(active.get_shifted_intensities().get_values())

        result = self._fit_spline_for_display(distances, shifted)
        if result is None:
            return
        d_fit, y_fit = result
        axes.plot(d_fit, y_fit, color="red", linestyle="--", linewidth=0.6)

    def plot_derivative_curve(
        self, datapoints: DatapointCollection, axes: Axes
    ) -> None:
        """Plot the B-spline fit through unshifted intensities on the given axes."""
        active = datapoints.get_active_datapoints()
        distances = np.array(active.get_distances().get_values())
        unshifted = np.array(active.get_unshifted_intensities().get_values())

        result = self._fit_spline_for_display(distances, unshifted)
        if result is None:
            return
        d_fit, y_fit = result
        axes.plot(d_fit, y_fit, color="blue", linestyle="-", linewidth=0.6)
