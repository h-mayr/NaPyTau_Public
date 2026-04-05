from pathlib import PurePath
from typing import List, Tuple

import tkinter as tk
from tkinter import filedialog

import customtkinter

from napytau.cli.cli_arguments import CLIArguments

from napytau.gui.components.checkbox_panel import CheckboxPanel
from napytau.gui.components.control_panel import ControlPanel
from napytau.gui.components.graph import Graph
from napytau.gui.components.logger import Logger, LogMessageType
from napytau.gui.components.menu_bar import MenuBar
from napytau.gui.components.toolbar import Toolbar
from napytau.import_export.import_export import (
    IMPORT_FORMAT_LEGACY,
    import_napytau_format_from_file,
    import_legacy_format_from_files,
    read_legacy_setup_data_into_data_set,
    read_napytau_setup_data_into_data_set,
)

from napytau.import_export.model.datapoint_collection import DatapointCollection
from napytau.import_export.model.dataset import DataSet
from napytau.import_export.model.relative_velocity import RelativeVelocity
from napytau.util.model.value_error_pair import ValueErrorPair

# Modes: "System" (standard), "Dark", "Light"
customtkinter.set_appearance_mode("System")
# Themes: "blue" (standard), "green", "dark-blue"
customtkinter.set_default_color_theme("blue")


class App(customtkinter.CTk):
    def __init__(self) -> None:
        """
        Constructor for the GUIApp, initializes the GUI.
        This is the logical entry point into the GUI.
        """
        super().__init__()

        self.dataset: Tuple[DataSet, List[dict]] = (
            DataSet(
                ValueErrorPair(RelativeVelocity(0), RelativeVelocity(0)),
                DatapointCollection([]),
            ),
            [],
        )
        # Datapoints
        self.datapoints_for_fitting: DatapointCollection = DatapointCollection([])
        self.datapoints_for_calculation: DatapointCollection = DatapointCollection([])

        # values
        self.tau = tk.IntVar()
        self.tau.set(2)

        # Fit mode: "lsq" | "smooth" | "coupled"
        self.fit_mode: str = "lsq"
        self.smoothing_factor: float | None = None
        self.polynomial_degree: int = 2

        # Knot auto-spacing
        self.knot_spacing_mode: str = "manual"
        self.n_auto_knots: int = 3

        # Monte Carlo iterations (0 = disabled)
        self.n_mc_iterations: int = 0

        # configure window
        self.title("NaPyTau")
        width = 1366
        height = 768
        self.geometry(f"{width}x{height}")

        """
        Configure grid. Current Layout:
        Four rows, two columns with
        - Menu bar in row 0
        - Graph from row 1 to 2, column 0
        - Checkbox panel in row 1, column 1
        - Control panel from row 2 to 3, column 1
        - Logger in row 3, column 0
        """
        # Row ratio: 3/8, 3/8, 1/4
        total_height = 8  # 3+3+2 = 8 parts
        self.grid_rowconfigure(0, weight=0)  # Menubar, weight=0 means minimal size
        self.grid_rowconfigure(1, weight=3, minsize=3 * height // total_height)
        # Reduce graph height by 30 to asure all components and their
        # separators are inside the window.
        self.grid_rowconfigure(2, weight=3, minsize=3 * height // total_height - 30)
        self.grid_rowconfigure(3, weight=2, minsize=2 * height // total_height)

        # column ratio: 2/3, 1/3
        total_width = 4  # 2+1 = 3 parts
        # Reduce graph width by 30 to asure all components and their
        # separators are inside the window.
        self.grid_columnconfigure(0, weight=2, minsize=3 * width // total_width - 30)
        self.grid_columnconfigure(1, weight=1, minsize=1 * width // total_width)

        # Define menu bar callback functions
        menu_bar_callbacks = {
            "open_directory": self.open_directory,
            "save_file": self.save_file,
            "read_setup": self.read_setup,
            "quit": self.quit,
            "change_appearance_mode": self.change_appearance_mode,
            "select_alpha_calc_mode": self.select_alpha_calc_mode,
            "set_fit_mode": self.set_fit_mode,
            "set_degree": self.set_degree,
            "set_smoothing_factor": self.set_smoothing_factor,
            "set_knot_spacing_mode": self.set_knot_spacing_mode,
            "set_n_auto_knots": self.set_n_auto_knots,
            "set_n_mc_iterations": self.set_n_mc_iterations,
        }

        # Initialize the menu bar
        self.menu_bar = MenuBar(self, menu_bar_callbacks)

        # Initialize the checkbox panel
        self.checkbox_panel = CheckboxPanel(self)

        # Initialize the graph
        self.graph: Graph = Graph(self)

        self.toolbar: Toolbar = Toolbar(self, self.graph.canvas)

        # Initialize the control panel
        self.control_panel: ControlPanel = ControlPanel(self)

        # Initialize the logger
        self.logger: Logger = Logger(self)

    def open_directory(self, mode: str) -> None:
        """
        Opens the file explorer and lets the user choose a file to open.
        """

        if mode == IMPORT_FORMAT_LEGACY:
            directory_path = filedialog.askdirectory(
                title="Choose directory",
                initialdir=".",
            )

            if directory_path:
                self.dataset = (
                    import_legacy_format_from_files(PurePath(directory_path)),
                    [],
                )
                self.logger.log_message(
                    f"chosen directory: {directory_path}", LogMessageType.INFO
                )

        else:
            file_path = filedialog.askopenfilename(
                title="Choose directory",
                filetypes=[("NaPyTau files", "*.json")],
            )

            if file_path:
                self.dataset = import_napytau_format_from_file(PurePath(file_path))
                self.logger.log_message(
                    f"chosen directory: {file_path}", LogMessageType.INFO
                )

        if len(self.dataset) > 0:
            self._reset_state()
            self.update_data_checkboxes()
            self.graph.update_plot()

    def save_file(self) -> None:
        """
        Saves the file.
        """
        self.logger.log_message("Saved file", LogMessageType.SUCCESS)

    def read_setup(self, mode: str) -> None:
        """
        Reads the setup.
        """

        if mode == IMPORT_FORMAT_LEGACY:
            file_path = filedialog.askopenfilename(
                title="Choose setup file",
                filetypes=[("Legacy setup files", "*.napset")],
                initialdir=".",
            )

            self.dataset = (
                read_legacy_setup_data_into_data_set(
                    self.dataset[0], PurePath(file_path)
                ),
                self.dataset[1],
            )

        else:
            if len(self.dataset) == 0 or len(self.dataset[1]) == 0:
                self.logger.log_message(
                    "No dataset loaded yet. Please load a dataset first.",
                    LogMessageType.ERROR,
                )
                return

            popup = customtkinter.CTkToplevel(self)
            popup.title("Select Setup")
            popup.geometry("300x150")

            self.update_idletasks()
            x = self.winfo_x() + (self.winfo_width() // 2) - (300 // 2)
            y = self.winfo_y() + (self.winfo_height() // 2) - (150 // 2)
            popup.geometry(f"+{x}+{y}")

            selected_setup = tk.StringVar(value="Choose setup")

            setup_names = list(map(lambda setup: setup["name"], self.dataset[1]))
            optionmenu = customtkinter.CTkOptionMenu(
                popup,
                values=setup_names,
                variable=selected_setup,
            )
            optionmenu.pack(pady=20)

            confirm_btn = customtkinter.CTkButton(
                popup,
                text="Confirm",
                command=lambda: self.confirm_selection(popup, selected_setup),
            )
            confirm_btn.pack(pady=10)

    def confirm_selection(
        self, popup: tk.Toplevel, selected_setup: tk.StringVar
    ) -> None:
        """
        Confirms the selected setup and closes the popup.
        """
        if selected_setup.get() == "Choose setup":
            self.logger.log_message("Please choose a setup.", LogMessageType.ERROR)
            return

        value = selected_setup.get()
        read_napytau_setup_data_into_data_set(
            self.dataset[0],
            self.dataset[1],
            value,
        )
        popup.destroy()
        self.logger.log_message(f"Setup '{value}' loaded.", LogMessageType.SUCCESS)

    def quit(self) -> None:
        """
        Quits the program.
        """
        self.destroy()

    def change_appearance_mode(self) -> None:
        """
        Changes the appearance mode to the variable appearance_mode.
        """
        customtkinter.set_appearance_mode(self.menu_bar.appearance_mode.get())
        self.logger.switch_logger_appearance(self.menu_bar.appearance_mode.get())
        self.graph.update_plot()

    def select_alpha_calc_mode(self) -> None:
        """
        Selects the alpha calculation mode.
        """
        self.logger.log_message(
            "Alpha calculation set to "
            + self.menu_bar.alpha_calc_mode.get()
            + " but not implemented yet!",
            LogMessageType.ERROR,
        )

    def set_fit_mode(self, mode: str) -> None:
        """Switch between LSQ, Smooth, and Coupled fit modes."""
        self.fit_mode = mode
        if mode == "lsq":
            self.smoothing_factor = None
        elif mode == "smooth":
            try:
                self.smoothing_factor = float(self.menu_bar.smoothing_var.get())
            except ValueError:
                self.smoothing_factor = 1.0
        # "coupled" → smoothing_factor irrelevant, leave as-is
        self.after(0, self._refresh_graph_and_calculation)

    def set_knot_spacing_mode(self, mode: str) -> None:
        """Set the auto-knot spacing mode and regenerate knots if not manual."""
        self.knot_spacing_mode = mode
        if mode != "manual":
            self._generate_auto_knots()
        self.after(0, self._refresh_graph_and_calculation)

    def set_n_auto_knots(self, n: int) -> None:
        """Set the number of interior auto-knots and regenerate."""
        self.n_auto_knots = n
        if self.knot_spacing_mode != "manual":
            self._generate_auto_knots()
        self.after(0, self._refresh_graph_and_calculation)

    def set_n_mc_iterations(self, n: int) -> None:
        """Set the number of Monte Carlo iterations."""
        self.n_mc_iterations = n
        self.after(0, self._refresh_graph_and_calculation)

    def _generate_auto_knots(self) -> None:
        """Compute interior knot positions and store them as sampling_points."""
        import numpy as np
        import scipy.constants

        active = self.active_dataset
        distances = active.get_datapoints().get_distances().get_values()
        if len(distances) < 2:
            return
        d_min, d_max = min(distances), max(distances)
        v = active.get_relative_velocity().value.get_velocity()
        c = scipy.constants.speed_of_light
        if v == 0:
            return
        n = self.n_auto_knots + 2  # include endpoints then slice interior
        if self.knot_spacing_mode == "equidistant":
            d_knots = np.linspace(d_min, d_max, n)[1:-1]
        else:  # "log"
            d_knots = np.geomspace(d_min, d_max, n)[1:-1]
        t_knots = (d_knots / (v * c)).tolist()
        self.dataset[0].set_sampling_points(t_knots)
        self.graph._knot_distances = d_knots.tolist()

    def set_degree(self, degree: int) -> None:
        """Set the B-spline degree and refresh."""
        self.polynomial_degree = degree
        self.after(0, self._refresh_graph_and_calculation)

    def set_smoothing_factor(self, value: float) -> None:
        """Set the smoothing factor, switch to Smooth mode, and refresh."""
        self.smoothing_factor = value
        self.menu_bar.fit_mode.set("smooth")
        self.after(0, self._refresh_graph_and_calculation)

    def _refresh_graph_and_calculation(self) -> None:
        """Update the graph and recalculate τ (deferred so menus close first)."""
        self.graph.update_plot()
        self.control_panel.recalculate()
        self.control_panel.update_status()

    def _reset_state(self) -> None:
        """Clear all per-dataset state before loading a new dataset."""
        self.datapoints_for_fitting = DatapointCollection([])
        self.datapoints_for_calculation = DatapointCollection([])
        self.graph._knot_distances = []
        self.graph.clear_knots()

    def update_data_checkboxes(self) -> None:
        """
        Updates the datapoint for the gui and updates both columns of the
        data checkboxes.
        Call this method if there are new datapoints.
        """
        for point in self.dataset[0].get_datapoints():
            self.datapoints_for_fitting.add_datapoint(point)
            self.datapoints_for_calculation.add_datapoint(point)

        self.checkbox_panel.update_data_checkboxes_fitting()
        self.checkbox_panel.update_data_checkboxes_calculation()

    @property
    def active_dataset(self) -> DataSet:
        """Return a DataSet with only the fitting-active datapoints."""
        ds = self.dataset[0]
        return DataSet(
            relative_velocity=ds.get_relative_velocity(),
            datapoints=ds.get_datapoints().get_active_datapoints(),
            sampling_points=ds.get_sampling_points(),
        )

    @property
    def active_dataset_for_calculation(self) -> DataSet:
        """Return a DataSet with only the calculation-active datapoints."""
        ds = self.dataset[0]
        return DataSet(
            relative_velocity=ds.get_relative_velocity(),
            datapoints=ds.get_datapoints().get_active_for_calculation_datapoints(),
            sampling_points=ds.get_sampling_points(),
        )

    def get_datapoints(self) -> DatapointCollection:
        """
        Returns the datapoints for fitting and calculation.
        """

        return self.dataset[0].get_datapoints()


def init(cli_arguments: CLIArguments) -> None:
    app = App()
    app.mainloop()
