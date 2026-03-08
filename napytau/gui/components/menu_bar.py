import tkinter as tk
from tkinter import simpledialog
import customtkinter
from typing import TYPE_CHECKING, Union

from napytau.gui.components.logger import LogMessageType


from napytau.import_export.import_export import (
    IMPORT_FORMAT_NAPYTAU,
    IMPORT_FORMAT_LEGACY,
)


if TYPE_CHECKING:
    from napytau.gui.app import App  # Import only for the type checking.


def open_dropdown_menu(dropdown_menu: tk.Menu, button: customtkinter.CTkButton) -> None:
    """
    On the given dropdown menu on the position of the given button.
    :param dropdown_menu: The dropdown menu to open.
    :param button: The button on which position the menu will be opened.
    """
    dropdown_menu.post(
        button.winfo_rootx(), button.winfo_rooty() + button.winfo_height()
    )


class MenuBar(customtkinter.CTkFrame):
    def __init__(self, parent: "App", callbacks: dict) -> None:
        """
        Initializes the menu bar and its items.
        :param parent: Parent widget to host the menubar.
        :param callbacks: The dictionary of callback functions for the menu bar.
        """
        super().__init__(parent)
        self.callbacks = callbacks

        self.parent = parent

        self.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.pack_propagate(False)

        # Setting up default values
        self.appearance_mode = tk.StringVar(value="system")
        self.alpha_calc_mode = tk.StringVar(value="sum ratio")
        self.mode = tk.StringVar(value=IMPORT_FORMAT_LEGACY)
        self.mode.trace_add("write", self.on_mode_change)
        self.fit_mode = tk.StringVar(value="lsq")
        self.degree_var = tk.StringVar(value=str(parent.polynomial_degree))
        self.smoothing_var = tk.StringVar(value="1.0")

        self._create_file_button()
        self._create_view_button()
        self._create_alpha_calc_button()
        self._create_mode_menu_button()
        self._create_fit_menu_button()

    def _create_file_button(self) -> None:
        """
        Creates the button in the menubar for all file operations.
        """
        self.file_menu = tk.Menu(self, tearoff=0)

        # Declare file_button in advance for type checking
        self.file_button: Union[customtkinter.CTkButton, None] = None

        self.file_button = customtkinter.CTkButton(
            self,
            text="File",
            command=lambda: open_dropdown_menu(self.file_menu, self.file_button),
        )
        self.file_button.grid(row=0, column=0, padx=5, pady=5)

        self.file_menu.add_command(
            label="Open",
            command=lambda: self.callbacks["open_directory"](self.mode.get()),
        )
        self.file_menu.add_command(label="Save", command=self.callbacks["save_file"])
        self.file_menu.add_command(
            label="Read Setup",
            command=lambda: self.callbacks["read_setup"](self.mode.get()),
        )
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.callbacks["quit"])

    def _create_view_button(self) -> None:
        """
        Creates the view button in the menubar.
        """
        self.view_menu = tk.Menu(self, tearoff=0)

        # Declare view_button in advance for type checking
        self.view_button: Union[customtkinter.CTkButton, None] = None

        self.view_button = customtkinter.CTkButton(
            self,
            text="View",
            command=lambda: open_dropdown_menu(self.view_menu, self.view_button),
        )
        self.view_button.grid(row=0, column=1, padx=5, pady=5)

        self.view_menu.add_radiobutton(
            label="Light Mode",
            variable=self.appearance_mode,
            value="light",
            command=self.callbacks["change_appearance_mode"],
        )
        self.view_menu.add_radiobutton(
            label="Dark Mode",
            variable=self.appearance_mode,
            value="dark",
            command=self.callbacks["change_appearance_mode"],
        )
        self.view_menu.add_radiobutton(
            label="System Mode",
            variable=self.appearance_mode,
            value="system",
            command=self.callbacks["change_appearance_mode"],
        )

    def _create_alpha_calc_button(self) -> None:
        """
        Creates the button for the alpha calculation settings in the menubar.
        """
        self.alpha_calc_menu = tk.Menu(self, tearoff=0)

        # Declare alpha_calc_button in advance for type checking
        self.alpha_calc_button: Union[customtkinter.CTkButton, None] = None

        self.alpha_calc_button = customtkinter.CTkButton(
            self,
            text="Alpha calculation",
            command=lambda: open_dropdown_menu(
                self.alpha_calc_menu, self.alpha_calc_button
            ),
        )
        self.alpha_calc_button.grid(row=0, column=2, padx=5, pady=5)

        self.alpha_calc_menu.add_radiobutton(
            label="Sum Ratio",
            variable=self.alpha_calc_mode,
            value="sum ratio",
            command=self.callbacks["select_alpha_calc_mode"],
        )
        self.alpha_calc_menu.add_radiobutton(
            label="Weighted Mean",
            variable=self.alpha_calc_mode,
            value="weighted mean",
            command=self.callbacks["select_alpha_calc_mode"],
        )

    def _create_mode_menu_button(self) -> None:
        """
        Create the Mode menu. Allowing the user to switch the import/export mode
        between `legacy` and `napytau`.
        """

        self.mode_menu = tk.Menu(self, tearoff=0)

        self.mode_button: Union[customtkinter.CTkButton, None] = None

        self.mode_button = customtkinter.CTkButton(
            self,
            text="Mode",
            command=lambda: open_dropdown_menu(self.mode_menu, self.mode_button),
        )

        self.mode_button.grid(row=0, column=3, padx=5, pady=5)

        self.mode_menu.add_radiobutton(
            label="Legacy",
            variable=self.mode,
            value=IMPORT_FORMAT_LEGACY,
        )
        self.mode_menu.add_radiobutton(
            label="Napytau",
            variable=self.mode,
            value=IMPORT_FORMAT_NAPYTAU,
        )

    def _create_fit_menu_button(self) -> None:
        """Creates the Fit dropdown menu (fit method, degree, smoothing factor)."""
        self.fit_menu = tk.Menu(self, tearoff=0)

        self.fit_button: Union[customtkinter.CTkButton, None] = None

        self.fit_button = customtkinter.CTkButton(
            self,
            text="Fit",
            command=lambda: open_dropdown_menu(self.fit_menu, self.fit_button),
        )
        self.fit_button.grid(row=0, column=4, padx=5, pady=5)

        # Fit method
        self.fit_menu.add_radiobutton(
            label="LSQ",
            variable=self.fit_mode,
            value="lsq",
            command=self._on_fit_mode_change,
        )
        self.fit_menu.add_radiobutton(
            label="Smooth",
            variable=self.fit_mode,
            value="smooth",
            command=self._on_fit_mode_change,
        )
        self.fit_menu.add_radiobutton(
            label="Coupled (shifted+unshifted)",
            variable=self.fit_mode,
            value="coupled",
            command=self._on_fit_mode_change,
        )

        self.fit_menu.add_separator()

        # Degree k submenu
        degree_menu = tk.Menu(self.fit_menu, tearoff=0)
        self.fit_menu.add_cascade(label="Degree k", menu=degree_menu)
        for k in range(1, 6):
            degree_menu.add_radiobutton(
                label=str(k),
                variable=self.degree_var,
                value=str(k),
                command=self._on_degree_change,
            )

        self.fit_menu.add_separator()

        # Smoothing factor s submenu
        smooth_menu = tk.Menu(self.fit_menu, tearoff=0)
        self.fit_menu.add_cascade(label="Smoothing factor s", menu=smooth_menu)
        for preset in ("0.01", "0.1", "0.5", "1.0", "2.0", "5.0", "10.0"):
            smooth_menu.add_radiobutton(
                label=preset,
                variable=self.smoothing_var,
                value=preset,
                command=self._on_smoothing_change,
            )
        smooth_menu.add_separator()
        smooth_menu.add_command(label="Custom…", command=self._on_smoothing_custom)

        self.fit_menu.add_separator()

        # Knot spacing submenu
        self.knot_spacing_var = tk.StringVar(value="manual")
        knot_menu = tk.Menu(self.fit_menu, tearoff=0)
        self.fit_menu.add_cascade(label="Knot Spacing", menu=knot_menu)
        for label, value in [
            ("Manual", "manual"),
            ("Equidistant", "equidistant"),
            ("Logarithmic", "log"),
        ]:
            knot_menu.add_radiobutton(
                label=label,
                variable=self.knot_spacing_var,
                value=value,
                command=self._on_knot_spacing_change,
            )
        knot_menu.add_separator()
        knot_menu.add_command(label="Number of knots…", command=self._ask_n_knots)

        self.fit_menu.add_separator()

        # Monte Carlo iterations
        self.fit_menu.add_command(
            label="MC Iterations…", command=self._ask_mc_iterations
        )

    def _on_fit_mode_change(self) -> None:
        self.callbacks["set_fit_mode"](self.fit_mode.get())

    def _on_knot_spacing_change(self) -> None:
        self.callbacks["set_knot_spacing_mode"](self.knot_spacing_var.get())

    def _ask_n_knots(self) -> None:
        value = simpledialog.askinteger(
            "Knots", "Number of interior knots:", minvalue=1, maxvalue=20
        )
        if value is not None:
            self.callbacks["set_n_auto_knots"](value)

    def _ask_mc_iterations(self) -> None:
        value = simpledialog.askinteger(
            "Monte Carlo",
            "Number of MC iterations (0 = disabled):",
            minvalue=0,
            maxvalue=10000,
        )
        if value is not None:
            self.callbacks["set_n_mc_iterations"](value)

    def _on_degree_change(self) -> None:
        self.callbacks["set_degree"](int(self.degree_var.get()))

    def _on_smoothing_change(self) -> None:
        self.callbacks["set_smoothing_factor"](float(self.smoothing_var.get()))

    def _on_smoothing_custom(self) -> None:
        value = simpledialog.askfloat(
            "Smoothing factor",
            "Enter smoothing factor s:",
            minvalue=0.0,
        )
        if value is not None:
            self.smoothing_var.set(str(value))
            self.callbacks["set_smoothing_factor"](value)

    def on_mode_change(self, name: str, index: str, mode_value: str) -> None:
        self.parent.logger.log_message(
            f"Mode changed! New mode: {self.mode.get()}", LogMessageType.INFO
        )
