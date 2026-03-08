import customtkinter
from typing import TYPE_CHECKING

from napytau.gui.model.log_message_type import LogMessageType

from napytau.core.core import (
    calculate_optimal_tau_factor,
    calculate_lifetime_for_custom_tau_factor,
)

if TYPE_CHECKING:
    from napytau.gui.app import App  # Import only for the type checking.


class ControlPanel(customtkinter.CTkFrame):
    def __init__(self, parent: "App"):
        """

        The panel with all the controls like timescale input,
        buttons for minimizing chi squared, calculating tau and absolute tau
        and displaying all results.

        :param parent: Parent widget to host the control panel.
        """
        super().__init__(parent, corner_radius=10)
        self.parent = parent

        # Main area for buttons and controls
        self.grid(
            row=2, column=1, rowspan=2, padx=(0, 10), pady=(10, 10), sticky="nsew"
        )
        self.grid_rowconfigure((0, 1), weight=1)
        self.grid_propagate(True)

        self.timescale = customtkinter.DoubleVar(value=1.0)

        self.result_chi_squared = customtkinter.StringVar(value="N/A")
        self.result_tau = customtkinter.StringVar(value="N/A")
        self.result_tau_error = customtkinter.StringVar(value="N/A")
        self.result_absolute_tau_t = customtkinter.StringVar(value="N/A")

        self._create_widgets()

    def _create_widgets(self) -> None:
        """
        Create the control panel widgets.
        """
        # Row 1: Timescale Controls
        timescale_widget = self._create_timescale_widget()
        timescale_widget.pack(fill="x", padx=5, pady=5)

        # Row 2: Chi squared display
        chi_squared_widget = self._create_chi_squared_widget()
        chi_squared_widget.pack(fill="x", padx=5, pady=5)

        # Row 3: Tau display
        tau_widget = self._create_tau_widget()
        tau_widget.pack(fill="x", padx=5, pady=5)

    def _create_timescale_widget(self) -> customtkinter.CTkFrame:
        """
        Create the timescale widget.
        """

        timescale_min = 0.01
        timescale_max = 100.0

        frame = customtkinter.CTkFrame(self)
        frame.columnconfigure(0, weight=1)  # Button "t [ps]"
        frame.columnconfigure(1, weight=1)  # Button "+0.1[ps]"
        frame.columnconfigure(2, weight=1)  # Button "-0.1[ps]"
        frame.columnconfigure(
            3, weight=2
        )  # Entry field (More weight to make it bigger)

        self._tau_factor_var = customtkinter.StringVar(value=str(self.timescale.get()))
        tau_factor = self._tau_factor_var

        def update_timescale() -> None:
            try:
                value = float(tau_factor.get())

                if timescale_min <= value <= timescale_max:
                    self.timescale.set(value)
                    self.parent.logger.log_message(
                        f"Timescale set to: {value}", LogMessageType.INFO
                    )
                    lifetime = calculate_lifetime_for_custom_tau_factor(
                        self.parent.active_dataset,
                        value,
                        self.parent.polynomial_degree,
                        self.parent.smoothing_factor,
                    )

                    self._apply_lifetime(lifetime)
                    self._tau_button_event(lifetime[0], lifetime[1])

                else:
                    self.parent.logger.log_message(
                        f"Error: Value out of valid range ({timescale_min:.2f}"
                        f" - {timescale_max:.2f}).",
                        LogMessageType.ERROR,
                    )
            except ValueError:
                self.parent.logger.log_message(
                    "Invalid input value, please enter a number.", LogMessageType.ERROR
                )

        # connects slider value to lifetime Entry results

        def sync_slider(value: float) -> None:
            if self._check_dataset_set():
                tau_factor.set(f"{value:.2f}")
                lifetime = calculate_lifetime_for_custom_tau_factor(
                    self.parent.active_dataset,
                    value,
                    self.parent.polynomial_degree,
                    self.parent.smoothing_factor,
                )

                self._apply_lifetime(lifetime)

        update_timescale_button = customtkinter.CTkButton(
            frame,
            text="t [ps]",
            command=update_timescale,
            width=10,
        )
        update_timescale_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Function for adding and subtracting from tau factor in 0.1ps steps

        add_on_tau_factor = (
            lambda value: tau_factor.set(f"{float(tau_factor.get()) + value:.2f}")
            if float(tau_factor.get()) + value >= 0.0
            else None
        )

        # Create buttons for adding and subtracting
        add_taufactor_button = customtkinter.CTkButton(
            frame,
            text="+0.1[ps]",
            command=lambda: add_on_tau_factor(0.1),
            width=10,
        )

        subtract_taufactor_button = customtkinter.CTkButton(
            frame,
            text="-0.1[ps]",
            command=lambda: add_on_tau_factor(-0.1),
            width=10,
        )

        # Create Entry and Slider
        entry = customtkinter.CTkEntry(
            frame, textvariable=tau_factor, justify="right", width=80
        )

        slider = customtkinter.CTkSlider(
            frame,
            from_=timescale_min,
            to=timescale_max,
            variable=self.timescale,
            command=sync_slider,
        )

        # Layout:
        add_taufactor_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        subtract_taufactor_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        entry.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        slider.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky="ew")

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        return frame

    def _create_chi_squared_widget(self) -> customtkinter.CTkFrame:
        """
        Create the chi squared widget.
        """
        frame = customtkinter.CTkFrame(self)
        frame.columnconfigure(2, weight=1)

        button = customtkinter.CTkButton(
            frame, text="Minimize", command=self._chi_squared_button_event
        )

        button.grid(row=0, column=0, padx=5, pady=5)

        label = customtkinter.CTkLabel(frame, text="χ²:")
        label.grid(row=0, column=1, padx=5, pady=5)

        result = customtkinter.CTkEntry(
            frame,
            textvariable=self.result_chi_squared,
            state="readonly",
            justify="right",
            width=0,
        )

        result.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        return frame

    def _create_tau_widget(self) -> customtkinter.CTkFrame:
        """
        Create the tau widget.
        """
        frame = customtkinter.CTkFrame(self)

        # Make all columns expand equally

        frame.columnconfigure(2, weight=1)

        button = customtkinter.CTkButton(
            frame, text="τ ± Δτ [ps]:", command=lambda: self._tau_button_event(0.0, 0.0)
        )

        label = customtkinter.CTkLabel(frame, text="|τ - t| [ps]:")

        result_tau_time_difference = customtkinter.CTkEntry(
            frame, textvariable=self.result_absolute_tau_t, state="readonly", width=150
        )

        # Secondary frame
        frame_secondary = customtkinter.CTkFrame(frame)
        frame_secondary.grid(
            row=1, column=0, columnspan=4, padx=0, pady=0, sticky="nsew"
        )

        # Make sure secondary frame columns expand as well
        frame_secondary.columnconfigure(0, weight=1)
        frame_secondary.columnconfigure(2, weight=1)

        result = customtkinter.CTkEntry(
            frame_secondary, textvariable=self.result_tau, state="readonly", width=100
        )
        separator = customtkinter.CTkLabel(frame_secondary, text="±")
        error = customtkinter.CTkEntry(
            frame_secondary,
            textvariable=self.result_tau_error,
            state="readonly",
            width=100,
        )

        # Layout adjustments
        button.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        label.grid(row=0, column=1, padx=0, pady=5, sticky="nsew")
        result_tau_time_difference.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        result.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        separator.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        error.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")

        return frame

    def _create_absolute_tau_t_widget(self) -> customtkinter.CTkFrame:
        """
        Create the absolute tau t widget.

        """
        frame = customtkinter.CTkFrame(self)
        frame.columnconfigure(2, weight=1)

        button = customtkinter.CTkButton(
            frame, text="Absolute τ", command=self._absolute_tau_button_event
        )
        button.grid(row=0, column=0, padx=5, pady=5)

        label = customtkinter.CTkLabel(frame, text="|τ - t| [ps]:")
        label.grid(row=0, column=1, padx=5, pady=5)

        result = customtkinter.CTkEntry(
            frame, textvariable=self.result_absolute_tau_t, state="readonly", width=100
        )

        result.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        return frame

    def _timescale_button_event(self) -> None:
        """
        Event if the timescale button is clicked.
        """

    def _timescale_slider_event(self, value: str) -> None:
        """
        Event for the timescale slider.
        :param value: The current value of the slider.
        """
        self.timescale.set(round(float(value), 2))

    def _apply_lifetime(self, lifetime: tuple[float, float]) -> None:
        """Set τ ± Δτ display, logging an error if τ is negative."""
        tau, delta_tau = lifetime
        if tau < 0:
            self.parent.logger.log_message(
                f"Calculation error: τ = {tau:.4g} ps is negative.",
                LogMessageType.ERROR,
            )
            return
        self.result_tau.set(str(tau))
        self.result_tau_error.set(str(delta_tau))

    def _tau_button_event(self, value: float, error: float) -> None:
        """
        Event if the tau button is clicked.
        """
        self.set_result_tau(value)
        self.set_result_tau_error(error)

    def _chi_squared_button_event(self) -> None:
        """
        Event if the chi2 button is clicked.
        """
        if self._check_dataset_set():
            optimal_tau = calculate_optimal_tau_factor(
                self.parent.active_dataset,
                (5, 100),
                1.0,
                self.parent.polynomial_degree,
                self.parent.smoothing_factor,
            )
            self.set_result_chi_squared(optimal_tau)

            # Update slider and entry field to the optimal tau factor
            self.timescale.set(optimal_tau)
            self._tau_factor_var.set(f"{optimal_tau:.2f}")

            # Recalculate τ ± Δτ for the new tau factor
            lifetime = calculate_lifetime_for_custom_tau_factor(
                self.parent.active_dataset,
                optimal_tau,
                self.parent.polynomial_degree,
                self.parent.smoothing_factor,
            )
            self._apply_lifetime(lifetime)

    def _absolute_tau_button_event(self) -> None:
        """
        Event if the absolute tau button is clicked.
        """

        self.set_result_absolute_tau_t(0.0)

    def set_result_chi_squared(self, chi_squared: float) -> None:
        """
        Set the chi squared result.
        :param chi_squared: The new value for chi squared.
        """
        self.result_chi_squared.set(chi_squared)

    def set_result_tau(self, tau: float) -> None:
        """
        Set the tau result.

        :param tau: The new value for tau.
        """
        self.result_tau.set(tau)

    def set_result_tau_error(self, tau_error: float) -> None:
        """

        Set the tau error result.

        :param tau_error: The new value for the tau error.
        """
        self.result_tau_error.set(tau_error)

    def set_result_absolute_tau_t(self, absolute_tau_t: float) -> None:
        """
        Set the absolute tau result.
        :param absolute_tau_t: The new value for the absolute tau value.
        """
        self.result_absolute_tau_t.set(absolute_tau_t)

    def recalculate(self) -> None:
        """
        Re-run the lifetime calculation with the current tau factor and
        update the result displays. Called automatically after knots change.
        """
        if not self._check_dataset_set():
            return
        try:
            value = self.timescale.get()
            lifetime = calculate_lifetime_for_custom_tau_factor(
                self.parent.active_dataset,
                value,
                self.parent.polynomial_degree,
                self.parent.smoothing_factor,
            )
            self._apply_lifetime(lifetime)
        except Exception as e:
            self.parent.logger.log_message(
                f"Calculation failed: {e}", LogMessageType.ERROR
            )

    def _check_dataset_set(self) -> bool:
        """
        Check if a dataset is set.
        :return: True if a dataset is set, False otherwise.
        """
        if self.parent.dataset is None:
            self.parent.logger.log_message(
                "No dataset loaded yet. Please load a dataset first.",
                LogMessageType.ERROR,
            )
            return False
        return True
