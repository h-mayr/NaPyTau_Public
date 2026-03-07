import tkinter as tk
from tkinter import Frame
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from napytau.gui.app import App

from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk, FigureCanvasTkAgg


class CustomToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas: FigureCanvasTkAgg, window: Frame, parent: "App") -> None:
        super().__init__(canvas, window)
        self._app = parent

        # Change background color of the message label
        self._message_label.config(
            bg=parent.graph.main_color,
            fg=parent.graph.secondary_color,
            font=("Arial", 10),
        )

        # Customize specific buttons
        for toolitem in self.toolitems:
            tool_name = toolitem[0]  # Get button name
            if tool_name in self._buttons:
                self._buttons[tool_name].config(
                    bg="green", relief="flat", highlightthickness=0
                )

        # Remove superfluous buttons
        self.winfo_children()[9].destroy()
        self.winfo_children()[6].destroy()

        # Knot mode toggle button — reflects current state of graph._knot_mode
        knot_active = parent.graph._knot_mode
        self._knot_button = tk.Button(
            self,
            text="✦ Knots: ON" if knot_active else "✦ Knots: OFF",
            bg="#ff8c00" if knot_active else "green",
            fg="black",
            relief="flat",
            highlightthickness=0,
            command=self._toggle_knot_mode,
        )
        self._knot_button.pack(side=tk.LEFT, padx=2)

    def _toggle_knot_mode(self) -> None:
        self._app.graph.toggle_knot_mode()
        if self._app.graph._knot_mode:
            self._knot_button.config(text="✦ Knots: ON", bg="#ff8c00")
        else:
            self._knot_button.config(text="✦ Knots: OFF", bg="green")
