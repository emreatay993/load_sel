# requirements:
#   pip install PyQt5 PyQtWebEngine plotly
#
# run:
#   python plot_paste_xy.py

import sys
import re
from typing import List

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QGroupBox, QGridLayout
)
from PyQt5.QtWebEngineWidgets import QWebEngineView

import plotly.graph_objs as go
import plotly.io as pio


def parse_numbers(text: str) -> List[float]:
    """
    Parse numbers separated by commas, semicolons, tabs, or whitespace.
    Ignores empty tokens. Uses dot as decimal separator.
    """
    # Normalize common separators to space
    cleaned = re.sub(r"[,\t;]+", " ", text.strip())
    # Also collapse multiple newlines/spaces to single spaces
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return []
    nums = []
    for tok in cleaned.split(" "):
        if not tok:
            continue
        try:
            nums.append(float(tok))
        except ValueError:
            # Try to catch values like "1e-3" with stray characters
            tok2 = re.sub(r"[^\dEe\+\-\.]", "", tok)
            if tok2:
                nums.append(float(tok2))
            else:
                raise
    return nums


class PlotPasteApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Paste → Plot (PyQt5 + Plotly)")
        self.resize(1000, 700)

        # --- Layouts ---
        root = QVBoxLayout(self)

        # Inputs box
        inputs_box = QGroupBox("Input data")
        grid = QGridLayout(inputs_box)

        self.x_edit = QTextEdit()
        self.x_edit.setPlaceholderText("Paste X values here\nExample:\n0, 1, 2, 3, 4")
        self.x_edit.setAcceptRichText(False)

        self.y_edit = QTextEdit()
        self.y_edit.setPlaceholderText("Paste Y values here\nExample:\n0\n1\n4\n9\n16")
        self.y_edit.setAcceptRichText(False)

        grid.addWidget(QLabel("X values"), 0, 0)
        grid.addWidget(QLabel("Y values"), 0, 1)
        grid.addWidget(self.x_edit, 1, 0)
        grid.addWidget(self.y_edit, 1, 1)

        # Buttons row
        btn_row = QHBoxLayout()
        self.plot_btn = QPushButton("Plot")
        self.clear_btn = QPushButton("Clear")
        btn_row.addStretch(1)
        btn_row.addWidget(self.clear_btn)
        btn_row.addWidget(self.plot_btn)

        grid.addLayout(btn_row, 2, 0, 1, 2)

        # Status
        self.status = QLabel("")
        self.status.setStyleSheet("color:#a00;")
        grid.addWidget(self.status, 3, 0, 1, 2)

        root.addWidget(inputs_box)

        # Plot area
        self.view = QWebEngineView()
        root.addWidget(self.view, 1)

        # Connections
        self.plot_btn.clicked.connect(self.on_plot)
        self.clear_btn.clicked.connect(self.on_clear)

        # Demo defaults
        self.x_edit.setText("0, 1, 2, 3, 4, 5, 6")
        self.y_edit.setText("0\n1\n4\n9\n16\n25\n36")

        # Initial plot
        self.on_plot()

    def on_clear(self):
        self.x_edit.clear()
        self.y_edit.clear()
        self.status.setText("")
        self.view.setHtml("<html><body style='font-family:Arial;padding:16px;'>"
                          "<h3>No plot</h3><p>Paste X and Y values, then press <b>Plot</b>.</p>"
                          "</body></html>")

    def on_plot(self):
        self.status.setText("")
        try:
            xs = parse_numbers(self.x_edit.toPlainText())
            ys = parse_numbers(self.y_edit.toPlainText())
        except Exception:
            self.status.setText("Parsing error. Use numbers separated by commas, spaces, or newlines. Decimal separator must be '.'.")
            return

        if not xs or not ys:
            self.status.setText("X and Y must not be empty.")
            return

        if len(xs) != len(ys):
            self.status.setText(f"Length mismatch: X has {len(xs)} values, Y has {len(ys)}.")
            return

        # Build Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name="Data"
        ))
        fig.update_layout(
            title="Pasted Data",
            xaxis_title="X",
            yaxis_title="Y",
            template="plotly_white",
            margin=dict(l=40, r=20, t=50, b=40),
            height=520,
        )

        html = pio.to_html(fig, full_html=True, include_plotlyjs="cdn")
        self.view.setHtml(html)

        self.status.setStyleSheet("color:#0a0;")
        self.status.setText(f"Plotted {len(xs)} points.")

def main():
    app = QApplication(sys.argv)
    w = PlotPasteApp()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
