# File: app/plotting/plotter.py

import os
import tempfile
import traceback
import plotly.graph_objects as go
import plotly.io as pio
from PyQt5 import QtCore
import pandas as pd
from endaq.calc.fft import rolling_fft
from endaq.plot import rolling_min_max_envelope, spectrum_over_time


class Plotter:
    """Handles all logic for creating Plotly figures."""

    def __init__(self):
        # Default settings, can be updated from the SettingsTab
        self.legend_font_size = 10
        self.default_font_size = 12
        self.hover_font_size = 15
        self.hover_mode = 'closest'
        self.legend_visible = True
        self.current_legend_position_index = 1 # 'top left'
        self.legend_positions = ['default', 'top left', 'top right', 'bottom right', 'bottom left']

    def _get_legend_position(self):
        """Gets the dictionary for the current legend position setting."""
        positions = {
            'default': {'x': 1.02, 'y': 1, 'xanchor': 'left', 'yanchor': 'top'},
            'top left': {'x': 0.01, 'y': 0.99, 'xanchor': 'left', 'yanchor': 'top'},
            'top right': {'x': 0.99, 'y': 0.99, 'xanchor': 'right', 'yanchor': 'top'},
            'bottom right': {'x': 0.99, 'y': 0.01, 'xanchor': 'right', 'yanchor': 'bottom'},
            'bottom left': {'x': 0.01, 'y': 0.01, 'xanchor': 'left', 'yanchor': 'bottom'}
        }
        position_name = self.legend_positions[self.current_legend_position_index]
        return positions.get(position_name, positions['default'])

    def _get_hover_template(self, index_name):
        """Generates the custom hovertemplate string based on the data domain."""
        is_freq_domain = 'freq' in index_name.lower()
        domain_label = 'Hz' if is_freq_domain else 'Time'
        return f"%{{fullData.name}}<br>{domain_label}: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>"

    def create_standard_figure(self, data_to_plot, title, y_axis_title="Value"):
        """
        This function intelligently handles two types of input:
        1. A single DataFrame with multiple columns (for plotting T1, T2, etc. on one graph)
        2. A dictionary of single-column DataFrames (for plotting multiple folders on one graph)
        """
        fig = go.Figure()

        if isinstance(data_to_plot, pd.DataFrame):
            # --- Case 1: Input is a DataFrame ---
            if data_to_plot.empty:
                return go.Figure()

            hover_template = self._get_hover_template(data_to_plot.index.name)
            for column_name in data_to_plot.columns:
                fig.add_trace(go.Scatter(
                    x=data_to_plot.index,
                    y=data_to_plot[column_name],
                    mode='lines',
                    name=column_name,  # The trace name is the column name
                    hovertemplate=hover_template
                ))
            x_axis_title = data_to_plot.index.name

        elif isinstance(data_to_plot, dict):
            # --- Case 2: Input is a dictionary of DataFrames ---
            df_dict = data_to_plot
            if not df_dict:
                return go.Figure()

            for trace_name, df in df_dict.items():
                if df is None or df.empty:
                    continue

                hover_template = self._get_hover_template(df.index.name)
                # This logic assumes each DataFrame in the dict has only one data column
                col_name = df.columns[0]

                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[col_name],
                    mode='lines',
                    name=trace_name, # The trace name is the dictionary key
                    hovertemplate=hover_template,
                    line=dict(dash='solid')
                ))
            # Assume all DataFrames have the same index name (x-axis label)
            x_axis_title = list(df_dict.values())[0].index.name
        else:
            # If input is neither, return an empty figure
            return go.Figure()

        self._apply_standard_layout(fig, title, x_axis_title, y_axis_title)
        return fig

    def create_spectrum_figure(self, df, num_slices, plot_type, freq_max=None, colorscale='Hot'):
        """
        Creates a spectrum plot (e.g., heatmap, waterfall) from time-series data.
        """
        if df is None or df.empty:
            return go.Figure()

        data_column_name = df.columns[0]
        fft_df = rolling_fft(df, num_slices=num_slices, add_resultant=True)

        # --- MODIFIED: Create the figure first ---
        fig = spectrum_over_time(
            fft_df,
            plot_type=plot_type,
            freq_max=freq_max,
            var_to_process=data_column_name
        )

        # Check if plot type is one that uses a colorscale
        if plot_type in ['Heatmap', 'Surface']:
            # Loop through all traces to find the spectrum plot type
            for trace in fig.data:
                # Check if the trace is a Heatmap or a Surface and apply the colorscale
                if isinstance(trace, (go.Heatmap, go.Surface)):
                    trace.colorscale = colorscale
                    break  # Stop after finding and updating the first one

        self._apply_standard_layout(fig, f"Spectrum Plot ({plot_type})", "Frequency (Hz)", "Time (s)")
        return fig

        # Apply the standard layout for a consistent look and feel
        self._apply_standard_layout(fig, f"Spectrum Plot ({plot_type})", "Frequency (Hz)", "Time (s)")
        return fig

    def create_comparison_figure(self, df1, df2, column, title):
        fig = go.Figure()
        x_label = df1.index.name
        hover_template = self._get_hover_template(x_label)

        fig.add_trace(go.Scatter(
            x=df1.index,
            y=df1[column],
            name=f"Original - {column}",
            hovertemplate=hover_template
        ))

        fig.add_trace(go.Scatter(
            x=df2.index,
            y=df2[column],
            name=f"Compare - {column}",
            hovertemplate=hover_template
        ))

        self._apply_standard_layout(fig, title, x_label, "Value")
        return fig

    def create_difference_figure(self, diff_df, title, y_title):
        fig = go.Figure()
        hover_template = self._get_hover_template(diff_df.index.name)

        for col in diff_df.columns:
            fig.add_trace(go.Scatter(
                x=diff_df.index,
                y=diff_df[col],
                name=col,
                hovertemplate=hover_template
            ))

        self._apply_standard_layout(fig, title, diff_df.index.name, y_title)
        return fig

    def create_rolling_envelope_figure(self, df_dict, title, desired_num_points, plot_as_bars):
        if not df_dict:
            return go.Figure()

        # Combine all dataframes for the endaq function
        # The keys (trace names) will become the column names
        combined_df = pd.concat([df.rename(columns={df.columns[0]: name}) for name, df in df_dict.items()], axis=1)

        if combined_df.empty:
            return go.Figure()

        fig = rolling_min_max_envelope(
            combined_df,
            opacity=0.6,
            desired_num_points=desired_num_points,
            plot_as_bars=plot_as_bars
        )
        x_axis_title = list(df_dict.values())[0].index.name
        self._apply_standard_layout(fig, title, x_axis_title, "Value")
        return fig

    def _apply_standard_layout(self, fig, title, x_axis_title, y_axis_title):
        """Applies a consistent style to all figures."""
        legend_pos = self._get_legend_position()
        fig.update_layout(
            title=title,
            margin=dict(l=40, r=20, t=50, b=40),
            legend=dict(
                font=dict(family='Open Sans', size=self.legend_font_size, color='black'),
                x=legend_pos['x'], y=legend_pos['y'],
                xanchor=legend_pos.get('xanchor', 'auto'),
                yanchor=legend_pos.get('yanchor', 'auto'),
                bgcolor='rgba(255, 255, 255, 0.6)'
            ),
            hoverlabel=dict(bgcolor='rgba(240, 240, 240, 0.9)', font_size=self.hover_font_size),
            hovermode=self.hover_mode,
            font=dict(family='Open Sans', size=self.default_font_size, color='black'),
            showlegend=self.legend_visible,
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title
        )

# Helper function (used by tab classes)
def load_fig_to_webview(fig, web_view):
    """Generates full HTML, saves to a temp file, and loads it into a QWebEngineView."""
    try:
        html_content = pio.to_html(fig, full_html=True, include_plotlyjs=True, config={'responsive': True})
        # Using a NamedTemporaryFile can be tricky with URLs. Let's manage it manually.
        # We'll store temp files on the web_view object itself to manage their lifecycle.
        if not hasattr(web_view, '_temp_files'):
            web_view._temp_files = []

        # Clean up old temp files for this specific widget
        for old_file in web_view._temp_files:
            try:
                os.remove(old_file)
            except OSError:
                pass
        web_view._temp_files.clear()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(html_content)
            file_path = tmp_file.name
            web_view._temp_files.append(file_path)

        web_view.setUrl(QtCore.QUrl.fromLocalFile(file_path))
        web_view.show()

    except Exception as e:
        print(f"Error loading figure to webview: {e}")
        tb = traceback.format_exc()
        error_html = f"<html><body><h1>Error loading plot</h1><pre>{e}</pre><pre>{tb}</pre></body></html>"
        web_view.setHtml(error_html)
