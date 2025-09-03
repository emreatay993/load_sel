# File: app/controllers/plot_controller.py

import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PyQt5 import QtCore

from ..analysis.data_processing import apply_data_section, apply_tukey_window, apply_low_pass_filter


class PlotController(QtCore.QObject):
    """
    Handles all logic for updating plots in response to UI changes.
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.plotter = self.main_window.plotter

    def _get_df(self):
        return self.main_window.df

    def _get_compare_df(self):
        return self.main_window.df_compare

    def _get_data_domain(self):
        return self.main_window.data_domain

    def _get_plot_df(self, cols, source_df=None):
        """Prepares a DataFrame for plotting with the correct index."""
        df = self._get_df()
        if df is None:
            return pd.DataFrame()
            
        source_df = source_df if source_df is not None else df
        data_domain = self._get_data_domain()

        if not all(col in source_df.columns for col in [data_domain] + cols):
            return pd.DataFrame()

        x_label = 'Time [s]' if data_domain == 'TIME' else 'Freq [Hz]'
        x_data = source_df[data_domain]
        plot_df = source_df[cols].copy()
        plot_df.index = x_data
        plot_df.index.name = x_label
        return plot_df

    def _should_exclude_component(self, col_name: str) -> bool:
        """
        Checks if a column should be excluded based on the T2/T3/R2/R3 filter,
        while correctly preserving resultants like 'T2/T3'.
        """
        # Match T2 but not T2/T3
        if re.search(r'\bT2\b', col_name) and not re.search(r'T2/T3', col_name):
            return True
        # Match T3 but not T2/T3
        if re.search(r'\bT3\b', col_name) and not re.search(r'T2/T3', col_name):
            return True
        # Match R2 but not R2/R3
        if re.search(r'\bR2\b', col_name) and not re.search(r'R2/R3', col_name):
            return True
        # Match R3 but not R2/R3
        if re.search(r'\bR3\b', col_name) and not re.search(r'R2/R3', col_name):
            return True
        return False

    def _filter_part_load_cols(self, all_columns, side, required_components, exclude):
        """
        A shared helper to filter DataFrame columns for part loads based on UI selections.
        """
        side_pattern = re.compile(rf'\b{re.escape(side)}\b')

        # Step 1: Find columns relevant to the selected side using a regex pattern.
        side_cols = [col for col in all_columns if side_pattern.search(col)]

        # Step 2: From those, find columns for the required components (e.g., 'T1', 'T2').
        component_cols = [col for col in side_cols if any(comp in col for comp in required_components)]

        # Step 3: Exclude any phase angle columns.
        final_cols = [col for col in component_cols if 'Phase_' not in col]

        # Step 4: If the exclude box is checked, remove T2, T3, R2, R3 (but keep resultants).
        if exclude:
            final_cols = [col for col in final_cols if not self._should_exclude_component(col)]

        return final_cols
    
    def _calculate_differences(self, columns):
        """Calculates the absolute difference between two dataframes for given columns."""
        df = self._get_df()
        df_compare = self._get_compare_df()
        data_domain = self._get_data_domain()
        
        if df is None or df_compare is None:
            return pd.DataFrame()

        diff_dict = {}
        for col in columns:
            if col not in df.columns or col not in df_compare.columns:
                continue

            mag1, mag2 = df[col], df_compare[col]

            if data_domain == 'FREQ':
                phase_col = f'Phase_{col}'
                if phase_col in df.columns and phase_col in df_compare.columns:
                    p1_rad = np.deg2rad(df[phase_col])
                    p2_rad = np.deg2rad(df_compare[phase_col])
                    diff = np.abs((mag1 * np.exp(1j * p1_rad)) - (mag2 * np.exp(1j * p2_rad)))
                else:
                    diff = np.abs(mag1 - mag2)
            else:
                diff = np.abs(mag1 - mag2)
            
            diff_dict[f'Δ {col}'] = diff
        return pd.DataFrame(diff_dict)

    # region Signal Slots
    @QtCore.pyqtSlot()
    def update_all_plots_from_settings(self):
        if self._get_df() is None: return

        settings_tab = self.main_window.tab_settings
        self.plotter.legend_font_size = int(settings_tab.legend_font_size_selector.currentText())
        self.plotter.default_font_size = int(settings_tab.default_font_size_selector.currentText())
        self.plotter.hover_font_size = int(settings_tab.hover_font_size_selector.currentText())
        self.plotter.hover_mode = settings_tab.hover_mode_selector.currentText()

        self.update_single_data_plots()
        self.update_interface_data_plots()
        self.update_part_loads_plots()
        self.update_time_domain_represent_plot()
        self.update_compare_data_plots()
        self.update_compare_part_loads_plots()

    @QtCore.pyqtSlot()
    def update_single_data_plots(self):
        df = self._get_df()
        if df is None: return
        tab = self.main_window.tab_single_data
        selected_col = tab.column_selector.currentText()
        if not selected_col: return

        is_multi_folder = df['DataFolder'].nunique() > 1
        dfs_for_plot = {}

        for folder_name, group_df in df.groupby('DataFolder'):
            plot_df_group = self._get_plot_df([selected_col], source_df=group_df)
            if self._get_data_domain() == 'TIME' and tab.filter_checkbox.isChecked():
                try:
                    cutoff = float(tab.cutoff_frequency_input.text())
                    order = tab.filter_order_input.value()
                    plot_df_group = apply_low_pass_filter(plot_df_group, selected_col, cutoff, order)
                except ValueError:
                    pass # Ignore if cutoff is not a valid number
            dfs_for_plot[folder_name if is_multi_folder else selected_col] = plot_df_group

        plot_title = f"{selected_col} Plot"
        if self.main_window.tab_settings.rolling_min_max_checkbox.isChecked() and self._get_data_domain() == 'TIME':
            try:
                points = int(self.main_window.tab_settings.desired_num_points_input.text())
                as_bars = self.main_window.tab_settings.plot_as_bars_checkbox.isChecked()
                fig = self.plotter.create_rolling_envelope_figure(dfs_for_plot, plot_title, points, as_bars)
            except ValueError:
                fig = self.plotter.create_standard_figure(dfs_for_plot, title=f"{plot_title} (Invalid Points)")
        else:
            fig = self.plotter.create_standard_figure(dfs_for_plot, title=plot_title)
        tab.display_regular_plot(fig)

        if self._get_data_domain() == 'FREQ' and not is_multi_folder:
            phase_col = f'Phase_{selected_col}'
            if phase_col in df.columns:
                phase_df = self._get_plot_df([phase_col])
                phase_fig = self.plotter.create_standard_figure({phase_col: phase_df}, f'Phase of {selected_col}', 'Phase [deg]')
                tab.set_phase_plot_visibility(True)
                tab.display_phase_plot(phase_fig)
            else:
                tab.set_phase_plot_visibility(False)
        else:
            tab.set_phase_plot_visibility(False)

        if self._get_data_domain() == 'TIME' and tab.spectrum_checkbox.isChecked() and not is_multi_folder:
            self.update_spectrum_plot_only()

    @QtCore.pyqtSlot()
    def update_interface_data_plots(self):
        df = self._get_df()
        if df is None: return
        tab = self.main_window.tab_interface_data
        interface = tab.interface_selector.currentText()
        side = tab.side_selector.currentText()
        if not interface or not side: return

        t_cols = [c for c in df.columns if c.startswith(interface) and side in c and any(s in c for s in ['T1', 'T2', 'T3', 'T2/T3']) and 'Phase_' not in c]
        r_cols = [c for c in df.columns if c.startswith(interface) and side in c and any(s in c for s in ['R1', 'R2', 'R3', 'R2/R3']) and 'Phase_' not in c]

        tab.display_t_series_plot(self.plotter.create_standard_figure(self._get_plot_df(t_cols), f'Translational Components - {side}'))
        tab.display_r_series_plot(self.plotter.create_standard_figure(self._get_plot_df(r_cols), f'Rotational Components - {side}'))

    @QtCore.pyqtSlot()
    def update_part_loads_plots(self):
        df = self._get_df()
        if df is None: return
        tab = self.main_window.tab_part_loads
        side = tab.side_filter_selector.currentText()
        if not side: return

        exclude = tab.exclude_checkbox.isChecked()
        df_processed = df.copy()

        # Call helper functions for data processing
        if self._get_data_domain() == 'TIME':
            if tab.section_checkbox.isChecked():
                df_processed = apply_data_section(df_processed, tab.section_min_input.text(),
                                                  tab.section_max_input.text())

            if tab.tukey_checkbox.isChecked():
                df_processed = apply_tukey_window(df_processed, tab.tukey_alpha_spin.value())

        # Filter columns from the (potentially) processed DataFrame
        t_cols = self._filter_part_load_cols(df_processed.columns, side, ['T1', 'T2', 'T3', 'T2/T3'], exclude)
        r_cols = self._filter_part_load_cols(df_processed.columns, side, ['R1', 'R2', 'R3', 'R2/R3'], exclude)

        # Use the processed DataFrame as the source for the plots
        tab.display_t_series_plot(self.plotter.create_standard_figure(self._get_plot_df(t_cols, source_df=df_processed),
                                                                      f'Translational Components - {side}'))
        tab.display_r_series_plot(self.plotter.create_standard_figure(self._get_plot_df(r_cols, source_df=df_processed),
                                                                      f'Rotational Components- {side}'))
    
    @QtCore.pyqtSlot()
    def update_time_domain_represent_plot(self):
        df = self._get_df()
        if df is None or self._get_data_domain() != 'FREQ': return
        
        tab = self.main_window.tab_time_domain_represent
        try:
            freq_text = tab.data_point_selector.currentText()
            if not freq_text or "Select a frequency" in freq_text: return
            freq = float(freq_text)

            selected_side = self.main_window.tab_part_loads.side_filter_selector.currentText()
            if not selected_side:
                tab.display_plot(go.Figure())
                return

            side_pattern = re.compile(rf'\b{re.escape(selected_side)}\b')
            plot_cols = [c for c in df.columns if side_pattern.search(c) and not c.startswith('Phase_') and any(s in c for s in ['T1', 'T2', 'T3', 'R1', 'R2', 'R3', 'T2/T3', 'R2/R3'])]

            theta = np.linspace(0, 360, 361)
            rads = np.radians(theta)
            plot_data = {}
            tab.current_plot_data = {}
            data_at_freq = df[df['FREQ'] == freq].iloc[0]

            for col in plot_cols:
                phase_col = f'Phase_{col}'
                if phase_col in data_at_freq:
                    amplitude = data_at_freq[col]
                    phase_deg = data_at_freq[phase_col]
                    y_data = amplitude * np.cos(rads - np.radians(phase_deg))
                    plot_data[col] = y_data
                    tab.current_plot_data[col] = {'theta': theta, 'y_data': y_data}
            
            df_time_domain = pd.DataFrame(plot_data, index=theta)
            df_time_domain.index.name = "Theta [deg]"
            
            title = f'Time Domain Representation at {freq} Hz for {selected_side}'
            fig = self.plotter.create_standard_figure(df_time_domain, title)
            tab.display_plot(fig)

        except (ValueError, IndexError) as e:
            print(f"Could not update time domain representation plot: {e}")
            tab.display_plot(go.Figure())
            
    @QtCore.pyqtSlot()
    def update_compare_data_plots(self):
        if self._get_df() is None or self._get_compare_df() is None: return
        tab = self.main_window.tab_compare_data
        selected_column = tab.compare_column_selector.currentText()
        if not selected_column: return

        df1 = self._get_plot_df([selected_column])
        df2 = self._get_plot_df([selected_column], source_df=self._get_compare_df())
        
        fig_compare = self.plotter.create_comparison_figure(df1, df2, selected_column, f'{selected_column} Comparison')
        tab.display_comparison_plot(fig_compare)
        
        diff_df = self._calculate_differences([selected_column])
        if diff_df.empty: return

        abs_diff_df = self._get_plot_df([], source_df=pd.DataFrame({'Absolute Difference': diff_df.iloc[:, 0]}))
        fig_abs_diff = self.plotter.create_standard_figure(abs_diff_df, f'{selected_column} Absolute Difference')
        tab.display_absolute_diff_plot(fig_abs_diff)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_diff = np.divide(100 * diff_df.iloc[:, 0], np.abs(self._get_df()[selected_column]))
            relative_diff.fillna(0, inplace=True)
        rel_diff_df = self._get_plot_df([], source_df=pd.DataFrame({'Relative Difference (%)': relative_diff}))
        fig_rel_diff = self.plotter.create_standard_figure(rel_diff_df, f'{selected_column} Relative Difference (%)', "Percent (%)")
        tab.display_relative_diff_plot(fig_rel_diff)

    @QtCore.pyqtSlot()
    def update_compare_part_loads_plots(self):
        if self._get_df() is None or self._get_compare_df() is None: return
        tab = self.main_window.tab_compare_part_loads
        selected_side = tab.side_filter_selector.currentText()
        if not selected_side: return

        exclude = tab.exclude_checkbox.isChecked()

        t_cols = self._filter_part_load_cols(self._get_df().columns, selected_side, ["T1", "T2", "T3", "T2/T3"], exclude)
        r_cols = self._filter_part_load_cols(self._get_df().columns, selected_side, ["R1", "R2", "R3", "R2/R3"], exclude)

        t_diff_df = self._get_plot_df([], source_df=self._calculate_differences(t_cols))
        r_diff_df = self._get_plot_df([], source_df=self._calculate_differences(r_cols))
        
        fig_t = self.plotter.create_standard_figure(t_diff_df, f'Translational Components, Difference (Δ) - {selected_side}')
        tab.display_t_series_plot(fig_t)
        
        fig_r = self.plotter.create_standard_figure(r_diff_df, f'Rotational Components, Difference (Δ) - {selected_side}')
        tab.display_r_series_plot(fig_r)

    @QtCore.pyqtSlot()
    def update_spectrum_plot_only(self):
        """A dedicated function that only updates the spectrum plot."""
        df = self._get_df()
        if df is None or self._get_data_domain() != 'TIME': return

        tab = self.main_window.tab_single_data
        selected_col = tab.column_selector.currentText()
        if not selected_col or not tab.spectrum_checkbox.isChecked(): return

        is_multi_folder = df['DataFolder'].nunique() > 1
        if is_multi_folder: return

        try:
            # Re-create the source DataFrame for the spectrum plot
            plot_df = self._get_plot_df([selected_col])
            if tab.filter_checkbox.isChecked():
                try:
                    cutoff = float(tab.cutoff_frequency_input.text())
                    order = tab.filter_order_input.value()
                    plot_df = apply_low_pass_filter(plot_df, selected_col, cutoff, order)
                except ValueError:
                    pass # Ignore if cutoff is not a valid number

            # Generate and display the spectrum plot
            fig_spec = self.plotter.create_spectrum_figure(
                plot_df,
                num_slices=int(tab.num_slices_input.text()),
                plot_type=tab.plot_type_selector.currentText(),
                colorscale=tab.colorscale_selector.currentText()
            )
            tab.set_spectrum_plot_visibility(True)
            tab.display_spectrum_plot(fig_spec)
        except (ValueError, IndexError, ZeroDivisionError) as e:
            print(f"Could not generate spectrum: {e}")
            tab.set_spectrum_plot_visibility(False)
    # endregion