# File: app/data_manager.py

import os
import sys
import pandas as pd
import re
from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox

class DataManager(QtCore.QObject):
    """
    Handles all data loading, parsing, and management.
    Emits a signal when data is successfully loaded.
    """
    # Signal carries: a pandas DataFrame, a string for the domain ('TIME'/'FREQ'),
    # and a string for the folder path.
    dataLoaded = QtCore.pyqtSignal(pd.DataFrame, str, str)
    dataLoadFailed = QtCore.pyqtSignal(str)
    comparisonDataLoaded = QtCore.pyqtSignal(pd.DataFrame)

    def __init__(self, parent=None):
        super().__init__(parent)

    def load_data_from_directory(self):
        folder = self._select_directory('Please select a directory for raw data and headers')
        if not folder:
            sys.exit(1)

        # This initial load is now just a special case of loading from a list of paths
        self.load_data_from_paths([folder])

    ## The core method for handling single or multiple folder selections
    def load_data_from_paths(self, folder_paths):
        """
        Loads data from a list of folder paths, validates them, combines them,
        and emits the result.
        """
        combined_dfs = []
        data_domain = None  # Determined by the first valid folder
        first_valid_folder = None

        for folder in folder_paths:
            try:
                # 1. Validate folder contents
                full_pld_files = self._get_file_path(folder, 'full.pld')
                max_pld_files = self._get_file_path(folder, 'max.pld')

                if not full_pld_files or not max_pld_files:
                    QMessageBox.warning(None, "Invalid Folder",
                                        f"Folder '{os.path.basename(folder)}' is missing required .pld files. Skipping.")
                    continue

                # 2. Read and determine domain for this folder
                dfs = [self._read_pld_file(path) for path in full_pld_files]
                df_temp = pd.concat(dfs, ignore_index=True)

                if 'FREQ' in df_temp.columns:
                    current_domain = 'FREQ'
                elif 'TIME' in df_temp.columns:
                    current_domain = 'TIME'
                else:
                    QMessageBox.warning(None, "Invalid Data",
                                        f"Data in '{os.path.basename(folder)}' has no TIME or FREQ column. Skipping.")
                    continue

                # 3. Validate Domain Consistency
                if data_domain is None:  # First valid folder sets the domain
                    data_domain = current_domain
                    first_valid_folder = folder
                elif current_domain != data_domain:
                    QMessageBox.warning(None, "Domain Mismatch",
                                        f"Folder '{os.path.basename(folder)}' has domain '{current_domain}' but expected '{data_domain}'. Skipping.")
                    continue

                # 4. Process headers and assign columns
                df_intf_before = self._read_pld_log_file(max_pld_files[0])
                new_columns = self._get_column_headers(df_intf_before, current_domain)

                additional_cols = len(df_temp.columns) - len(new_columns)
                if additional_cols > 0:
                    new_columns.extend([f"Extra_Column_{i}" for i in range(1, additional_cols + 1)])
                df_temp.columns = new_columns[:len(df_temp.columns)]

                # 5. CRUCIAL: Add the DataFolder column for grouping later
                df_temp['DataFolder'] = os.path.basename(folder)
                combined_dfs.append(df_temp)

            except Exception as e:
                QMessageBox.critical(None, "Load Error", f"Failed to load data from '{os.path.basename(folder)}':\n{e}")
                continue  # Skip to the next folder

        if not combined_dfs:
            self.dataLoadFailed.emit("No valid data could be loaded from the selected folder(s).")
            return

        # Concatenate all valid DataFrames
        final_df = pd.concat(combined_dfs, ignore_index=True)

        # Sort the final DataFrame by the domain column
        final_df = final_df.sort_values(by=data_domain).reset_index(drop=True)

        # Save the combined data to a CSV
        final_df.to_csv("full_data.csv", index=False)
        print(f"Data from {len(combined_dfs)} folder(s) processed and saved to full_data.csv")

        # Emit the signal with the combined results
        self.dataLoaded.emit(final_df, data_domain, first_valid_folder)

    # Helper to avoid code duplication
    def _get_column_headers(self, df_intf_before, data_domain):
        """Determines the correct column headers based on the data domain."""
        if data_domain == 'FREQ':
            df_intf = self._insert_phase_columns(df_intf_before)
            return ['NO', 'FREQ'] + df_intf.iloc[0].tolist()
        elif data_domain == 'TIME':
            return ['NO', 'TIME'] + df_intf_before.iloc[0].tolist()
        return []

    def _select_directory(self, title):
        # Now a private helper method
        folder = QFileDialog.getExistingDirectory(None, title)
        return folder

    def _get_file_path(self, folder, file_suffix):
        # Now a private helper method
        return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(file_suffix)]

    def _read_pld_log_file(self, file_path):
        # Now a private helper method
        df = pd.read_csv(file_path, delimiter='|', skipinitialspace=True, skip_blank_lines=True)
        df = df.iloc[:, 1].dropna().str.strip().to_frame()
        return df.T

    def _insert_phase_columns(self, df):
        # Now a private helper method
        interface_labels = df.iloc[0, :].copy()
        transformed_labels = []
        for label in interface_labels:
            transformed_labels.append(label)
            phase_label = f"Phase_{label}"
            transformed_labels.append(phase_label)
        df = pd.DataFrame([transformed_labels], index=["Interface Label"])
        return df

    def _read_pld_file(self, file_path):
        # Now a private helper method
        df = pd.read_csv(file_path, delimiter='|', skipinitialspace=True, skip_blank_lines=True, comment='_', low_memory=False)
        df = df.apply(pd.to_numeric)
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        df.columns = df.columns.str.strip()
        df.reset_index(drop=True, inplace=True)
        return df

    def load_comparison_data(self):
        """Loads a secondary dataset for comparison purposes."""
        folder = self._select_directory('Please select a directory for COMPARISON data')
        if not folder:
            return

        try:
            # This logic is identical to the initial load, but simplified
            file_path_full_data = self._get_file_path(folder, 'full.pld')
            file_path_headers_data = self._get_file_path(folder, 'max.pld')

            if not file_path_full_data or not file_path_headers_data:
                QMessageBox.critical(None, 'Error', "No required files found in comparison folder.")
                return

            dfs = [self._read_pld_file(path) for path in file_path_full_data]
            df_compare = pd.concat(dfs, ignore_index=True)

            df_intf_before = self._read_pld_log_file(file_path_headers_data[0])

            if 'FREQ' in df_compare.columns:
                domain = 'FREQ'
            elif 'TIME' in df_compare.columns:
                domain = 'TIME'
            else:
                raise ValueError("Comparison data has no FREQ or TIME column.")

            new_columns = self._get_column_headers(df_intf_before, domain)
            additional_cols = len(df_compare.columns) - len(new_columns)
            if additional_cols > 0:
                new_columns.extend([f"Extra_Col_{i + 1}" for i in range(additional_cols)])
            df_compare.columns = new_columns[:len(df_compare.columns)]
            self.comparisonDataLoaded.emit(df_compare)

        except Exception as e:
            QMessageBox.critical(None, 'Error', f"An error occurred loading comparison data: {str(e)}")