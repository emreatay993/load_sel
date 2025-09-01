# File: app/analysis/ansys_exporter.py

import os
import re
import shutil
import traceback
from collections import OrderedDict
import pandas as pd
import numpy as np
import math
from PyQt5.QtWidgets import QMessageBox


class AnsysExporter:
    """
    Handles all logic for creating and exporting Ansys Mechanical templates.
    This class contains all dependencies on the ansys-mechanical-core library.
    """

    def __init__(self):
        self.app_ansys = None
        self.Model = None
        self.ExtAPI = None
        self.DataModel = None
        self.Quantity = None
        self.Ansys = None

    def _init_ansys_session(self):
        """Initializes a new Ansys Mechanical session."""
        try:
            print("Importing ansys-mechanical-core library...")
            import ansys.mechanical.core as mech
            from ansys.mechanical.core import global_variables
            print("Imported.")

            print("Starting Ansys Mechanical...")
            self.app_ansys = mech.App()
            print("Ansys Mechanical session started.")

            globals_map = global_variables(self.app_ansys)
            self.Model = globals_map["Model"]
            self.ExtAPI = globals_map["ExtAPI"]
            self.DataModel = globals_map["DataModel"]
            self.Quantity = globals_map["Quantity"]
            self.Ansys = globals_map["Ansys"]

            self.ExtAPI.Application.ActiveUnitSystem = self.Ansys.ACT.Interfaces.Common.MechanicalUnitSystem.StandardMKS
            return True
        except Exception as e:
            QMessageBox.critical(None, 'Ansys Error',
                                 f"Failed to initialize Ansys Mechanical.\nEnsure it is installed and configured correctly.\n\nError: {e}")
            return False

    def _close_ansys_session(self):
        """Closes the Ansys Mechanical session."""
        if self.app_ansys:
            try:
                self.app_ansys.close()
                print("Ansys Mechanical session closed.")
            except Exception as e:
                print(f"Error closing Ansys session: {e}")

    def create_harmonic_template(self, df_export, data_domain):
        if not self._init_ansys_session():
            return
        try:
            all_keys = ["T1", "T2", "T3", "R1", "R2", "R3", "Phase_T1", "Phase_T2", "Phase_T3", "Phase_R1", "Phase_R2",
                        "Phase_R3"]

            def get_full_interface_name(column_name):
                column_name = re.sub(r'\s+[TR][1-3]$', '', column_name)
                return re.sub(r'^Phase_', '', column_name)

            full_interfaces = OrderedDict()
            for col in df_export.columns:
                if col != "FREQ":
                    interface_name = get_full_interface_name(col)
                    if interface_name not in full_interfaces:
                        full_interfaces[interface_name] = []
                    full_interfaces[interface_name].append(col)

            def create_full_interface_dict(df, columns):
                interface_dict = {key: [0] * len(df) for key in all_keys}
                for col in columns:
                    key = col.split()[-1]
                    if col.startswith("Phase_"):
                        interface_dict["Phase_" + key] = df[col].tolist()
                    else:
                        interface_dict[key] = df[col].tolist()
                return interface_dict

            interface_dicts_full = {interface: create_full_interface_dict(df_export, cols) for interface, cols in
                                    full_interfaces.items()}
            list_of_part_interface_names = list(interface_dicts_full.keys())

            scale_factor = 1000
            list_of_all_frequencies = df_export["FREQ"].tolist()
            list_of_all_frequencies_as_quantity = [self.Quantity(val, "Hz") for val in list_of_all_frequencies]

            self.Model.AddGeometryImportGroup().AddGeometryImport()

            analysis_static = self.Model.AddStaticStructuralAnalysis()
            analysis_settings_static = analysis_static.AnalysisSettings
            analysis_settings_static.PropertyByName("SolverUnitsControl").InternalValue = 1

            analysis_modal = self.Model.AddModalAnalysis()
            analysis_settings_modal = analysis_modal.AnalysisSettings
            self.DataModel.GetObjectsByName("Pre-Stress (None)")[0].PreStressICEnvironment = analysis_static
            analysis_settings_modal.PropertyByName("NumModesToFind").InternalValue = 100
            analysis_settings_modal.PropertyByName("RangeSearch").InternalValue = 1
            analysis_settings_modal.PropertyByName("MaxFrequency").InternalValue = df_export['FREQ'].max() * 1.5
            analysis_settings_modal.PropertyByName("MSUPSkipExpansion").InternalValue = 1
            analysis_settings_modal.PropertyByName("SolverUnitsControl").InternalValue = 1

            analysis_HR = self.Model.AddHarmonicResponseAnalysis()
            analysis_settings_HR = analysis_HR.AnalysisSettings
            analysis_settings_HR.PropertyByName("HarmonicForcingFrequencyMax").InternalValue = df_export['FREQ'].max()
            self.DataModel.GetObjectsByName("Pre-Stress/Modal (None)")[0].PreStressICEnvironment = analysis_modal
            analysis_settings_HR.PropertyByName("MSUPSkipExpansion").InternalValue = 1
            analysis_settings_HR.PropertyByName("CombineDistResultFile").InternalValue = 1
            analysis_settings_HR.PropertyByName("ExpandResultFrom").InternalValue = 1
            analysis_settings_HR.PropertyByName("HarmonicForcingFrequencyIntervals").InternalValue = 1
            analysis_settings_HR.PropertyByName("HarmonicSolutionMethod").InternalValue = 1
            analysis_settings_HR.PropertyByName("ConstantDampingValue").InternalValue = 0.02

            interface_index_no = 1
            for interface_name in list_of_part_interface_names:
                CS_interface = self.Model.CoordinateSystems.AddCoordinateSystem()
                CS_interface.Name = "CS_" + interface_name

                RP_interface = self.Model.AddRemotePoint()
                RP_interface.Name = "RP_" + interface_name
                RP_interface.CoordinateSystem = CS_interface
                RP_interface.PilotNodeAPDLName = "RP_" + str(interface_index_no)

                force_HR = analysis_HR.AddRemoteForce()
                force_HR.DefineBy = self.Ansys.Mechanical.DataModel.Enums.LoadDefineBy.Components
                force_HR.Name = "RF_" + interface_name
                force_HR.PropertyByName("GeometryDefineBy").InternalValue = 2
                force_HR.Location = RP_interface
                force_HR.Suppressed = False
                force_HR_index_name = "RF_" + str(interface_index_no)

                moment_HR = analysis_HR.AddMoment()
                moment_HR.DefineBy = self.Ansys.Mechanical.DataModel.Enums.LoadDefineBy.Components
                moment_HR.Name = "RM_" + interface_name + "_(For Visualization Purposes Only, Delete it or Keep it Suppressed)"
                moment_HR.PropertyByName("GeometryDefineBy").InternalValue = 2
                moment_HR.Location = RP_interface
                moment_HR.Suppressed = True
                moment_HR_index_name = "RM_" + str(interface_index_no)

                list_of_fx_values = [self.Quantity(v, "kN") for v in interface_dicts_full[interface_name]["T1"]]
                list_of_fy_values = [self.Quantity(v, "kN") for v in interface_dicts_full[interface_name]["T2"]]
                list_of_fz_values = [self.Quantity(v, "kN") for v in interface_dicts_full[interface_name]["T3"]]
                list_of_angle_fx_values = [self.Quantity(v, "deg") for v in
                                           interface_dicts_full[interface_name]["Phase_T1"]]
                list_of_angle_fy_values = [self.Quantity(v, "deg") for v in
                                           interface_dicts_full[interface_name]["Phase_T2"]]
                list_of_angle_fz_values = [self.Quantity(v, "deg") for v in
                                           interface_dicts_full[interface_name]["Phase_T3"]]

                df_load_table_fx = pd.DataFrame(
                    {'FREQ': list_of_all_frequencies, 'T1': interface_dicts_full[interface_name]["T1"]}).set_index(
                    'FREQ')
                df_load_table_phase_fx = pd.DataFrame({'FREQ': list_of_all_frequencies,
                                                       'Phase_T1': interface_dicts_full[interface_name][
                                                           "Phase_T1"]}).set_index('FREQ')
                df_load_table_fx_real = pd.DataFrame(
                    df_load_table_fx['T1'] * np.cos(df_load_table_phase_fx['Phase_T1']), columns=['Real_T1'])
                df_load_table_fx_imag = pd.DataFrame(
                    df_load_table_fx['T1'] * np.sin(df_load_table_phase_fx['Phase_T1']), columns=['Imag_T1'])

                df_load_table_fy = pd.DataFrame(
                    {'FREQ': list_of_all_frequencies, 'T2': interface_dicts_full[interface_name]["T2"]}).set_index(
                    'FREQ')
                df_load_table_phase_fy = pd.DataFrame({'FREQ': list_of_all_frequencies,
                                                       'Phase_T2': interface_dicts_full[interface_name][
                                                           "Phase_T2"]}).set_index('FREQ')
                df_load_table_fy_real = pd.DataFrame(
                    df_load_table_fy['T2'] * np.cos(df_load_table_phase_fy['Phase_T2']), columns=['Real_T2'])
                df_load_table_fy_imag = pd.DataFrame(
                    df_load_table_fy['T2'] * np.sin(df_load_table_phase_fy['Phase_T2']), columns=['Imag_T2'])

                df_load_table_fz = pd.DataFrame(
                    {'FREQ': list_of_all_frequencies, 'T3': interface_dicts_full[interface_name]["T3"]}).set_index(
                    'FREQ')
                df_load_table_phase_fz = pd.DataFrame({'FREQ': list_of_all_frequencies,
                                                       'Phase_T3': interface_dicts_full[interface_name][
                                                           "Phase_T3"]}).set_index('FREQ')
                df_load_table_fz_real = pd.DataFrame(
                    df_load_table_fz['T3'] * np.cos(df_load_table_phase_fz['Phase_T3']), columns=['Real_T3'])
                df_load_table_fz_imag = pd.DataFrame(
                    df_load_table_fz['T3'] * np.sin(df_load_table_phase_fz['Phase_T3']), columns=['Imag_T3'])

                list_of_mx_values = [self.Quantity(v, "kN m") for v in interface_dicts_full[interface_name]["R1"]]
                list_of_my_values = [self.Quantity(v, "kN m") for v in interface_dicts_full[interface_name]["R2"]]
                list_of_mz_values = [self.Quantity(v, "kN m") for v in interface_dicts_full[interface_name]["R3"]]
                list_of_angle_mx_values = [self.Quantity(v, "deg") for v in
                                           interface_dicts_full[interface_name]["Phase_R1"]]
                list_of_angle_my_values = [self.Quantity(v, "deg") for v in
                                           interface_dicts_full[interface_name]["Phase_R2"]]
                list_of_angle_mz_values = [self.Quantity(v, "deg") for v in
                                           interface_dicts_full[interface_name]["Phase_R3"]]

                df_load_table_mx = pd.DataFrame(
                    {'FREQ': list_of_all_frequencies, 'R1': interface_dicts_full[interface_name]["R1"]}).set_index(
                    'FREQ')
                df_load_table_phase_mx = pd.DataFrame({'FREQ': list_of_all_frequencies,
                                                       'Phase_R1': interface_dicts_full[interface_name][
                                                           "Phase_R1"]}).set_index('FREQ')
                df_load_table_mx_real = pd.DataFrame(
                    df_load_table_mx['R1'] * np.cos(df_load_table_phase_mx['Phase_R1']), columns=['Real_R1'])
                df_load_table_mx_imag = pd.DataFrame(
                    df_load_table_mx['R1'] * np.sin(df_load_table_phase_mx['Phase_R1']), columns=['Imag_R1'])

                df_load_table_my = pd.DataFrame(
                    {'FREQ': list_of_all_frequencies, 'R2': interface_dicts_full[interface_name]["R2"]}).set_index(
                    'FREQ')
                df_load_table_phase_my = pd.DataFrame({'FREQ': list_of_all_frequencies,
                                                       'Phase_R2': interface_dicts_full[interface_name][
                                                           "Phase_R2"]}).set_index('FREQ')
                df_load_table_my_real = pd.DataFrame(
                    df_load_table_my['R2'] * np.cos(df_load_table_phase_my['Phase_R2']), columns=['Real_R2'])
                df_load_table_my_imag = pd.DataFrame(
                    df_load_table_my['R2'] * np.sin(df_load_table_phase_my['Phase_R2']), columns=['Imag_R2'])

                df_load_table_mz = pd.DataFrame(
                    {'FREQ': list_of_all_frequencies, 'R3': interface_dicts_full[interface_name]["R3"]}).set_index(
                    'FREQ')
                df_load_table_phase_mz = pd.DataFrame({'FREQ': list_of_all_frequencies,
                                                       'Phase_R3': interface_dicts_full[interface_name][
                                                           "Phase_R3"]}).set_index('FREQ')
                df_load_table_mz_real = pd.DataFrame(
                    df_load_table_mz['R3'] * np.cos(df_load_table_phase_mz['Phase_R3']), columns=['Real_R3'])
                df_load_table_mz_imag = pd.DataFrame(
                    df_load_table_mz['R3'] * np.sin(df_load_table_phase_mz['Phase_R3']), columns=['Imag_R3'])

                force_HR.XComponent.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
                force_HR.YComponent.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
                force_HR.ZComponent.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
                force_HR.XPhaseAngle.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
                force_HR.YPhaseAngle.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
                force_HR.ZPhaseAngle.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
                force_HR.XComponent.Output.DiscreteValues = list_of_fx_values
                force_HR.YComponent.Output.DiscreteValues = list_of_fy_values
                force_HR.ZComponent.Output.DiscreteValues = list_of_fz_values
                force_HR.XPhaseAngle.Output.DiscreteValues = list_of_angle_fx_values
                force_HR.YPhaseAngle.Output.DiscreteValues = list_of_angle_fy_values
                force_HR.ZPhaseAngle.Output.DiscreteValues = list_of_angle_fz_values

                moment_HR.XComponent.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
                moment_HR.YComponent.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
                moment_HR.ZComponent.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
                moment_HR.XPhaseAngle.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
                moment_HR.YPhaseAngle.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
                moment_HR.ZPhaseAngle.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
                moment_HR.XComponent.Output.DiscreteValues = list_of_mx_values
                moment_HR.YComponent.Output.DiscreteValues = list_of_my_values
                moment_HR.ZComponent.Output.DiscreteValues = list_of_mz_values
                moment_HR.XPhaseAngle.Output.DiscreteValues = list_of_angle_mx_values
                moment_HR.YPhaseAngle.Output.DiscreteValues = list_of_angle_my_values
                moment_HR.ZPhaseAngle.Output.DiscreteValues = list_of_angle_mz_values

                command_snippet_RM = analysis_HR.AddCommandSnippet()
                command_snippet_RM.Name = "Commands_RM_" + interface_name

                apdl_lines_RMx = self._create_APDL_table(df_load_table_mx_real * scale_factor,
                                                         "table_X_" + moment_HR_index_name, data_domain)
                apdl_lines_RMy = self._create_APDL_table(df_load_table_my_real * scale_factor,
                                                         "table_Y_" + moment_HR_index_name, data_domain)
                apdl_lines_RMz = self._create_APDL_table(df_load_table_mz_real * scale_factor,
                                                         "table_Z_" + moment_HR_index_name, data_domain)
                apdl_lines_RMxi = self._create_APDL_table(df_load_table_mx_imag * scale_factor,
                                                          "table_Xi_" + moment_HR_index_name, data_domain)
                apdl_lines_RMyi = self._create_APDL_table(df_load_table_my_imag * scale_factor,
                                                          "table_Yi_" + moment_HR_index_name, data_domain)
                apdl_lines_RMzi = self._create_APDL_table(df_load_table_mz_imag * scale_factor,
                                                          "table_Zi_" + moment_HR_index_name, data_domain)

                command_snippet_RM.AppendText(''.join(apdl_lines_RMx))
                command_snippet_RM.AppendText(''.join(apdl_lines_RMy))
                command_snippet_RM.AppendText(''.join(apdl_lines_RMz))
                command_snippet_RM.AppendText(''.join(apdl_lines_RMxi))
                command_snippet_RM.AppendText(''.join(apdl_lines_RMyi))
                command_snippet_RM.AppendText(''.join(apdl_lines_RMzi))
                command_snippet_RM.AppendText("\n\n! Apply the load on the remote point specified for the interface\n")
                command_snippet_RM.AppendText(f"nsel,s,node,,RP_{interface_index_no}\n")
                command_snippet_RM.AppendText(
                    f"f, all, mx, %table_X_{moment_HR_index_name}%, %table_Xi_{moment_HR_index_name}%\n")
                command_snippet_RM.AppendText(
                    f"f, all, my, %table_Y_{moment_HR_index_name}%, %table_Yi_{moment_HR_index_name}%\n")
                command_snippet_RM.AppendText(
                    f"f, all, mz, %table_Z_{moment_HR_index_name}%, %table_Zi_{moment_HR_index_name}%\n")
                command_snippet_RM.AppendText("nsel,all\n")

                def are_all_zeroes(*lists):
                    return all(all(x == 0 for x in lst) for lst in lists)

                if are_all_zeroes(interface_dicts_full[interface_name]["R1"],
                                  interface_dicts_full[interface_name]["R2"],
                                  interface_dicts_full[interface_name]["R3"]):
                    moment_HR.Delete()
                    command_snippet_RM.Delete()
                if are_all_zeroes(interface_dicts_full[interface_name]["T1"],
                                  interface_dicts_full[interface_name]["T2"],
                                  interface_dicts_full[interface_name]["T3"]):
                    force_HR.Delete()

                interface_index_no += 1

            save_path = os.path.join(os.getcwd(), "WE_Loading_Template_Harmonic.mechdat")
            self.app_ansys.save(save_path)

            # Define paths for the folder and the two possible .acmo files
            base_name_for_folder = os.path.splitext(save_path)[0]
            dir_path_to_delete = f"{base_name_for_folder}_Mech_Files"
            acmo_file_1_to_delete = f"{save_path}.acmo"
            acmo_file_2_to_delete = f"{save_path}_Independent.acmo"

            # Delete the folder if it exists
            if os.path.exists(dir_path_to_delete) and os.path.isdir(dir_path_to_delete):
                shutil.rmtree(dir_path_to_delete)
                print(f"Cleaned up folder: {dir_path_to_delete}")

            # Delete the first .acmo file if it exists
            if os.path.exists(acmo_file_1_to_delete) and os.path.isfile(acmo_file_1_to_delete):
                os.remove(acmo_file_1_to_delete)
                print(f"Cleaned up file: {acmo_file_1_to_delete}")

            # Delete the second .acmo file if it exists
            if os.path.exists(acmo_file_2_to_delete) and os.path.isfile(acmo_file_2_to_delete):
                os.remove(acmo_file_2_to_delete)
                print(f"Cleaned up file: {acmo_file_2_to_delete}")

            self.app_ansys.print_tree(self.DataModel.Project.Model)
            QMessageBox.information(None, "Extraction Complete",
                                    f"Harmonic template created successfully:\n\n{save_path}")
            os.startfile(os.getcwd())

        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(None, 'Ansys Export Error',
                                 f"An error occurred during harmonic template creation:\n{e}\n\nTraceback:\n{tb}")
        finally:
            self._close_ansys_session()

    def create_transient_template(self, df_export, data_domain, sample_rate):
        if not self._init_ansys_session():
            return
        try:
            all_keys = ["T1", "T2", "T3", "R1", "R2", "R3"]

            def get_full_interface_name(column_name):
                return re.sub(r'\s+[TR][1-3]$', '', column_name)

            full_interfaces = OrderedDict()
            for col in df_export.columns:
                if col != "TIME":
                    interface_name = get_full_interface_name(col)
                    if interface_name not in full_interfaces:
                        full_interfaces[interface_name] = []
                    full_interfaces[interface_name].append(col)

            def create_full_interface_dict(df, columns):
                interface_dict = {key: [0] * len(df) for key in all_keys}
                for col in columns:
                    key = col.split()[-1]
                    interface_dict[key] = df[col].tolist()
                return interface_dict

            interface_dicts_full = {interface: create_full_interface_dict(df_export, cols) for interface, cols in
                                    full_interfaces.items()}
            list_of_part_interface_names = list(interface_dicts_full.keys())

            scale_factor = 1000
            list_of_all_time_points = df_export["TIME"].tolist()

            self.Model.AddGeometryImportGroup().AddGeometryImport()

            analysis_static = self.Model.AddStaticStructuralAnalysis()
            analysis_settings_static = analysis_static.AnalysisSettings
            analysis_settings_static.PropertyByName("SolverUnitsControl").InternalValue = 1

            analysis_modal = self.Model.AddModalAnalysis()
            analysis_settings_modal = analysis_modal.AnalysisSettings
            self.DataModel.GetObjectsByName("Pre-Stress (None)")[0].PreStressICEnvironment = analysis_static
            analysis_settings_modal.PropertyByName("NumModesToFind").InternalValue = 100
            analysis_settings_modal.PropertyByName("RangeSearch").InternalValue = 1
            analysis_settings_modal.PropertyByName("MaxFrequency").InternalValue = sample_rate
            analysis_settings_modal.PropertyByName("SolverUnitsControl").InternalValue = 1

            analysis_TR = self.Model.AddTransientStructuralAnalysis()
            analysis_settings_TR = analysis_TR.AnalysisSettings
            self.DataModel.GetObjectsByName("Modal (None)")[0].ModalICEnvironment = analysis_modal
            analysis_settings_TR.PropertyByName("TimeStepDefineby").InternalValue = 0
            analysis_settings_TR.PropertyByName("NumberOfSubSteps").InternalValue = len(list_of_all_time_points)
            analysis_settings_TR.PropertyByName("EndTime").InternalValue = max(list_of_all_time_points)
            analysis_settings_TR.PropertyByName("MSUPSkipExpansion").InternalValue = 1
            analysis_settings_TR.PropertyByName("CombineDistResultFile").InternalValue = 1
            analysis_settings_TR.PropertyByName("ExpandResultFrom").InternalValue = 1
            analysis_settings_TR.PropertyByName("ConstantDampingValue").InternalValue = 0.02
            analysis_settings_TR.PropertyByName("SolverUnitsControl").InternalValue = 1
            analysis_settings_TR.IncludeResidualVector = True

            python_code_for_cleanup = analysis_TR.Solution.AddPythonCodeEventBased()
            python_code_for_cleanup.TargetCallback = self.Ansys.Mechanical.DataModel.Enums.PythonCodeTargetCallback.OnAfterPost
            python_code_for_cleanup.Text = """
import os

def delete_files_except_mcf(directory):
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path) and not file_name.endswith(".mcf"):
            os.remove(file_path)

def after_post(this, solution):
    solution_directory_path = solution.WorkingDir
    if solution_directory_path:
        delete_files_except_mcf(solution_directory_path)
    else:
        print("Error: Working directory not found.")
"""
            number_of_rows_in_mechanical = 50000
            number_of_partitions = math.ceil(len(list_of_all_time_points) / number_of_rows_in_mechanical)

            interface_index_no = 1
            for interface_name in list_of_part_interface_names:
                NS_interface = self.Model.AddNamedSelection()
                NS_interface.Name = "NS_" + interface_name

                CS_interface = self.Model.CoordinateSystems.AddCoordinateSystem()
                CS_interface.Name = "CS_" + interface_name

                RP_interface = self.Model.AddRemotePoint()
                RP_interface.Name = "RP_" + interface_name
                RP_interface.CoordinateSystem = CS_interface
                RP_interface.PilotNodeAPDLName = "RP_" + str(interface_index_no)

                df_load_table_fx = pd.DataFrame(
                    {'TIME': list_of_all_time_points, 'T1': interface_dicts_full[interface_name]["T1"]})
                df_load_table_fy = pd.DataFrame(
                    {'TIME': list_of_all_time_points, 'T2': interface_dicts_full[interface_name]["T2"]})
                df_load_table_fz = pd.DataFrame(
                    {'TIME': list_of_all_time_points, 'T3': interface_dicts_full[interface_name]["T3"]})
                df_load_table_mx = pd.DataFrame(
                    {'TIME': list_of_all_time_points, 'R1': interface_dicts_full[interface_name]["R1"]})
                df_load_table_my = pd.DataFrame(
                    {'TIME': list_of_all_time_points, 'R2': interface_dicts_full[interface_name]["R2"]})
                df_load_table_mz = pd.DataFrame(
                    {'TIME': list_of_all_time_points, 'R3': interface_dicts_full[interface_name]["R3"]})

                partitioned_df_load_table_fx = self._partition_dataframe_for_load_input(df_load_table_fx,
                                                                                        number_of_rows_in_mechanical)
                partitioned_df_load_table_fy = self._partition_dataframe_for_load_input(df_load_table_fy,
                                                                                        number_of_rows_in_mechanical)
                partitioned_df_load_table_fz = self._partition_dataframe_for_load_input(df_load_table_fz,
                                                                                        number_of_rows_in_mechanical)
                partitioned_df_load_table_mx = self._partition_dataframe_for_load_input(df_load_table_mx,
                                                                                        number_of_rows_in_mechanical)
                partitioned_df_load_table_my = self._partition_dataframe_for_load_input(df_load_table_my,
                                                                                        number_of_rows_in_mechanical)
                partitioned_df_load_table_mz = self._partition_dataframe_for_load_input(df_load_table_mz,
                                                                                        number_of_rows_in_mechanical)

                force_TR_list_of_all_partitions = []
                moment_TR_list_of_all_partitions = []

                for i in range(number_of_partitions):
                    force_TR = analysis_TR.AddRemoteForce()
                    force_TR.DefineBy = self.Ansys.Mechanical.DataModel.Enums.LoadDefineBy.Components
                    force_TR.Name = f"RF_{interface_name}_Part_{i + 1}"
                    force_TR.PropertyByName("GeometryDefineBy").InternalValue = 1
                    force_TR.Location = NS_interface
                    force_TR.CoordinateSystem = CS_interface
                    force_TR_list_of_all_partitions.append(force_TR)

                    moment_TR = analysis_TR.AddMoment()
                    moment_TR.DefineBy = self.Ansys.Mechanical.DataModel.Enums.LoadDefineBy.Components
                    moment_TR.Name = f"RM_{interface_name}_Part_{i + 1}"
                    moment_TR.PropertyByName("GeometryDefineBy").InternalValue = 2
                    moment_TR.Location = RP_interface
                    moment_TR_list_of_all_partitions.append(moment_TR)

                    force_TR.XComponent.Inputs[0].DiscreteValues = [self.Quantity(v, "s") for v in
                                                                    partitioned_df_load_table_fx[i]['TIME']]
                    force_TR.YComponent.Inputs[0].DiscreteValues = [self.Quantity(v, "s") for v in
                                                                    partitioned_df_load_table_fy[i]['TIME']]
                    force_TR.ZComponent.Inputs[0].DiscreteValues = [self.Quantity(v, "s") for v in
                                                                    partitioned_df_load_table_fz[i]['TIME']]
                    force_TR.XComponent.Output.DiscreteValues = [self.Quantity(v, "kN") for v in
                                                                 partitioned_df_load_table_fx[i]['T1']]
                    force_TR.YComponent.Output.DiscreteValues = [self.Quantity(v, "kN") for v in
                                                                 partitioned_df_load_table_fy[i]['T2']]
                    force_TR.ZComponent.Output.DiscreteValues = [self.Quantity(v, "kN") for v in
                                                                 partitioned_df_load_table_fz[i]['T3']]

                    moment_TR.XComponent.Inputs[0].DiscreteValues = [self.Quantity(v, "s") for v in
                                                                     partitioned_df_load_table_mx[i]['TIME']]
                    moment_TR.YComponent.Inputs[0].DiscreteValues = [self.Quantity(v, "s") for v in
                                                                     partitioned_df_load_table_my[i]['TIME']]
                    moment_TR.ZComponent.Inputs[0].DiscreteValues = [self.Quantity(v, "s") for v in
                                                                     partitioned_df_load_table_mz[i]['TIME']]
                    moment_TR.XComponent.Output.DiscreteValues = [self.Quantity(v, "kN m") for v in
                                                                  partitioned_df_load_table_mx[i]['R1']]
                    moment_TR.YComponent.Output.DiscreteValues = [self.Quantity(v, "kN m") for v in
                                                                  partitioned_df_load_table_my[i]['R2']]
                    moment_TR.ZComponent.Output.DiscreteValues = [self.Quantity(v, "kN m") for v in
                                                                  partitioned_df_load_table_mz[i]['R3']]

                def are_all_zeroes(*lists):
                    return all(all(x == 0 for x in lst) for lst in lists)

                if are_all_zeroes(interface_dicts_full[interface_name]["R1"],
                                  interface_dicts_full[interface_name]["R2"],
                                  interface_dicts_full[interface_name]["R3"]):
                    for moment_TR in moment_TR_list_of_all_partitions:
                        moment_TR.Delete()
                if are_all_zeroes(interface_dicts_full[interface_name]["T1"],
                                  interface_dicts_full[interface_name]["T2"],
                                  interface_dicts_full[interface_name]["T3"]):
                    for force_TR in force_TR_list_of_all_partitions:
                        force_TR.Delete()

                interface_index_no += 1

            save_path = os.path.join(os.getcwd(), "WE_Loading_Template_Transient.mechdat")
            self.app_ansys.save(save_path)

            # Define paths for the folder and the two possible .acmo files
            base_name_for_folder = os.path.splitext(save_path)[0]
            dir_path_to_delete = f"{base_name_for_folder}_Mech_Files"
            acmo_file_1_to_delete = f"{save_path}.acmo"
            acmo_file_2_to_delete = f"{save_path}_Independent.acmo"

            # Delete the folder if it exists
            if os.path.exists(dir_path_to_delete) and os.path.isdir(dir_path_to_delete):
                shutil.rmtree(dir_path_to_delete)
                print(f"Cleaned up folder: {dir_path_to_delete}")

            # Delete the first .acmo file if it exists
            if os.path.exists(acmo_file_1_to_delete) and os.path.isfile(acmo_file_1_to_delete):
                os.remove(acmo_file_1_to_delete)
                print(f"Cleaned up file: {acmo_file_1_to_delete}")

            # Delete the second .acmo file if it exists
            if os.path.exists(acmo_file_2_to_delete) and os.path.isfile(acmo_file_2_to_delete):
                os.remove(acmo_file_2_to_delete)
                print(f"Cleaned up file: {acmo_file_2_to_delete}")

            self.app_ansys.print_tree(self.DataModel.Project.Model)
            QMessageBox.information(None, "Extraction Complete",
                                    f"Transient template created successfully:\n\n{save_path}")
            os.startfile(os.getcwd())

        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(None, 'Ansys Export Error',
                                 f"An error occurred during transient template creation:\n{e}\n\nTraceback:\n{tb}")
        finally:
            self._close_ansys_session()

    def _create_APDL_table(self, result_df, table_name, data_domain):
        num_rows, num_cols = result_df.shape
        apdl_lines = []
        values = result_df.values
        row_indices = result_df.index

        apdl_lines.append(f"\n*DIM,{table_name},TABLE,{num_rows},{num_cols},1,{data_domain}\n\n")
        apdl_lines.extend(
            [f"*SET,{table_name}({i + 1},0,1),{row_index}\n" for i, row_index in enumerate(row_indices)])
        apdl_lines.append("\n")

        for i in range(num_rows):
            for j in range(num_cols):
                if not pd.isna(values[i, j]):
                    apdl_lines.append(f"*SET,{table_name}({i + 1},{j + 1},1),{values[i, j]}\n")
        return apdl_lines

    @staticmethod
    def _partition_dataframe_for_load_input(df, partition_size):
        partitions = []
        prev_last_row = None
        time_column = df.columns[0]
        data_column = df.columns[1]

        for i in range(0, len(df), partition_size):
            partition = df.iloc[i:i + partition_size]
            zero_row = pd.DataFrame({time_column: [0], data_column: [0]})
            if prev_last_row is not None:
                prev_last_row_zero = prev_last_row.copy()
                prev_last_row_zero[data_column] = 0
                partition_with_last = pd.concat([zero_row, prev_last_row_zero, partition]).reset_index(drop=True)
            else:
                partition_with_last = pd.concat([zero_row, partition]).reset_index(drop=True)

            prev_last_row = partition.tail(1)
            partitions.append(partition_with_last)

        return partitions
