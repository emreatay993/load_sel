"""
© K. Emre Atay

plasticity_correction_batch.py

Neuber plasticity correction with temperature-dependent multilinear hardening data.
"""

# region Import libraries
from __future__ import annotations
import math
import time
import unittest
from typing import Tuple, List
from numba import njit
from numba import prange
import numpy as np
import pandas as pd
# endregion

# region Input: MATERIAL DATABASE
# region Enter temperature vs. Young's Modulus of material
# Temperatures at which hardening curves are tabulated  (°C)
TEMP_TABLE = np.array([22.0], dtype=float)

# Young’s modulus at those temps (MPa)
YOUNGS_MODULUS_TABLE = np.array([70e3], dtype=float)
# endregion

# region Enter plastic-strain vs true-stress curves:
# rows = temperatures, columns = curve points                 (-)
EPS_P_TABLE = np.array([
    [0.0,   0.3007]
], dtype=float)

# Matching true-stress points (MPa)
SIG_TP_TABLE = np.array([
    [400.0, 1004.3]
], dtype=float)

N_PTS = SIG_TP_TABLE.shape[1]           # points per curve
# endregion
# endregion

# region Data Loading Function
def load_fea_output(filepath: str) -> Tuple[pd.DataFrame, str]:
    """
    Loads data from a whitespace-separated FEA file and auto-detects column names.

    It reads the header to find the name of the last column (the value column)
    and returns a clean, two-column DataFrame with standardized names
    ['NodeID', 'Value'] along with the original name of the value column.

    Args:
        filepath: The path to the input text file.

    Returns:
        A tuple containing:
        - A pandas DataFrame with 'NodeID' and 'Value' columns.
        - The auto-detected name of the value column from the file's header.
    """
    try:
        # Use pandas to read the space-separated file and infer the header.
        raw_df = pd.read_csv(filepath, sep='\\t')

        # Get the name of the first and last columns from the file.
        node_col_name = raw_df.columns[0]
        value_col_name = raw_df.columns[-1]

        # Create a clean DataFrame with standardized column names for internal use.
        clean_df = pd.DataFrame({
            "NodeID": raw_df[node_col_name].astype(int),
            "Value": raw_df[value_col_name]
        })

        print(f"  - Successfully loaded '{filepath}'. Detected value column: '{value_col_name}'")
        return clean_df, value_col_name

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        raise
    except Exception as e:
        print(f"Error: Failed to parse the file '{filepath}'. Reason: {e}")
        raise
# endregion

# region Helper routines for interpolation
@njit(cache=True)
def interpolate_plastic_strain(target_stress: float,
                               stress_points: np.ndarray,
                               plastic_strain_points: np.ndarray) -> float:
    """
    Interpolates or extrapolates the plastic strain for a given true stress using a tabulated stress-strain curve.

    Parameters
    ----------
    target_stress : float
        The true stress (MPa) at which to find the plastic strain.
    stress_points : np.ndarray
        Array of tabulated true stress values (MPa), sorted in ascending order.
    plastic_strain_points : np.ndarray
        Array of tabulated plastic strain values (dimensionless), matching stress_points.

    Returns
    -------
    float
        The estimated plastic strain corresponding to target_stress.

    Notes
    -----
    - If target_stress is less than the first tabulated value, returns 0.0 (material behaves elastically).
    - If target_stress is between tabulated points, returns a linearly interpolated value.
    - If target_stress is above the highest tabulated value, returns a linearly extrapolated value.
    """
    # If you are below the first tabulated stress: stress is purely elastic, so no plastic strain
    if target_stress <= stress_points[0]:
        return 0.0

    # Loop through each segment between tabulated points to find where the target_stress fits
    for idx in range(len(stress_points) - 1):
        stress_start, stress_end = stress_points[idx], stress_points[idx + 1]
        if stress_start <= target_stress <= stress_end:
            strain_start, strain_end = plastic_strain_points[idx], plastic_strain_points[idx + 1]

            # Linear interpolation between the two points
            return strain_start + (target_stress - stress_start) * (strain_end - strain_start) / (stress_end - stress_start)

    # If above the last tabulated point, perform linear extrapolation using the last two points
    stress_start, stress_end = stress_points[-2], stress_points[-1]
    strain_start, strain_end = plastic_strain_points[-2], plastic_strain_points[-1]
    slope = (strain_end - strain_start) / (stress_end - stress_start)
    return strain_end + (target_stress - stress_end) * slope

@njit(cache=True)
def interpolate_properties_by_temperature(my_temp: float, my_true_stress: float) -> Tuple[float, float]:
    """
    Interpolate Young's modulus and plastic strain at a given temperature and stress,
    blending between two bounding temperature curves.

    Parameters
    ----------
    my_temp : float
        The temperature (°C) for which properties are desired.
    my_true_stress : float
        The true stress (MPa) for which plastic strain is needed.

    Parameters (Global)
    ----------
    TEMP_TABLE : np.ndarray
        Array of tabulated temperatures (°C), sorted ascending.
    YOUNGS_MODULUS_TABLE : np.ndarray
        Array of Young's modulus values (MPa), matching temperatures.
    SIG_TP_TABLE : np.ndarray
        2D array of tabulated true stresses (MPa) for each temperature row.
    EPS_P_TABLE : np.ndarray
        2D array of plastic strain values for each temperature row.

    Returns
    -------
    (float, float)
        Tuple containing:
        - Interpolated Young's modulus at the given temperature (MPa).
        - Interpolated plastic strain at the given temperature and true stress (dimensionless).

    Notes
    -----
    - If temperature is below the lowest tabulated value, the first curve is used.
    - If above the highest, the last curve is used.
    - Otherwise, linear interpolation is performed between the bounding curves.
    """

    # region Find indices for temperatures just below and just above the requested temperature
    lower_index = np.searchsorted(TEMP_TABLE, my_temp) - 1
    lower_index = max(lower_index, 0)
    upper_index = min(lower_index + 1, TEMP_TABLE.size - 1)
    # endregion

    # region Calculate interpolation weight (0.0 if at lower_index, 1.0 if at upper_index)
    if lower_index == upper_index:
        weight = 0.0
    else:
        weight = (my_temp - TEMP_TABLE[lower_index]) / (TEMP_TABLE[upper_index] - TEMP_TABLE[lower_index])
    # endregion

    # region Interpolate Young's modulus between bounding temperatures
    E_T = (1 - weight) * YOUNGS_MODULUS_TABLE[lower_index] + weight * YOUNGS_MODULUS_TABLE[upper_index]
    # endregion

    # region Interpolate plastic strain for each bounding curve at the given true stress
    eps_low = interpolate_plastic_strain(my_true_stress, SIG_TP_TABLE[lower_index], EPS_P_TABLE[lower_index])
    eps_high = interpolate_plastic_strain(my_true_stress, SIG_TP_TABLE[upper_index], EPS_P_TABLE[upper_index])
    eps_p_T = (1 - weight) * eps_low + weight * eps_high
    # endregion

    return E_T, eps_p_T

@njit(cache=True)
def interpolate_yield_strength(my_temp: float) -> float:
    """
    Numba-JITed yield strength estimate by linearly blending
    the first tab point of each curve at temperature T.

    Parameters
    ----------
    temperature : float
        The temperature (°C) at which to estimate the yield strength.
    temperatures : np.ndarray
        1D array of tabulated temperatures (°C), sorted ascending.

    Parameters (Global)
    ----------
    SIG_TP_TABLE : np.ndarray
        2D array of tabulated true stresses (MPa) for each temperature row.
    TEMP_TABLE : np.ndarray
        Array of tabulated temperatures (°C), sorted ascending.

    Returns
    -------
    float
        The estimated yield strength (MPa) at the given interpolated temperature.

    Notes
    -----
    - If the temperature is below the tabulated range, the lowest value is returned.
    - If above the highest, the highest value is returned.
    - Otherwise, the value is linearly interpolated between the two bounding temperatures.
    """

    # region find lower index
    lower_index = np.searchsorted(TEMP_TABLE, my_temp) - 1
    lower_index = max(lower_index, 0)
    upper_index = min(lower_index + 1, len(TEMP_TABLE) - 1)
    # endregion

    # region If at the lower or upper boundary, no interpolation needed, otherwise, compute interpolation weight
    if lower_index == upper_index:
        weight = 0.0
    else:
        temp_low = TEMP_TABLE[lower_index]
        temp_high = TEMP_TABLE[upper_index]
        weight = (my_temp - temp_low) / (temp_high - temp_low)
    # endregion

    # region Linear interpolation of yield strength between the bounding temperatures
    yield_low = SIG_TP_TABLE[lower_index, 0]
    yield_high = SIG_TP_TABLE[upper_index, 0]
    interpolated_yield_strength = (1.0 - weight) * yield_low + weight * yield_high
    # endregion

    return interpolated_yield_strength
# endregion

# region Neuber correction engine
@njit(cache=True)
def solve_neuber_scalar(my_elastic_stress: float,
                        my_temp: float,
                        max_iterations: int = 60,
                        tolerance: float = 1e-10) -> Tuple[float, float]:
    """
    Solve Neuber's rule for a single node, with temperature-dependent plasticity.

    Neuber's equation:
        (σₑ)² / E  =  σ · (εₑ + εₚ)

    where
      σₑ   = elastic (FE) stress,
      σ    = corrected (true) stress,
      εₑ   = elastic strain at the corrected stress = σ / E,
      εₚ   = plastic strain.

    Parameters
    ----------
    my_elastic_stress : float
        The (elastic) stress from linear analysis (MPa).
    my_temp : float
        The local temperature of node (°C).
    max_iterations : int
        Maximum allowed iterations for Newton's method.
    tolerance : float
        Relative convergence tolerance for Newton's method.

    Parameters (Global)
    ----------
    TEMP_TABLE : np.ndarray
        1D array of tabulated temperatures (°C).
    YOUNGS_MODULUS_TABLE : np.ndarray
        1D array of Young's modulus values (MPa), matching temperatures.
    SIG_TP_TABLE : np.ndarray
        2D array of tabulated true stresses (MPa) for each temperature.
    EPS_P_TABLE : np.ndarray
        2D array of plastic strain values for each temperature.

    Returns
    -------
    tuple[float, float]
        (corrected_true_stress [MPa], plastic_strain [dimensionless])
    """

    # region (Edge-case) Exit the function early with zero values, if you somehow have non-positive stress
    if my_elastic_stress <= 0.0:
        return 0.0, 0.0
    # endregion

    # region Initial guess: use elastic solution or yield strength, whichever is lower
    interpolated_yield_strength = interpolate_yield_strength(my_temp)
    sigma = my_elastic_stress if my_elastic_stress < interpolated_yield_strength else interpolated_yield_strength
    if sigma <= 0.0:
        sigma = 1e-6  # To avoid divide by zero
    # endregion

    # region Newton-Raphson iteration to solve for corrected true stress
    for _ in prange(max_iterations):
        E, ε_p = interpolate_properties_by_temperature(my_temp, sigma)

        # region Define Neuber residual: it should be zero at solution
        neuber_residual = sigma / E + ε_p - (my_elastic_stress ** 2) / (sigma * E)
        # endregion

        # region Numerical derivative with respect to sigma (finite difference)
        delta_sigma = 1e-6 * max(abs(sigma), 1.0)
        E2, ε_p2 = interpolate_properties_by_temperature(my_temp, sigma + delta_sigma)
        neuber_residual2 = (sigma + delta_sigma) / E2 + ε_p2 - (my_elastic_stress ** 2) / ((sigma + delta_sigma) * E2)
        residual_derivative = (neuber_residual2 - neuber_residual) / delta_sigma
        # endregion

        # region Newton-Raphson update step
        step   = neuber_residual / residual_derivative
        sigma_next = sigma - step
        # Guard against stepping into negative stress (edge-case)
        if sigma_next <= 0.0:
            sigma_next = sigma * 0.5
        # endregion

        # region Check convergence based on default tolerance value
        if abs(step) / sigma < tolerance:
            sigma = sigma_next
            break
        sigma = sigma_next
        # endregion
    # endregion

    # region Calculate final plastic strain at the converged sigma
    _, eps_plastic_final = interpolate_properties_by_temperature(my_temp, sigma)
    return sigma, eps_plastic_final
    # endregion

@njit(cache=True, parallel=True)
def solve_neuber_vector(nodal_elastic_stresses_array: np.ndarray,
                        nodal_temperatures_array:   np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized wrapper for Neuber correction over arrays of nodes.

    Parameters
    ----------
    nodal_elastic_stresses_array : np.ndarray
        Array of elastic stress values at each node (MPa).
    nodal_temperatures_array : np.ndarray
        Array of temperature values at each node (°C).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of (corrected true stress [MPa], plastic strain [dimensionless]).
    """
    n = nodal_elastic_stresses_array.size
    corrected_stress = np.empty(n, dtype=np.float64)
    plastic_strain   = np.empty(n, dtype=np.float64)
    for i in prange(n):
        corrected_stress[i], plastic_strain[i] = \
            solve_neuber_scalar(float(nodal_elastic_stresses_array[i]), float(nodal_temperatures_array[i]))
    return corrected_stress, plastic_strain
# endregion

# region Handle inputs and run the neuber solver, generating output CSVs
def run_plasticity_correction(stress_filepath: str, temp_filepath: str, output_filepath: str):
    """
    Orchestrates the entire analysis workflow: loading, validation, solving, and writing.
    """
    # 1. Load data and automatically get original column names
    print("Step 1: Loading input files...")
    stress_df, stress_col_name = load_fea_output(stress_filepath)
    temp_df, temp_col_name = load_fea_output(temp_filepath)

    # 2. Verify node consistency
    print("\nStep 2: Verifying node consistency...")
    stress_nodes = set(stress_df["NodeID"])
    temp_nodes = set(temp_df["NodeID"])
    if stress_nodes != temp_nodes:
        # Error handling for mismatched nodes
        nodes_only_in_stress = stress_nodes - temp_nodes
        nodes_only_in_temp = temp_nodes - stress_nodes
        error_message = "Node ID mismatch found between input files.\n"
        if nodes_only_in_stress:
            error_message += f"  - {len(nodes_only_in_stress)} nodes found only in stress file (e.g., {list(nodes_only_in_stress)[:5]})\n"
        if nodes_only_in_temp:
            error_message += f"  - {len(nodes_only_in_temp)} nodes found only in temperature file (e.g., {list(nodes_only_in_temp)[:5]})"
        raise ValueError(error_message)
    print("  - Node sets are identical.")

    # 3. Prepare data for solver
    # Rename 'Value' columns to their original names to avoid clashes during merge
    stress_df.rename(columns={"Value": stress_col_name}, inplace=True)
    temp_df.rename(columns={"Value": temp_col_name}, inplace=True)
    merged_df = pd.merge(stress_df, temp_df, on="NodeID")

    # 4. Run the Neuber correction solver
    print("\nStep 3: Applying Neuber correction...")
    start_time = time.perf_counter()
    corrected_stress, plastic_strain = solve_neuber_vector(
        merged_df[stress_col_name].to_numpy(),
        merged_df[temp_col_name].to_numpy()
    )
    end_time = time.perf_counter()
    print(f"  - Solver finished in {end_time - start_time:.5f} seconds for {len(merged_df)} nodes.")

    # 5. Write results to output file
    print(f"\nStep 4: Writing results to '{output_filepath}'...")
    output_df = pd.DataFrame({
        "NodeID": merged_df["NodeID"],
        f"Corrected_{stress_col_name}": corrected_stress,
        "Plastic_Strain": plastic_strain
    })
    output_df.to_csv(output_filepath, index=False, float_format="%.6e")
    print("  - Processing complete.")

# endregion

# region SELF-TESTS (Unit-Tests)
# class _NeuberTests(unittest.TestCase):
#     """Run with:  python -m unittest neuber_plasticity.py  -or-  python file.py -v"""
#
#     def _almost(self, a, b, rel=1e-8):
#         self.assertAlmostEqual(a, b, delta=rel * max(abs(a), abs(b), 1.0))
#
#     # — Basic sanity ----------------------------------------------------------------
#     def test_elastic_below_yield(self):
#         σ_e, T = 200.0, 20.0               # below first yield point (250 MPa)
#         σcorr, εp = solve_neuber_scalar(σ_e, T)
#         self._almost(σcorr, σ_e)
#         self._almost(εp,     0.0)
#
#     def test_exact_yield(self):
#         σ_e, T = 250.0, 20.0               # exactly first tab point
#         σcorr, εp = solve_neuber_scalar(σ_e, T)
#         self._almost(σcorr, σ_e)
#         self._almost(εp,     0.0)
#
#     # — Plastic region --------------------------------------------------------------
#     def test_plastic_in_curve(self):
#         σ_e, T = 320.0, 20.0               # within curve range
#         σcorr, εp = solve_neuber_scalar(σ_e, T)
#         self.assertLess(σcorr, σ_e)
#         self.assertGreater(εp,  0.0)
#
#     def test_plastic_above_curve(self):
#         σ_e, T = 450.0, 20.0               # above last tab point
#         σcorr, εp = solve_neuber_scalar(σ_e, T)
#         self.assertGreaterEqual(σcorr, 0.0)
#         self.assertLess(σcorr, σ_e)
#         self.assertGreater(εp,  0.0)
#
#     # — Temperature extremes --------------------------------------------------------
#     def test_low_temperature(self):
#         σ_e, T = 150.0, -50.0              # below min temp – use first curve
#         σcorr, εp = solve_neuber_scalar(σ_e, T)
#         self._almost(σcorr, σ_e)
#         self._almost(εp,     0.0)
#
#     def test_high_temperature(self):
#         σ_e, T = 150.0, 600.0              # above max temp – use last curve
#         σcorr, εp = solve_neuber_scalar(σ_e, T)
#         self._almost(σcorr, σ_e)
#         self._almost(εp,     0.0)
#
#     # — Vector driver ---------------------------------------------------------------
#     def test_vector_wrapper(self):
#         σ_e = np.array([200.0, 320.0, 450.0])
#         T   = np.array([20.0,  20.0,  350.0])
#         σcorr, εp = solve_neuber_vector(σ_e, T)
#         self.assertEqual(σcorr.shape, σ_e.shape)
#         self.assertEqual(εp.shape,    σ_e.shape)
#         self.assertLess(σcorr[1], σ_e[1])   # 320 MPa case → plastic
#
#     # — Quick performance sanity (tiny batch) ---------------------------------------
#     def test_tiny_performance(self):
#         rng = np.random.default_rng(0)
#         σ_e = rng.uniform(50, 600, 10_000)
#         T   = rng.uniform(20, 500, 10_000)
#         t0  = time.perf_counter()
#         _   = solve_neuber_vector(σ_e, T)
#         self.assertLess(time.perf_counter() - t0, 0.5)   # < 0.5 s for 10 k
# endregion

# region Define a demo for performance check
def _demo_single_node():
    σ_e, T = 260.0, 20.0
    σcorr, εp = solve_neuber_scalar(σ_e, T)
    print("\nSingle-node demo")
    print(f"  Input : σ_e = {σ_e:.1f} MPa,  T = {T:.1f} °C")
    print(f"  Output: σ_corr = {σcorr:.1f} MPa,  ε_p = {εp:.3e}\n")

def _demo_small_cloud(n: int = 100_000):
    rng = np.random.default_rng(1)
    σ_e = rng.uniform(50, 600, size=n)
    T   = rng.uniform(22, 22.01, n)
    t0  = time.perf_counter()
    σc, εp = solve_neuber_vector(σ_e, T)
    dt  = time.perf_counter() - t0
    print(f"Processed {n/1e3:.0f} k nodes in {dt:.3f} s  →  "
          f"{n/dt/1e6:.2f} M nodes/s (pure NumPy)")
# endregion

# region Run the main routine
if __name__ == "__main__":
    # Run unit tests when executed directly
    #unittest.main(verbosity=2, exit=False)

    # Extra demos (comment out if not needed)
    _demo_single_node()
    _demo_small_cloud(500000)

    # Run the code for user input
    STRESS_FILE = "stress_225_Mpa.txt"
    TEMP_FILE = "temperature_22_C.txt"
    OUTPUT_FILE = "neuber_corrected_results.csv"
    try:
        run_plasticity_correction(
            stress_filepath=STRESS_FILE,
            temp_filepath=TEMP_FILE,
            output_filepath=OUTPUT_FILE
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"\nANALYSIS FAILED: {e}")

# endregion
