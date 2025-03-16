from ansys.dpf import post

solution = post.load_simulation(r"C:\Users\emre_\OneDrive\Desktop\J\ANSYS\Benchmark\Vibration_Fatigue\Vibration_Fatigue3_files\dp0\SYS-13\MECH", simulation_type = post.common.AvailableSimulationTypes.modal_mechanical)

stress_of_mode = solution.stress_nodal(modes=[1])

stress_of_mode.array
