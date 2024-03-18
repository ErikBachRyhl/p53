import numpy as np
import h5py

def slice_trajectory(simulation_dict, time_start, time_end):
    """
    Slices the trajectory of a simulation to keep only the data within the specified time range.

    Parameters:
    - simulation_dict: dict, containing simulation data with keys "k2_val", "trajectory", and "time".
    - time_start: float, the start of the time range.
    - time_end: float, the end of the time range.

    Returns:
    - A dictionary with the same structure as simulation_dict but with the trajectory data sliced to the specified time range.
    """
    # Extract the trajectory and time arrays
    p_values = simulation_dict["trajectory"]["p"]
    m_values = simulation_dict["trajectory"]["m"]
    M_values = simulation_dict["trajectory"]["M"]
    time_arr = simulation_dict["time"]

    # Find indices where time is within the specified range
    indices = [i for i, t in enumerate(time_arr) if time_start <= t <= time_end]
    
    # Slice arrays based on the found indices
    sliced_trajectory = {
        "k2_val": simulation_dict["k2_val"],
        "trajectory": {
            "p": [p_values[i] for i in indices],
            "m": [m_values[i] for i in indices],
            "M": [M_values[i] for i in indices],
        },
        "time": [time_arr[i] for i in indices]
    }

    return sliced_trajectory

def sliced_simulations(simulations_dictionary, t_start, t_end):
    sliced = []
    for simulation in simulations_dictionary:
        print(len(simulation))
        sliced.append(slice_trajectory(simulation, t_start, t_end))
    
    return sliced

# # Function for choosing simulations ending in fixed point.
# def find_fixed_points_simulations(simulations):
#     fixed_points = []
#     for simulation in simulations:
#         # Access the specific structure as described
#         last_five_values = np.round(simulation["trajectory"]["p"][-5:], 4)
        
#         # Check if the last five values are the same
#         if np.all(last_five_values == last_five_values[0]):
#             fixed_points.append(simulation)
    
#     print(f"No. of simulations settling to fixed points: {len(fixed_points)}")
#     return fixed_points

def read_simulation(h5_file, k2_val):
    """
    Read a specific simulation based on k2_val from an HDF5 file.

    Parameters:
    - h5_file: Path to the HDF5 file.
    - k2_val: The k2 value of the simulation to read.

    Returns:
    A dictionary containing the k2_val, trajectories, and time_arr.
    """

    with h5py.File(h5_file, 'r') as f:
        data = {}
        grp = f[str(k2_val)]
        data['k2_val'] = k2_val
        data['trajectory'] = {key: grp[key][:] for key in ('p', 'm', 'M')}
        data['time'] = grp['time'][:]
    return data

def read_all_simulations(h5_file):
    """
    Read all simulations from an HDF5 file and return them as an array of dictionaries.

    Parameters:
    - h5_file: Path to the HDF5 file.

    Returns:
    An array of dictionaries, each representing a simulation with 'k2_val', 'trajectories', and 'time_arr'.
    """
    simulations = []
    with h5py.File(h5_file, 'r') as f:
        # Iterate over all groups (each representing a simulation)
        for k2_val in f.keys():
            grp = f[k2_val]
            trajectories = {key: grp[key][:] for key in ('p', 'm', 'M')}
            time_arr = grp['time'][:]
            simulations.append({"k2_val": k2_val, "trajectory": trajectories, "time": time_arr})
    return simulations