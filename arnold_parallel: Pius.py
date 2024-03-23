import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from multiprocessing import Pool
import pickle
import gzip
import time




# Simulation params
vi = 0.01
Vol = vi*10**(-14); NA = 6.02*10**(23); Cal = NA*Vol*10**(-6)
TC = 0.158
kc1 = 0.15*Cal*TC
# production of p53 [molecules/min]
kc2 = 0.1*TC
# degradation of p53 by mdm2 [1/min]
kc3 = 0.1*Cal
# degradation of p53 by mdm2 [molecules]
kc4 = 0.1/Cal*TC
# production of mdm2 - mRNA [1/(molecules*min)]
kc5 = 0.1*TC 
# degradation of mdm2 - mRNA [1/min]
kc6 = 0.2*TC 
# production of mdm2 [1/min]
kc7 = 0.1*TC 
# degradation of mdm2 [1/min]
kc8 = 0.0036 
# binding of mdm2 and nutlin [1/(molecules*min)]

p_transient, m_transient, Mf_transient, Mb_transient = (68.7677614186749, 39.447611187479886, 62.52296546950754, -1.4584740455754785e-07)
T_int, A_int = (329.1125843870209, 32.35464308364189) # These are from the initial conditions found in "Nutlin-tilføjelse.ipynb" so we don't have to run a new transient every time.

# Equations
def p_change(t, p, m, Mf, Mb, n):
    dpdt = kc1 - kc2*Mf*(p / (kc3 + p))
    return dpdt

def m_change(t, p, m, Mf, Mb, n):
    dmdt = (kc4 * p**2 - kc5 * m)
    return dmdt

def M_free_change(t, p, m, Mf, Mb, n):
    dM_freedt = kc6 * m - kc7 * Mf - kc8*n*Mf
    return dM_freedt

def M_bound_change(t, p, m, Mf, Mb, n):
    dM_bounddt = - kc7*Mb + kc8*n*Mf
    return dM_bounddt

def n_change(t, p, m, Mf, Mb, n):
    dndt = -kc8*n*Mf
    return dndt

def samlet_system(t, y):
    dydt = np.zeros_like(y)
    dydt[0] = p_change(t, *y)
    dydt[1] = m_change(t, *y)
    dydt[2] = M_free_change(t, *y)
    dydt[3] = M_bound_change(t, *y)
    dydt[4] = n_change(t, *y)
    return dydt

def sim_onoff_nutlin(oscillationer, OOmega, A_ext):
    # This is correct, and it is working! Remember to count in period values, not frequency! If you count in frequency, you are counting the reciprocal.
    T_ext = OOmega * T_int
    theoretical_small_omega = T_int / T_ext # Is this correct?

    history = {"t":[] ,"p": [], "m": [], "Mf": [], "Mb":[], "n":[]}

    for i in range(oscillationer):
        if i == 0:
            state = p_transient, m_transient, Mf_transient, Mb_transient, A_ext
            sys = solve_ivp(samlet_system, (0, T_ext), state, method='RK45', max_step=5, dense_output=True)
            t, p, m, Mf, Mb, n = np.array(sys["t"]), sys["y"][0], sys["y"][1], sys["y"][2], sys["y"][3], sys["y"][4]
            history["t"].extend(t)
            history["p"].extend(p)
            history["m"].extend(m)
            history["Mf"].extend(Mf)
            history["Mb"].extend(Mb)
            history["n"].extend(n)
        
        # Every period, Nutlin concentration is reset to A_ext
        state = history["p"][-1], history["m"][-1], history["Mf"][-1], history["Mb"][-1], A_ext
        sys = solve_ivp(samlet_system, (0, T_ext), state, method='RK45', max_step=5, dense_output=True)
        t, p, m, Mf, Mb, n = np.array(sys["t"]), sys["y"][0], sys["y"][1], sys["y"][2], sys["y"][3], sys["y"][4]
        history["t"].extend(t + history["t"][-1])
        history["p"].extend(p)
        history["m"].extend(m)
        history["Mf"].extend(Mf)
        history["Mb"].extend(Mb)
        history["n"].extend(n)
    
    return history, theoretical_small_omega, T_ext

# Worker function to be executed by each process
def worker(args):
    omega_list, coupling_strength_list, A_ext_list = args
    all_tongues = [0.5, 1, 1.5, 2, 2.5]
    arnold_tongue_dict = {"Omega": [], "calculated_ratio": [], "coupling_strength": [], "A_ext": [], "entrainment_value": [], "exception": [], "trajectory": []}
    
    for i in range(len(omega_list)):
        for j, A_ext in enumerate(A_ext_list):
            oscillationer = 200

            simulation, _, _ = sim_onoff_nutlin(oscillationer, omega_list[i], A_ext)
            t_sim, p_sim, m_sim, Mb_sim, Mf_sim, n_sim = np.array(simulation["t"]), np.array(simulation["p"]), np.array(simulation["m"]), np.array(simulation["Mb"]), np.array(simulation["Mf"]), np.array(simulation["n"])

            # Apply a small window and require peak higher than mean
            window = 500
            peaks_internal = len(find_peaks(p_sim[window:], height=np.mean(p_sim))[0])
            peaks_external = len(find_peaks(n_sim[window:], height=np.mean(n_sim))[0])
            
            Omega_ratio = 0
            exception = False
            
            # If for some reason this fraction throws an error, we log it instead of the simulation just crashing
            try:
                # 1/Omega because we are using frequency, which is 1/T
                Omega_ratio = peaks_internal/peaks_external
            except:
                exception = True

            # We chose this as our accuracy to begin with, but it seems like we are getting results which resembles the ground "truth", so we are happy with it so far
            rounded_ratio = np.round(Omega_ratio, 4)

            entrainment = 0 # Corresponds to no entrainment

            # Arbritrary threshold, we hope it works. But it is symmetric here, so we don't understand why the entrainment is shifted.
            for tongue_value in all_tongues:
                if tongue_value - 1e-2 <= rounded_ratio <= tongue_value + 1e-2:
                    entrainment = tongue_value
            
            arnold_tongue_dict["Omega"].append(omega_list[i])
            arnold_tongue_dict["coupling_strength"].append(coupling_strength_list[j])
            arnold_tongue_dict["A_ext"].append(A_ext)
            arnold_tongue_dict["calculated_ratio"].append(Omega_ratio)
            arnold_tongue_dict["entrainment_value"].append(entrainment)
            arnold_tongue_dict["exception"].append(exception)
            arnold_tongue_dict["trajectory"].append([t_sim, p_sim, m_sim, Mb_sim, Mf_sim, n_sim])
    
    return arnold_tongue_dict

def arnold_tongue_simulering_parallel(omega_list, coupling_strength_list, A_ext_list, num_processes):
    # Splitting omega_list into chunks for each process
    chunk_size = len(omega_list) // num_processes
    omega_chunks = [omega_list[i:i + chunk_size] for i in range(0, len(omega_list), chunk_size)]
    
    # Each process will get its own omega range and the full A_ext_list
    args_list = [(omega_chunk, coupling_strength_list, A_ext_list) for omega_chunk in omega_chunks]
    
    # Initialize multiprocessing pool and distribute the work
    with Pool(processes=num_processes) as pool:
        results = pool.map(worker, args_list)
    
    # Combine results from all processes
    combined_result = {"Omega": [], "calculated_ratio": [], "coupling_strength": [], "A_ext": [], "entrainment_value": [], "exception": [], "trajectory": []}
    for result in results:
        combined_result["Omega"].extend(result["Omega"])
        combined_result["coupling_strength"].extend(result["coupling_strength"])
        combined_result["A_ext"].extend(result["A_ext"])
        combined_result["calculated_ratio"].extend(result["calculated_ratio"])
        combined_result["entrainment_value"].extend(result["entrainment_value"])
        combined_result["exception"].extend(result["exception"])
        combined_result["trajectory"].extend(result["trajectory"])
    
    return combined_result

def save_data(data, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    # Record the start time
    start_time = time.time()

    antal_omegaer = 64
    antal_A_ext = 64

    omega_list = np.array(np.linspace(0.01, 3.1, antal_omegaer))
    coupling_strengths = np.array(np.linspace(0.01, 5, antal_A_ext))
    A_ext_list = coupling_strengths * A_int

    # DANGER - FOR RUNNING ON MODI MOUNT!
    arnold_dict_parallel = arnold_tongue_simulering_parallel(omega_list, coupling_strengths, A_ext_list, 64)

    save_data(arnold_dict_parallel, f'arnold/arnold_tongue_dict{antal_omegaer}by{antal_A_ext}.pkl.gz')

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")