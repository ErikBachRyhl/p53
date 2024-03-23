import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from multiprocessing import Pool
import pickle
import gzip

import time

# defining parameters from Krishna/Jensen
k_Nin = 5.4 # min^{-1}
k_lin = 0.018 # min^{-1} 
k_t = 1.03 # mu * M^{-1} * min^{-1}
k_tl = 0.24 # min^{-1}
K_I = 0.035 # mu * M
K_N = 0.029 # mu * M
gamma_m = 0.017 # min^{-1}
alpha = 1.05 # mu * M^{-1} * min^{-1}
N_tot = 1.0 # mu * M
k_a = 0.24 # min^{-1}
k_i = 0.18 # min^{-1}
k_p = 0.036 # min^{-1}
k_A20 = 0.0018 # mu * M
IKK_tot = 2.0 # mu * M
A20 = 0.0026 # mu * M

# defining constant to keep TNF>0
k = 0.5

no_osc = 200

# Transient finish params
T_intt = 108.44444444444444
A_intt = 0.1293511878383227
p0 = [0.037770776771579556, 0.4050017925580332, 4.076546559955566, 0.1799999999999995, 1.5500000000000012]

# defining equations from Krishna/Jensen
def N_n_change(t, N_n, I_m, I, IKK_a, IKK_i, TNF):
    dN_ndt = k_Nin * (N_tot - N_n) * K_I / (K_I + I) - k_lin * I * (N_n / (K_N + N_n))
    return dN_ndt

def I_m_change(t, N_n, I_m, I, IKK_a, IKK_i, TNF):
    dI_mdt = k_t * (N_n**2) - gamma_m * I_m
    return dI_mdt

def I_change(t, N_n, I_m, I, IKK_a, IKK_i, TNF):
    dIdt = k_tl * I_m - alpha * IKK_a * (N_tot - N_n) * I / (K_I + I)
    return dIdt

def IKK_a_change(t, N_n, I_m, I, IKK_a, IKK_i, TNF):
    dIKK_adt = k_a * TNF * (IKK_tot - IKK_a - IKK_i) - k_i * IKK_a
    return dIKK_adt

def IKK_i_change(t, N_n, I_m, I, IKK_a, IKK_i, TNF):
    dIKK_idt = k_i * IKK_a - k_p * IKK_i * k_A20 / (k_A20 + A20 * TNF)
    return dIKK_idt

def system_nfkb(t, y, Amp_new, T_ext):
    # Extract current state variables, assume TNF is not part of y here
    N_n, I_m, I, IKK_a, IKK_i = y
    # Dynamically compute TNF based on current time t (this means it will be a constant passed to the next equations for this dt)
    TNF = k + Amp_new * np.sin(np.pi * (1 / T_ext) * t*2*np.pi)
    # calling differential equation functions, passing TNF as an argument
    # Updating function definitions accordingly if they require TNF
    return [N_n_change(t, *y, TNF), 
            I_m_change(t, *y, TNF), 
            I_change(t, *y, TNF), 
            IKK_a_change(t, *y, TNF), 
            IKK_i_change(t, *y, TNF)]

# --------

# defining function to oscillate TNF (sinusoidal)
def TNF_sin_osc(oscillations, OOmega, A_ext):

    # calculating the period of TNF in order to simulate with the desired ratio of external over internal period
    T_extt = OOmega * T_intt
    omega = T_intt / T_extt

    history = {"t":[] ,"N_n": [], "I_m": [], "I": [], "IKK_a":[], "IKK_i":[], "TNF":[]}

    
    for i in range(oscillations):
        if i == 0:
            state = p0
        else:
            state = history["N_n"][-1], history["I_m"][-1], history["I"][-1], history["IKK_a"][-1], history["IKK_i"][-1]

        sys = solve_ivp(system_nfkb, (0, T_extt), state, args=(A_ext, T_extt), method='RK45', max_step=1, dense_output=True)
        
        N_n, I_m, I, IKK_a, IKK_i = sys["y"][0], sys["y"][1], sys["y"][2], sys["y"][3], sys["y"][4]

        new_t = np.array(sys["t"]) + (history["t"][-1] if history["t"] else 0)

        history["t"].extend(new_t)
        history["N_n"].extend(N_n)
        history["I_m"].extend(I_m)
        history["I"].extend(I)
        history["IKK_a"].extend(IKK_a)
        history["IKK_i"].extend(IKK_i)
        history["TNF"].extend(k + A_ext * np.sin(np.pi * (1 / T_extt) * new_t))

    return history, omega, T_extt

# Worker function to be executed by each process
def worker(args):
    omega_list, coupling_strength_list, A_ext_list = args
    all_tongues = [0.5, 1, 1.5, 2, 2.5]
    arnold_tongue_dict = {"Omega": [], "calculated_ratio": [], "coupling_strength": [], "A_ext": [], "entrainment_value": [], "exception": [], "trajectory": []}
    
    for i in range(len(omega_list)):
        for j, A_ext in enumerate(A_ext_list):
            simulation, _, _ = TNF_sin_osc(no_osc, omega_list[i], A_ext)
            t_sim, N_n_sim, I_m_sim, I_sim, _, _, TNF_sim = np.array(simulation["t"]), np.array(simulation["N_n"]), np.array(simulation["I_m"]), np.array(simulation["I"]), np.array(simulation["IKK_a"]), np.array(simulation["IKK_i"]), np.array(simulation["TNF"])
            # Apply a small window and require peak higher than mean
            window = 1500 # Around the same ratio as the window for p53. Maybe this needs to be much larger...
            peaks_internal = len(find_peaks(N_n_sim[window:], height=np.mean(N_n_sim))[0])
            peaks_external = len(find_peaks(TNF_sim[window:], height=np.mean(TNF_sim))[0])
            
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
            arnold_tongue_dict["trajectory"].append({"t": t_sim ,"N_n": N_n_sim, "I_m": I_m_sim, "I": I_sim, "TNF":TNF_sim})
    
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

    omega_list = np.array(np.linspace(0.01, 3, antal_omegaer))
    coupling_strengths = np.array(np.linspace(0.1, 5, antal_A_ext))
    A_ext_list = coupling_strengths * A_intt

    # DANGER - FOR RUNNING ON MODI MOUNT!
    arnold_dict_parallel = arnold_tongue_simulering_parallel(omega_list, coupling_strengths, A_ext_list, 4)

    save_data(arnold_dict_parallel, f'arnold_sims/nfkb_arnold{antal_omegaer}by{antal_A_ext}.pkl.gz')

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")

