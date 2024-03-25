import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from multiprocessing import Pool
import pickle
import gzip

from tqdm import tqdm

kc1 = 300; # production of p53 [nM/h]
kc2 = 1 # degradation of p53 by mdm2 [1/h]
kc3 = 6; # degradation of p53 by mdm2 [nM]
kc4 = 0.03; # production of mdm2 - mRNA [1/(nM*h)]
kc5 = 0.5 # degradation of mdm2 - mRNA [1/h]
kc6 = 1.4 # production of mdm2 [1/h]
kc7 = 0.6 # degradation of mdm2 [1/h] 0.2
kc8 = 0.01 # binding of Mdm2 and nutlin [1/h]
kc9 = 1 # unbinding of Mdm2 and nutlin [1/h]

# Fetched from "p53 chaos with transient"
x0 = [93.89461266006715, 253.74067308694282, 343.7284733975012, 5.778558282567199e-21]
T_int, A_int = 5.579249993159683, 42.923495054906745

def p_change(t, p, m, Mf, Mb, n):
    dpdt = kc1 - kc2*Mf*(p / (kc3 + p))
    return dpdt

def m_change(t, p, m, Mf, Mb, n):
    dmdt = kc4 * p**2 - kc5 * m
    return dmdt

def M_free_change(t, p, m, Mf, Mb, n):
    dM_freedt = kc6 * m - kc7 * Mf - kc8*n*Mf + kc9 * Mb # Last term new
    return dM_freedt

def M_bound_change(t, p, m, Mf, Mb, n):
    dM_bounddt = - kc7*Mb + kc8*n*Mf - kc9 * Mb # Last term new
    return dM_bounddt

def n_change(t, p, m, Mf, Mb, n):
    dndt = -kc8*n*Mf + kc9 * Mb # Last term new
    return dndt

def system_nutlin_free(t, y):
    dydt = np.zeros_like(y)
    dydt[0] = p_change(t, *y)
    dydt[1] = m_change(t, *y)
    dydt[2] = M_free_change(t, *y)
    dydt[3] = M_bound_change(t, *y)
    dydt[4] = n_change(t, *y)
    return dydt

def n_zero(t, p, m, Mf, Mb, n):
    dndt = 0
    return dndt

def system_nutlin_forced_zero(t, y):
    dydt = np.zeros_like(y)
    dydt[0] = p_change(t, *y)
    dydt[1] = m_change(t, *y)
    dydt[2] = M_free_change(t, *y)
    dydt[3] = M_bound_change(t, *y)
    dydt[4] = n_zero(t, *y)
    return dydt


def sim_onoff_nutlin(oscillationer, A_ext, OOmega):
    T_external = OOmega * T_int

    history = {"t":[] ,"p": [], "m": [], "Mf": [], "Mb":[], "n":[]}

    no_half_periods = 2 * oscillationer

    half_period = T_external/2

    for i in tqdm(range(no_half_periods + 1)):
        if i == 0:
            state = np.concatenate((x0, [A_ext]))
            sys = solve_ivp(system_nutlin_free, (0, half_period), state, method='LSODA', max_step=0.5, dense_output=True)
        elif (i != 0 and i % 2 == 0):
            state = history["p"][-1], history["m"][-1], history["Mf"][-1], history["Mb"][-1], A_ext
            sys = solve_ivp(system_nutlin_free, (0, half_period), state, method='LSODA', max_step=0.5, dense_output=True)
        else:
            state = history["p"][-1], history["m"][-1], history["Mf"][-1], history["Mb"][-1], 0
            sys = solve_ivp(system_nutlin_forced_zero, (0, half_period), state, method='LSODA', max_step=0.5, dense_output=True)

        p, m, Mf, Mb, n = sys["y"][0], sys["y"][1], sys["y"][2], sys["y"][3], sys["y"][4]
        
        new_t = np.array(sys["t"]) + (history["t"][-1] if history["t"] else 0)
        history["t"].extend(new_t)
        history["p"].extend(p)
        history["m"].extend(m)
        history["Mf"].extend(Mf)
        history["Mb"].extend(Mb)
        history["n"].extend(n)
    
    return history, T_external

def worker(args):
    omega_list, coupling_strength_list, A_ext_list = args
    all_tongues = [0.5, 1, 1.5, 2, 2.5, 3]
    arnold_tongue_dict = {"Omega": [], "T_ext":[], "calculated_ratio": [], "coupling_strength": [], "A_ext": [], "entrainment_value": [], "exception": []}
    
    for i in range(len(omega_list)):
        print(i)
        for j in range(len(A_ext_list)):
            oscillationer = 300

            simulation, T_external = sim_onoff_nutlin(oscillationer, A_ext_list[j], omega_list[i])
            _, p_sim, _, _, _, n_sim = np.array(simulation["t"]), np.array(simulation["p"]), np.array(simulation["m"]), np.array(simulation["Mb"]), np.array(simulation["Mf"]), np.array(simulation["n"])
            
            peaks_external_idx = find_peaks(n_sim, height=np.mean(n_sim))[0]
            window_idx = peaks_external_idx[99]

            peaks_external = len(find_peaks(n_sim[window_idx:], height=np.mean(n_sim[window_idx:]))[0])
            peaks_internal = len(find_peaks(p_sim[window_idx:], height=np.mean(p_sim[window_idx:]), prominence = 10)[0])
            
            Omega_ratio = 0
            exception = False
        
            try:
                Omega_ratio = peaks_internal/peaks_external
            except:
                exception = True

            rounded_ratio = np.round(Omega_ratio, 4)

            entrainment = 0
            for tongue_value in all_tongues:
                if tongue_value - 1e-2 <= rounded_ratio <= tongue_value + 1e-2:
                    entrainment = tongue_value
            
            arnold_tongue_dict["Omega"].append(omega_list[i])
            arnold_tongue_dict["coupling_strength"].append(coupling_strength_list[j])
            arnold_tongue_dict["A_ext"].append(A_ext_list[j])
            arnold_tongue_dict["T_ext"].append(T_external)
            arnold_tongue_dict["calculated_ratio"].append(Omega_ratio)
            arnold_tongue_dict["entrainment_value"].append(entrainment)
            arnold_tongue_dict["exception"].append(exception)
    
    return arnold_tongue_dict

def arnold_tongue_simulering_parallel(omega_list, coupling_strength_list, A_ext_list, num_processes):
    chunk_size = len(omega_list) // num_processes
    omega_chunks = [omega_list[i:i + chunk_size] for i in range(0, len(omega_list), chunk_size)]

    args_list = [(omega_chunk, coupling_strength_list, A_ext_list) for omega_chunk in omega_chunks]
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(worker, args_list)
    
    combined_result = {"Omega": [], "T_ext":[], "calculated_ratio": [], "coupling_strength": [], "A_ext": [], "entrainment_value": [], "exception": []}
    for result in results:
        combined_result["Omega"].extend(result["Omega"])
        combined_result["coupling_strength"].extend(result["coupling_strength"])
        combined_result["A_ext"].extend(result["A_ext"])
        combined_result["T_ext"].extend(result["T_ext"])
        combined_result["calculated_ratio"].extend(result["calculated_ratio"])
        combined_result["entrainment_value"].extend(result["entrainment_value"])
        combined_result["exception"].extend(result["exception"])
    
    return combined_result

def save_data(data, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    antal_omegaer = 12
    antal_A_ext = 36

    omega_list = np.linspace(0.001, 3.5, antal_omegaer)

    coupling_strengths, _, _ = np.split(np.linspace(0.001, 10, antal_A_ext), 3)

    A_ext_list = coupling_strengths * A_int
    arnold_dict_parallel = arnold_tongue_simulering_parallel(omega_list, coupling_strengths, A_ext_list, 4)

    save_data(arnold_dict_parallel, f'arnold/1_p53_chaos_{antal_omegaer}by{antal_A_ext}.pkl.gz')