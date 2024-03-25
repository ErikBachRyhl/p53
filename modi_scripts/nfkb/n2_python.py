import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from multiprocessing import Pool
import pickle
import gzip

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


# Mean value for TNF oscillations
k = 0.5

# This is fetched from "nfkappab.ipynb" where we calculated it with constant TNF = 0.5
x0 = [0.038121575605998374, 0.4043266277640023, 4.065919561814218, 0.1799999999999994, 1.5500000000000014]
T_int = 108.47222222222223
A_int = 0.12940028915485896

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

def system_nfkb(t, y, A_ext, T_ext):
    N_n, I_m, I, IKK_a, IKK_i = np.maximum(y, 0)  # Enforce non-negativity
    TNF = np.maximum(k + A_ext * np.sin(2 * np.pi * (1 / T_ext) * t), 0)
    
    return [N_n_change(t, *y, TNF), 
            I_m_change(t, *y, TNF), 
            I_change(t, *y, TNF), 
            IKK_a_change(t, *y, TNF), 
            IKK_i_change(t, *y, TNF)]


def TNF_sin_osc(oscillations, A_ext, OOmega):
    T_external = OOmega * T_int

    history = {"t":[] ,"N_n": [], "I_m": [], "I": [], "IKK_a":[], "IKK_i":[], "TNF":[]}
    
    for i in range(oscillations):
        if i == 0:
            state = x0
        else:
            state = history["N_n"][-1], history["I_m"][-1], history["I"][-1], history["IKK_a"][-1], history["IKK_i"][-1]

        sys = solve_ivp(system_nfkb, (0, T_external), state, args=(A_ext, T_external,), method='LSODA', max_step=1.5, dense_output=True)

        N_n, I_m, I, IKK_a, IKK_i = sys["y"][0], sys["y"][1], sys["y"][2], sys["y"][3], sys["y"][4]

        new_t = np.array(sys["t"]) + (history["t"][-1] if history["t"] else 0)
        history["t"].extend(new_t)
        history["N_n"].extend(N_n)
        history["I_m"].extend(I_m)
        history["I"].extend(I)
        history["IKK_a"].extend(IKK_a)
        history["IKK_i"].extend(IKK_i)
        history["TNF"].extend(k + A_ext * np.sin(2*np.pi * (1 / T_external) * new_t))

    return history, T_external

def worker(args):
    omega_list, coupling_strength_list, A_ext_list = args
    all_tongues = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    arnold_tongue_dict = {"Omega": [], "T_ext":[], "calculated_ratio": [], "coupling_strength": [], "A_ext": [], "entrainment_value": [], "exception": []}
    
    for i in range(len(omega_list)):
        print(i)
        for j in range(len(A_ext_list)):
            oscillationer = 300

            simulation, T_external = TNF_sin_osc(oscillationer, A_ext_list[j], omega_list[i])
            N_n, TNF_sim = np.array(simulation["N_n"]), np.array(simulation["TNF"])
            
            peaks_external_idx = find_peaks(TNF_sim, height=np.mean(TNF_sim))[0]
            window_idx = peaks_external_idx[99]

            peaks_external = len(find_peaks(TNF_sim[window_idx:], height=np.mean(TNF_sim[window_idx:]))[0])
            peaks_internal = len(find_peaks(N_n[window_idx:], height=np.mean(N_n[window_idx:]), prominence = 0.05)[0])
            
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
    antal_omegaer = 576
    antal_A_ext = 576

    omega_list = np.linspace(0.001, 3.5, antal_omegaer)

    _, coupling_strengths, _ = np.split(np.linspace(0.001, 3.8, antal_A_ext), 3)

    A_ext_list = coupling_strengths * A_int
    arnold_dict_parallel = arnold_tongue_simulering_parallel(omega_list, coupling_strengths, A_ext_list, 64)

    save_data(arnold_dict_parallel, f'arnold/2_nfkb_{antal_omegaer}by{antal_A_ext}.pkl.gz')