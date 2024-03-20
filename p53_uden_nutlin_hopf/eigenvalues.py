import sympy as sp
from sympy.abc import p, m, M
from sympy import Matrix
import numpy as np
import matplotlib.pyplot as plt

sp.init_printing()

#definerer symbolerne for parameterne
k1 = sp.Symbol('k1')
k2 = sp.Symbol('k2')
k3 = sp.Symbol('k3')
k4 = sp.Symbol('k4')
k5 = sp.Symbol('k5')
k6 = sp.Symbol('k6')
k7 = sp.Symbol('k7')

#definerer funktioner
dp = k1 - k2 * (M * p) / (k3 + p)
dm = k4 * p**2 - k5 * m
dM = k6 * m - k7 * M

#udregner indgange jacobiantmatricen
ddpdp = sp.diff(dp, p)
ddpdm = sp.diff(dp, m)
ddpdM = sp.diff(dp, M)
ddmdp = sp.diff(dm, p)
ddmdm = sp.diff(dm, m)
ddmdM = sp.diff(dm, M)
ddMdp = sp.diff(dM, p)
ddMdm = sp.diff(dM, m)
ddMdM = sp.diff(dM, M)

#defienrer jacobiantmatricen
jacobian = Matrix([[ddpdp, ddpdm, ddpdM], [ddmdp, ddmdm, ddmdM], [ddMdp, ddMdm, ddMdM]])

#laver funktion som udregner egenværdier med fikspunkt og k2 som input
def eigenvals(fp, k1, k2, k3, k4, k5, k6, k7):
    #indsætter paramtere og fikspunkt i jacobianten
    ev_jac = jacobian.subs({p: fp[0], m: fp[1], M: fp[2], k1: k1, k2: k2, k3: k3, k4: k4, k5: k5, k6: k6, k7: k7})
    #defienerer liste med nuller
    eigenvalues = np.zeros(len(jacobian[0,:]))
    #fylder listen ud med egenværdier
    for i in range (len(jacobian[0,:])):
        eigenvalues[i] = sp.re(list(ev_jac.eigenvals().keys())[i])
    return eigenvalues

# Denne funktion tager en matrix med (komplekse) egenværdier og plotter dem i (Re, Im)-planet.
# Hver sammenhørende række i matrixen har samme index (fordi de stammer fra samme værdi af k_2)
#Husk nul-indeksering!! (jeg har korrigeret for det på plottet)
def plot_egenværdier(matrix_med_egenværdier):

    plt.figure(figsize=(8, 6))

    for i, row in enumerate(matrix_med_egenværdier):
        reel = [z.real for z in row]
        imaginær = [z.imag for z in row]
        plt.scatter(reel, imaginær, label=f'Row {i+1}')

        for j, z in enumerate(row):
            shift = 0.3
            plt.text(z.real + shift, z.imag + shift, f'{i+1}', fontsize=9, ha='center', va='center', color="red")
    
    plt.xlim(-max(abs(5+max(np.real(matrix_med_egenværdier).flatten())), abs(min(-5+np.real(matrix_med_egenværdier).flatten()))),
              max(abs(5+max(np.real(matrix_med_egenværdier).flatten())), abs(min(-5+np.real(matrix_med_egenværdier).flatten()))))
    plt.ylim(-max(abs(5+max(np.imag(matrix_med_egenværdier).flatten())), abs(min(-5+np.imag(matrix_med_egenværdier).flatten()))),
              max(abs(5+max(np.imag(matrix_med_egenværdier).flatten())), abs(min(-5+np.imag(matrix_med_egenværdier).flatten()))))

    plt.axhline(0, color='black', linewidth=2)
    plt.axvline(0, color='black', linewidth=2)
    
    plt.xlabel('Reel')
    plt.ylabel('Imaginær')
    plt.title('Plot of egenværdier for varierende værdier af k2')
    
    plt.grid(True)
    plt.show()