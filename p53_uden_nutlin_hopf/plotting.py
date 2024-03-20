import numpy as np
import matplotlib.pyplot as plt

def plot_systems_evolution_2d_3d(simulations, single_fig = True, dpi = 200, filename = "testing", save_fig = False):
    # Calculate the number of simulations to determine the size of the grid
    n = len(simulations)
    
    time_arr = simulations[0]["time"] # Each simulation has same time scale, so only extract once
    
    if not single_fig:
        for i, simulation in enumerate(simulations):
            # Create a new figure for each simulation
            fig = plt.figure(figsize=(12, 4), dpi=dpi)

            # Extract system values
            p_values = simulation["trajectory"]["p"]
            m_values = simulation["trajectory"]["m"]
            M_values = simulation["trajectory"]["M"]

            # 2D plot for the i-th simulation
            ax_2d = fig.add_subplot(1, 2, 1)
            ax_2d.plot(time_arr, p_values, "g-", label='p (Runge Kutta)')
            ax_2d.plot(time_arr, m_values, label='m (Runge Kutta)')
            ax_2d.plot(time_arr, M_values, label='M (Runge Kutta)')
            ax_2d.set_xlabel('Time [~min]')
            ax_2d.set_ylabel('Molecule Concentration [A.U.]')
            ax_2d.set_title(f'k2 = {simulation["k2_val"]} of system evolution')
            ax_2d.legend()

            # 3D plot for the i-th simulation
            ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
            ax_3d.plot(p_values, m_values, M_values, label='System Trajectory')
            ax_3d.set_xlabel('p')
            ax_3d.set_ylabel('m')
            ax_3d.set_zlabel('M')
            ax_3d.set_title('3D Phase Space')
            ax_3d.legend()

            plt.tight_layout()

            if save_fig:
                plt.savefig(f"figures/{filename}_{i}.png", dpi=dpi)
            else:
                # Display the plot
                plt.show()
    else:
        # Create a figure with subplots in an Nx2 grid
        fig = plt.figure(figsize=(12, 4 * n), dpi=dpi)  # Adjust the figure size
        for i, simulation in enumerate(simulations):
            # Extract system values
            p_values = simulation["trajectory"]["p"]
            m_values = simulation["trajectory"]["m"]
            M_values = simulation["trajectory"]["M"]

            # 2D plot for the i-th simulation
            ax_2d = fig.add_subplot(n, 2, 2*i + 1)  # Correct index for 2D plot
            ax_2d.plot(time_arr, p_values, "g-", label='p (Runge Kutta)')
            ax_2d.plot(time_arr, m_values, label='m (Runge Kutta)')
            ax_2d.plot(time_arr, M_values, label='M (Runge Kutta)')
            ax_2d.set_xlabel('Time [~min]')
            ax_2d.set_ylabel('Molecule Concentration [A.U.]')
            ax_2d.set_title(f'k2 = {simulation["k2_val"]} of system evolution')
            ax_2d.legend()

            # 3D plot for the i-th simulation
            ax_3d = fig.add_subplot(n, 2, 2*i + 2, projection='3d')  # Correct index for 3D plot
            ax_3d.plot(p_values, m_values, M_values, label='System Trajectory')
            ax_3d.set_xlabel('p')
            ax_3d.set_ylabel('m')
            ax_3d.set_zlabel('M')
            ax_3d.set_title(f'3D Phase Space')
            ax_3d.legend()

        plt.title("Simulating the p53 network")

        if save_fig:
            plt.savefig(f"figures/{filename}.png", dpi=dpi)

        plt.tight_layout()
        plt.show()


def plot_systems_evolution_p53vsmdm2(simulations, single_fig = True, dpi = 100, filename = "testing", save_fig = False):
    # Calculate the number of simulations to determine the size of the grid
    n = len(simulations)
    
    time_arr = simulations[0]["time"] # Each simulation has same time scale, so only extract once
    # Create a figure with subplots in an Nx2 grid

    if not single_fig:
        for i, simulation in enumerate(simulations):
            # Create a new figure for each simulation
            fig = plt.figure(figsize=(12, 4), dpi=dpi)

            # Extract system values
            p_values = simulation["trajectory"]["p"]
            m_values = simulation["trajectory"]["m"]
            M_values = simulation["trajectory"]["M"]

            # 2D plot for the i-th simulation
            ax_2d = fig.add_subplot(1, 2, 1)  # Correct index for 2D plot
            ax_2d.plot(p_values, m_values, "g-", label='p (Runge Kutta)')
            # ax_2d.plot(time_arr, m_values, label='m (Runge Kutta)')
            # ax_2d.plot(time_arr, M_values, label='M (Runge Kutta)')
            ax_2d.set_xlabel('p53 Concentration [A.U.]')
            ax_2d.set_ylabel('mdm2 Concentration [A.U.]')
            ax_2d.set_title(f'k2 = {simulation["k2_val"]} of system evolution')
            ax_2d.legend()

            if save_fig:
                plt.savefig(f"figures/{filename}_{i}.png", dpi=dpi)
            
            plt.show()
    else:
        fig = plt.figure(figsize=(12, 4 * n), dpi=dpi)  # Adjust the figure size
        for i, simulation in enumerate(simulations):
            # Extract system values
            p_values = simulation["trajectory"]["p"]
            m_values = simulation["trajectory"]["m"]
            M_values = simulation["trajectory"]["M"]

            # 2D plot for the i-th simulation
            ax_2d = fig.add_subplot(n, 2, 2*i + 1)  # Correct index for 2D plot
            ax_2d.plot(p_values, m_values, "g-", label='p (Runge Kutta)')
            # ax_2d.plot(time_arr, m_values, label='m (Runge Kutta)')
            # ax_2d.plot(time_arr, M_values, label='M (Runge Kutta)')
            ax_2d.set_xlabel('p53 Concentration [A.U.]')
            ax_2d.set_ylabel('mdm2 Concentration [A.U.]')
            ax_2d.set_title(f'k2 = {simulation["k2_val"]} of system evolution')
            ax_2d.legend()

        plt.title("Simulating the p53 network")

        if save_fig:
            plt.savefig(f"figures/{filename}.png", dpi=dpi)

        plt.tight_layout()
        plt.show()

def plot_systems_evolution_p53_time_delay(simulations, single_fig = True, dpi = 100, filename = "testing", save_fig = False):
    # Calculate the number of simulations to determine the size of the grid
    n = len(simulations)
    
    time_arr = simulations[0]["time"] # Each simulation has same time scale, so only extract once
    # Create a figure with subplots in an Nx2 grid

    if not single_fig:
        for i, simulation in enumerate(simulations):
            # Create a new figure for each simulation
            fig = plt.figure(figsize=(12, 4), dpi=dpi)

            # Extract system values
            p_values = simulation["trajectory"]["p"]
            m_values = simulation["trajectory"]["m"]
            M_values = simulation["trajectory"]["M"]

            # 2D plot for the i-th simulation
            ax_2d = fig.add_subplot(1, 2, 1)  # Correct index for 2D plot
            ax_2d.plot(p_values, m_values, "g-", label='p (Runge Kutta)')
            # ax_2d.plot(time_arr, m_values, label='m (Runge Kutta)')
            # ax_2d.plot(time_arr, M_values, label='M (Runge Kutta)')
            ax_2d.set_xlabel('p53 Concentration [A.U.]')
            ax_2d.set_ylabel('mdm2 Concentration [A.U.]')
            ax_2d.set_title(f'k2 = {simulation["k2_val"]} of system evolution')
            ax_2d.legend()

            if save_fig:
                plt.savefig(f"figures/{filename}_{i}.png", dpi=dpi)
            
            plt.show()
    else:
        fig = plt.figure(figsize=(12, 4 * n), dpi=dpi)  # Adjust the figure size
        for i, simulation in enumerate(simulations):
            # Extract system values
            p_values = simulation["trajectory"]["p"]
            m_values = simulation["trajectory"]["m"]
            M_values = simulation["trajectory"]["M"]

            # 2D plot for the i-th simulation
            ax_2d = fig.add_subplot(n, 2, 2*i + 1)  # Correct index for 2D plot
            ax_2d.plot(p_values, m_values, "g-", label='p (Runge Kutta)')
            # ax_2d.plot(time_arr, m_values, label='m (Runge Kutta)')
            # ax_2d.plot(time_arr, M_values, label='M (Runge Kutta)')
            ax_2d.set_xlabel('p53 Concentration [A.U.]')
            ax_2d.set_ylabel('mdm2 Concentration [A.U.]')
            ax_2d.set_title(f'k2 = {simulation["k2_val"]} of system evolution')
            ax_2d.legend()

        plt.title("Simulating the p53 network")

        if save_fig:
            plt.savefig(f"figures/{filename}.png", dpi=dpi)

        plt.tight_layout()
        plt.show()

    