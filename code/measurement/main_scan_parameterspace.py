import numpy as np
import csv
import os

from measurement import init, functions

""" 
Script for obtaining measurement data (the distribution of the forward flow fraction gamma and the integral forward flow
fraction Gamma) for a specific range of t_p and t_off values. During the measurements, either the volume flow of the 
Pulsed Jet Actuators (PJAs) or the provided pressure can be kept constant, leading to a different behavior of the PJAs.
"""

# Choose between constant volume flow or constant pressure
static_vol_flow = False  # = False for constant pressure

# Initial / target p_control and volume flow are defined in the init.py script

# Define the parameter space
# 1ms - 25ms
t_p = np.arange(0.001, 0.025, 0.001)  # (s)
t_off = np.arange(0.001, 0.025, 0.001)  # (s)

# Define the file path for saving the data and write the first row
save_dir = r'C:\Users\Kalibrier-Windkanal\Documents\AFC_ReinforcementLearning\save_dir'
file_path = 'const_p-control0.5V_1-8ms_in1ms.txt'
with open(os.path.join(save_dir, file_path), 'a', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(['t_p', 't_off', 'vol_flow', 'p_control', 'gamma_global', 'gamma_local[0]', 'gamma_local[1]',
                     'gamma_local[2]', 'gamma_local[3]', 'gamma_local[4]', 'gamma_local[5]', 'gamma_local[6]',
                     'gamma_local[7]'])

# Initialize sensors
init.init_sensors()

# Get sensor offset for wall-shear-stress sensors
print('Offset Measurement for Wall-Shear-Stress Sensors')
functions.read_signal(init.Tau)
init.Tau.offset = np.mean(init.Tau.data_in, axis=1)
print(init.Tau.offset)

# Wait for wind tunnel to be started
input('Turn on wind tunnel. Wait for the velocity to reach 20m/s and then type any key to continue')

try:
    # Loop through parameter space
    for i in range(len(t_p)):
        for j in range(len(t_off)):
            # Display current t_p and t_off values
            print(f'\nt_p = {t_p[i]} s')
            print(f't_off = {t_off[j]} s')

            # Get global and local Gamma
            gamma_global, gamma_local, vol_flow, p_control = functions.measurement_workflow(t_p[i], t_off[j],
                                                                                            static_vol_flow)

            # Save the data
            with open(os.path.join(save_dir, file_path), 'a', newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                writer.writerow([t_p[i], t_off[j], vol_flow, p_control, gamma_global, gamma_local[0], gamma_local[1],
                                 gamma_local[2], gamma_local[3], gamma_local[4], gamma_local[5], gamma_local[6],
                                 gamma_local[7]])

    functions.measurement_stop()

except:
    functions.measurement_stop()

