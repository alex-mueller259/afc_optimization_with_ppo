import numpy as np
import time
import scipy
import measurement.init as init


# Function to generate square wave signal based off t_off, t_p and an amplitude
def generate_signal(t_off, t_p, sampling_rate, amplitude):
    # Generate time values
    t_end = t_off + t_p
    t_signal = np.linspace(0, t_end, int(sampling_rate * t_end), endpoint=False)

    # Generate square wave signal
    signal = np.zeros_like(t_signal)
    signal[t_signal >= t_off] = amplitude

    return signal


# Function to write the signal continuously
def write_continuous_signal(writer_obj):
    try:
        writer_obj.writer.write_many_sample(writer_obj.signal)
        writer_obj.task.start()

    except():
        print('Kontinuierliches Signal konnte nicht ausgegeben werden')
        writer_obj.task.close()


# Function to stop a task
def stop_task(task):
    if not task.is_task_done():
        task.stop()
        task.close()


# Function to read input signals
def read_signal(reader_obj):
    try:
        reader_obj.reader.read_many_sample(reader_obj.data_in, number_of_samples_per_channel=reader_obj.num_samples,
                                           timeout=(reader_obj.measuring_time + 5))
        reader_obj.task.start()
        reader_obj.task.wait_until_done(timeout=(reader_obj.measuring_time + 5))
        reader_obj.task.stop()
    except():
        print('Messung konnte nicht durchgeführt werden')


# Function to convert a measured voltage (V) into a volume flow (l/min)
def vol_flow_cal(volt):
    # For a volume flow sensor with the range 2-200 l/min
    volt_min = 0
    volt_max = 10
    vol_flow_min = 2
    vol_flow_max = 200

    # Get linear function
    p = np.polynomial.polynomial.polyfit((volt_min, volt_max), (vol_flow_min, vol_flow_max), 1)
    vol_flow = p[0] + volt * p[1]

    return vol_flow


# Function for the AFC/volume flow control loop
def afc_control():
    print('AFC Control Start')

    # Initial p_control
    p_control = init.Signal_Out.signal_const_value

    # Wait for volume flow to level off
    time.sleep(10)

    # Get current volume flow
    data_v = init.vol_flow.task.read(number_of_samples_per_channel=init.vol_flow.num_samples)
    v_dot_ist = np.mean(data_v)  # calc mean
    v_dot_ist = vol_flow_cal(v_dot_ist)  # convert from voltage to volume flow (l/min)

    # Display current p_control and volume flow
    print(f'\rp_control = {p_control:.3f} V;    vol_flow = {v_dot_ist:.3f} l/min', end='')

    # Set limit values for p_control
    p_control_min = 0.15
    p_control_max = 9.75

    # Set up linear function for waiting time (waiting time will get smaller if the change in p_control gets smaller)
    d_p_control_min = 0.01
    d_p_control_max = 0.5
    wait_time_max = 10
    wait_time_min = 4

    f_wait_poly_function = np.polynomial.polynomial.polyfit((d_p_control_min, d_p_control_max),
                                                            (wait_time_min, wait_time_max), 1)

    # Initialize the convergence parameters
    d_v_percent = 1 - (v_dot_ist / init.target_vol_flow)
    d_p_control = 1  # is set to a high number so that the while loop will start

    # The control loop will exit if the current volume flow is within 1% of the target volume flow AND the change of
    # p_control (voltage that is send to the pressure valve) is less than 0.005V in comparison to the previous step
    while (abs(d_v_percent) > 0.01) or (abs(d_p_control) > 0.005):
        # Store previous value for p_control
        p_control_old = p_control

        # Calculate new change in p_control based off the change in the volume flow
        d_p_control = p_control * d_v_percent  # might add a factor here

        # Make sure that the d_p_control and p_control stay in the defined boundaries
        if d_p_control > 0.25:
            p_control = p_control + 0.25
        else:
            p_control = p_control + p_control * d_v_percent
        if p_control > p_control_max:
            p_control = p_control_max
        elif p_control < p_control_min:
            p_control = p_control_min

        # Recalculate the change in p_control
        d_p_control = p_control - p_control_old
        # print('d_p_control {:.5f}'.format(d_p_control))

        # Output the new p_control
        init. Signal_Out.task.stop()
        init.Signal_Out.signal_const_value = p_control
        init.Signal_Out.init_signal()
        write_continuous_signal(init.Signal_Out)

        # Calculate the time to wait before starting the volume flow measurement
        wait_time = f_wait_poly_function[0] + abs(d_p_control) * f_wait_poly_function[1]
        # Check if the wait_time is in the defined boundaries
        if wait_time > wait_time_max:
            wait_time = wait_time_max
        elif wait_time < wait_time_min:
            wait_time = wait_time_min

        # Wait
        time.sleep(wait_time)

        # Get current volume flow
        data_v = init.vol_flow.task.read(number_of_samples_per_channel=init.vol_flow.num_samples)
        v_dot_ist = np.mean(data_v)  # calc mean
        v_dot_ist = vol_flow_cal(v_dot_ist)  # convert from voltage to volume flow (l/min)

        # Display current p_control and volume flow
        print(f'\rp_control = {p_control:.3f} V;    vol_flow = {v_dot_ist:.3f} l/min', end='')

        # Calculate the delta (in %) between the current and target volume flow
        d_v_percent = 1 - (v_dot_ist / init.target_vol_flow)

    print('\nAFC Control Finished')


# Function to calculate both the local and global forward flow fraction
def calc_gamma(data):
    [m, n] = np.shape(data)  # m=num_channels, n=num_samples

    # Local forward flow fraction (= number of values > 0 / number of overall values)
    gamma_local = np.zeros(m)
    for i in range(m):
        gamma_local[i] = (data[i, :] > 0).sum() / n

    # Global forward flow fraction

    # x-vectors of sensor position in (m)
    x1 = np.arange(0.082, (0.082 + 0.04 * 6), 0.04)  # for 6 sensors on the upper plate
    # x2 = np.arange(0.386, (0.386 + 0.042 * 3), 0.042)  # for 3 sensors on the lower plate
    x2 = np.array([0.386, 0.386 + 0.042 * 2])  # for 2 sensors on the lower plate
    x = np.concatenate((x1, x2))

    # Distance between first and last wall shear stress sensor
    dx = x[-1] - x[0]

    # Integrate over the local gammas
    gamma_global = 1 / dx * scipy.integrate.trapezoid(y=gamma_local, x=x)

    return gamma_global, gamma_local


# Function for the overall measurement workflow
def measurement_workflow(t_p, t_off, static_vol_flow):
    # Apply new values for t_p and t_off
    init.Signal_Out.signal_sw = generate_signal(t_off, t_p, init.Signal_Out.sampling_rate, amplitude=5)
    init.Signal_Out.init_signal()

    # Start AFC signal
    if not init.Signal_Out.task.is_task_done():
        init.Signal_Out.task.stop()
    write_continuous_signal(init.Signal_Out)

    # Set the target volume flow if selected
    # Otherwise the initial p_control will be kept and the volume flow will change depending on t_p and t_off
    if static_vol_flow:
        # AFC controller (only do that if the AFC is actually on, so t_p != 0)
        if t_p != 0:
            afc_control()  # Comment out this line to disable the AFC controller !!!
    else:
        # Wait
        time.sleep(10)

    # Get current volume flow (to save as additional data)
    data_v = init.vol_flow.task.read(number_of_samples_per_channel=init.vol_flow.num_samples)
    v_dot_ist = np.mean(data_v)  # calc mean
    v_dot_ist = vol_flow_cal(v_dot_ist)  # convert from voltage to volume flow (l/min)

    # Wall-Shear-Stress measurement
    print('Wall-Shear-Stress Measurement Start')
    read_signal(init.Tau)
    print('\rWall-Shear-Stress Measurement Finished', end='')
    # Apply Offset to measured data
    data_off = np.transpose(np.transpose(init.Tau.data_in) - init.Tau.offset)
    # print(np.mean(init.Tau.data_in, axis=1))

    # Calculate Gamma
    print('\rCalculate Gamma', end='')
    gamma_global, gamma_local = calc_gamma(data_off)

    return gamma_global, gamma_local, v_dot_ist, init.Signal_Out.signal_const_value


def measurement_stop():
    # Set Square Signal to zero and stop the task
    init.Signal_Out.task.stop()
    init.Signal_Out.signal_sw = np.zeros(10)
    init.Signal_Out.signal_const_value = 0
    init.Signal_Out.init_signal()
    write_continuous_signal(init.Signal_Out)
    time.sleep(0.5)
    init.Signal_Out.task.stop()
    init.Signal_Out.task.close()

    # Stop all other tasks
    for sensor_obj in [init.vol_flow, init.Tau]:
        stop_task(sensor_obj.task)

    print('\nProgram stopped')
