import nidaqmx
from measurement import Channel


def init_sensors():
    global vol_flow, Tau, Signal_Out, p, init_p_control, target_vol_flow

    # Generate object for volume flow measurement
    vol_flow = Channel.Channel(channel='Dev1/ai0', sampling_rate=1000, measuring_time=1, min_val=0.0,
                               max_val=10.0, terminal_config_in=nidaqmx.constants.TerminalConfiguration.RSE)

    # Generate object for wall-shear-stress / forward flow fraction measurement
    tau_dict = {'tau1': 'Dev1/ai1',
                'tau2': 'Dev1/ai2',
                'tau3': 'Dev1/ai3',
                'tau4': 'Dev1/ai4',
                'tau5': 'Dev1/ai5',
                'tau6': 'Dev1/ai6',
                'tau7': 'Dev1/ai7',
                'tau8': 'Dev1/ai8'
                # 'tau9': 'Dev7/ai17' # this one needs to be repaired
                }

    # Set up measurement parameters
    sampling_rate_in = 500
    measuring_time = 10
    num_samples = measuring_time * sampling_rate_in

    Tau = Channel.MultiChannelIn(tau_dict, num_samples, sampling_rate_in, measuring_time, min_val=-10.0,
                                 max_val=10.0, terminal_config_in=nidaqmx.constants.TerminalConfiguration.RSE)

    # Generate object for the output signal containing both the square wave signal for the AFC controller and the
    # voltage signal for the pressure valve
    sampling_rate_out = 1000
    init_p_control = 0.5  # Initial voltage for pressure valve 0.75V ~ 60 l/min; 1.35V ~ 80 l/min
    target_vol_flow = 60

    channel_list = ['Dev1/ao0', 'Dev1/ao1']
    Signal_Out = Channel.DoubleChannelOut(channel_list, sampling_rate_out, init_p_control)
