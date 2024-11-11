import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.stream_writers import AnalogMultiChannelWriter
import numpy as np


# Class for a Single Channel Object with either an input or output task
class Channel:
    def __init__(self, channel, sampling_rate, measuring_time,
                 terminal_config_in=nidaqmx.constants.TerminalConfiguration.DEFAULT, min_val=None, max_val=None,
                 out=False):
        # Assign variables
        self.channel = channel
        self.sampling_rate = sampling_rate
        self.num_samples = measuring_time * self.sampling_rate

        # Set up task
        self.terminal_config = terminal_config_in
        self.task = nidaqmx.Task()
        if not out:
            self.task.ai_channels.add_ai_voltage_chan(self.channel, terminal_config=self.terminal_config,
                                                      min_val=min_val, max_val=max_val)
        else:
            self.task.ao_channels.add_ao_voltage_chan(self.channel, terminal_config=self.terminal_config)

        self.task.timing.cfg_samp_clk_timing(rate=self.sampling_rate, source='',
                                             active_edge=nidaqmx.constants.Edge.RISING,
                                             sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                                             samps_per_chan=self.num_samples)


# Class for a Multi Channel Input object with an input task for each channel and an Analog Multi Channel Reader
class MultiChannelIn:
    def __init__(self, channel_dict, num_samples, sampling_rate, measuring_time, min_val=None, max_val=None,
                 terminal_config_in=nidaqmx.constants.TerminalConfiguration.DEFAULT):
        # Assign variables
        self.num_samples = num_samples
        self.sampling_rate = sampling_rate
        self.measuring_time = measuring_time

        # Create array for data input
        self.data_in = np.empty((len(channel_dict), num_samples))

        # Define input task
        self.task = nidaqmx.Task()
        self.terminal_config = terminal_config_in
        for channel_name in channel_dict.values():
            self.task.ai_channels.add_ai_voltage_chan(channel_name, min_val=min_val, max_val=max_val,
                                                      terminal_config=self.terminal_config)
        self.task.timing.cfg_samp_clk_timing(rate=self.sampling_rate, source='',
                                             active_edge=nidaqmx.constants.Edge.RISING,
                                             sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                                             samps_per_chan=self.num_samples)

        # Define input reader
        self.reader = AnalogMultiChannelReader(self.task.in_stream)


# Class for a Double Channel Output with two output tasks and an Analog Multi Channel Writer
# This class only works if one of the channels has a continuous output signal
# This class can be rewritten to serve as a MultiChannelOutput (change the way the signals are initialized)
class DoubleChannelOut:
    def __init__(self, channel_list, sampling_rate, signal_const_value):
        # Initialize signal
        self.signal_sw = None  # vector of square wave signal
        self.signal_const_value = signal_const_value  # float value of the continuous signal
        self.signal_const = None  # vector of the continuous signal (needs have same length as square wave signal)
        self.signal = None  # matrix of the output signal (both signals put together)

        self.sampling_rate = sampling_rate

        # Define output task
        self.task = nidaqmx.Task()
        for channel_name in channel_list:
            self.task.ao_channels.add_ao_voltage_chan(channel_name)
        self.task.timing.cfg_samp_clk_timing(rate=sampling_rate,
                                             sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

        # Define output writer
        self.writer = AnalogMultiChannelWriter(self.task.out_stream, auto_start=False)

    # Function to initialize the output signal:
    # The continuous signal must have the same length as the square wave signal
    def init_signal(self):
        signal_len = len(self.signal_sw)
        self.signal_const = self.signal_const_value * np.ones(signal_len)

        self.signal = np.vstack((self.signal_sw, self.signal_const))
