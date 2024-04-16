import json
import torch
import numpy as np
from mpi4py import MPI
from .mpi_data_utils import mpi_gather_objects
from torch.utils.tensorboard import SummaryWriter


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Logger(object):
    """  Logs data and pushes to TensorBoard  """

    def __init__(self, output_directory, device='cpu'):
        """  Construct LoggerMPI object  """
        self.summary_writer = SummaryWriter(log_dir=output_directory)
        self.graph_logged = False
        self.device = device

    def log_scalar(self, key, value, x):
        """  Logs a scalar y value, using MPI to determine x value  """
        self.summary_writer.add_scalar(key, value, x)

    def log_mean_value(self, key, value, x):
        """  Adds the mean of a given data list to the log  """
        if len(value) > 0:
            self.summary_writer.add_scalar(key, np.mean(value), x)

    def log_config(self, config_obj):
        config_str = json.dumps(config_obj, indent=2, cls=NumpyEncoder)
        config_str = "".join("\t" + line for line in config_str.splitlines(True))
        self.summary_writer.add_text('config', config_str, global_step=0)

    def log_graph(self, observations, network):
        """  Initialize TensorBoard logging of model graph """
        if not self.graph_logged:
            input_obs = torch.from_numpy(observations).float().to(self.device)
            self.summary_writer.add_graph(network, input_obs)
        self.graph_logged = True

    def flush(self):
        self.summary_writer.flush()


class LoggerMPI(object):
    """  Logs data across multiple MPI processes and pushes to TensorBoard  """

    def __init__(self, output_directory, device='cpu'):
        """  Construct LoggerMPI object  """
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.summary_writer = SummaryWriter(log_dir=output_directory)
            self.graph_logged = False
            self.device = device

    def log_config(self, config_obj):
        if MPI.COMM_WORLD.Get_rank() == 0:
            config_str = json.dumps(config_obj, indent=2, cls=NumpyEncoder)
            config_str = "".join("\t" + line for line in config_str.splitlines(True))
            self.summary_writer.add_text('config', config_str, global_step=0)

    def log_scalar(self, key, value, x, offset):
        """  Logs a scalar y value, using MPI to determine x value  """
        xs = mpi_gather_objects(MPI.COMM_WORLD, x)
        if MPI.COMM_WORLD.Get_rank() == 0:
            offset += np.sum(xs)
            self.summary_writer.add_scalar(key, value, offset)

    def log_mean_value(self, key, value, x, offset):
        """
        Collects data lists from all processes and adds their means to the logs.  If normalize is True
        plots the mean over all training experiences
        """
        values = mpi_gather_objects(MPI.COMM_WORLD, value)
        values = self.flatten_list(values)
        xs = mpi_gather_objects(MPI.COMM_WORLD, x)
        if MPI.COMM_WORLD.Get_rank() == 0:
            offset += np.sum(xs)
            if len(values) > 0:
                if None not in values:
                    self.summary_writer.add_scalar(key, np.mean(values), offset)

    def log_graph(self, observations, network):
        """  Initialize TensorBoard logging of model graph """
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not self.graph_logged:
                input_obs = torch.from_numpy(observations).float().to(self.device)
                self.summary_writer.add_graph(network, input_obs)
            self.graph_logged = True

    def flush(self):
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.summary_writer.flush()

    @staticmethod
    def flatten_list(nested_list):
        return [item for sublist in nested_list for item in sublist]
