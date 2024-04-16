import sys
import numpy as np
from mpi4py import MPI

# MPI utilities for PyTorch
# Largely taken from Spinning Up; re-factored


def mpi_avg(comm, x):
    """  Average a value across all processes  """
    num_procs = comm.Get_size()
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    comm.Allreduce(x, buff, op=MPI.SUM)
    return buff / num_procs


def mpi_sum(comm, x):
    """  Sum a value across all processes  """
    return mpi_op(comm, x, MPI.SUM)


def mpi_op(comm, x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    comm.Allreduce(x, buff, op=op)
    return buff[0] if scalar else buff


def average_grads(comm, parameters):
    for p in parameters:
        if p.grad is None:
            print("WARNING: network parameter gradient does not exist.")
            continue
        p_grad_numpy = p.grad.cpu().numpy()
        avg_p_grad = mpi_avg(comm, p.grad.cpu())
        p_grad_numpy[:] = avg_p_grad[:]


def sync_weights(comm, parameters):
    for p in parameters:
        p_numpy = p.data.cpu().numpy()
        comm.Bcast(p_numpy, root=0)


def mpi_gather_objects(comm, x):
    """  Collect objects from across processes  """
    rank = comm.Get_rank()
    if x is not None:
        entry = [rank, x]
    else:
        entry = [rank, []]
    collection = comm.allgather(entry)
    collection.sort(key=lambda v: v[0])
    collection = [v[1] for v in collection]
    return collection


def collect_dict_of_arrays(comm, x):
    """  Collect a dictionary of numpy arrays across processes  """
    collected_dictionaries = mpi_gather_objects(comm, x)
    combined_dictionary = {}
    for k, v in collected_dictionaries[0].items():
        value_array = np.array([]).reshape((0,))
        for dictionary in collected_dictionaries:
            value_array = np.concatenate((value_array, dictionary[k]))
        combined_dictionary[k] = value_array
    return combined_dictionary


def collect_dict_of_lists(comm, x):
    """  Collect a dictionary of numpy arrays across processes  """
    collected_dictionaries = mpi_gather_objects(comm, x)
    combined_dictionary = {}
    for k, v in collected_dictionaries[0].items():
        value_list = []
        for dictionary in collected_dictionaries:
            if k in dictionary.keys():
                value_list += dictionary[k]
        combined_dictionary[k] = value_list
    return combined_dictionary


def print_now(input_string):
    """  Allows for immediate printing, in particular when running MPI"""
    print(input_string)
    sys.stdout.flush()


# Additional tools employed by Spinning Up, with comm made shown explicitly here:


def print_zero(comm, input_string):
    if comm.Get_rank() == 0:
        print(input_string)


def sync_and_detach_grads_from_subset(comm, parameters):
    """
    Computes an average gradient among all processes, but not all processes need
    to have gradient information. The gradient is not written back to the network
    but rather returned as a list of numpy arrays.
    """
    param_grads = []
    for p in parameters:
        if p is None or p.grad is None:
            avg_p_grad = mpi_avg_filter_nones(comm, None)
        else:
            avg_p_grad = mpi_avg_filter_nones(comm, p.grad)
        param_grads.append(avg_p_grad.copy())
    return param_grads


def mpi_avg_filter_nones(comm, x):
    """  Average across all processes, but skip None values  """
    if x is not None:
        entry = [np.asarray(x, dtype=np.float32)]
    else:
        entry = []
    collection = comm.allgather(entry)
    filtered_collection = []
    for c in collection:
        if len(c) > 0:
            filtered_collection.append(c[0])
    avg = np.mean(filtered_collection, axis=0)
    return avg


def mpi_sorted_gather(comm, x):
    """
    Collect a list of data across procs, indexed by process rank
    allows None values to be passed which can be filtered later
    these will show up as empty lists, while valid values will be
    length-one lists, i.e. return_value = [[1], [3], [], [1], [], ...]
    """
    rank = comm.Get_rank()
    if x is not None:
        entry = [[rank], [x]]
    else:
        entry = [[rank], []]
    collection = comm.allgather(entry)
    collection.sort(key=lambda v: v[0])
    collection = [v[1] for v in collection]
    return collection


def mpi_statistics_scalar(comm, x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.
    Args:
        comm: MPI comm object
        x: An array containing samples of the scalar to produce statistics for
        with_min_and_max (bool): If true, return min and max of x in addition to mean and std
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum(comm, [np.sum(x), len(x)])
    mean = global_sum / global_n
    global_sum_sq = mpi_sum(comm, np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # Compute global std
    if with_min_and_max:
        global_min = mpi_op(comm, np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(comm, np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std
