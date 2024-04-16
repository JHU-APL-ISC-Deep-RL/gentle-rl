import os
import torch
from gentle.common.buffers import OffPolicyBuffer

# CPU/GPU usage regulation.  One can assign more than one thread here, but it is probably best to use 1 in most cases.
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_buffer(config, iteration=-1):
    """  Loads replay buffer  """
    buffer = OffPolicyBuffer(capacity=config.buffer_size,
                             obs_dim=config.environment.observation_space.shape[0],
                             act_dim=config.environment.action_space.shape[0],
                             store_costs=config.store_costs)
    if iteration == -1:
        buffer.load(os.path.join(config.model_folder, 'buffer-latest.p.tar.gz'))
    else:
        buffer.load(os.path.join(config.model_folder, 'buffer-' + str(iteration) + '.p.tar.gz'))
    return buffer


def get_model_grads(model):
    grads = [param.grad.detach().flatten() for param in model.parameters() if param.grad is not None]
    return torch.cat(grads)


def get_model_params(model):
    params = [param.detach().flatten() for param in model.parameters()]
    return torch.cat(params)


def compute_model_norm(model):
    """  Computes the 2-norm of all parameters in a model  """
    params = [param.detach().flatten() for param in model.parameters()]
    return torch.cat(params).norm()


def compute_model_grad_norm(model):
    """  Computes the 2-norm of the gradient over all parameters in a model  """
    grads = get_model_grads(model)
    return grads.norm()


def compute_cos_similarity(v1, v2):
    return torch.nn.functional.cosine_similarity(v1, v2, dim=0)


def compute_update_norm(old_params, model):
    """  Compute norm of update to model  """
    new_params = [param.detach().flatten() for param in model.parameters()]
    return (torch.cat(new_params) - old_params).norm()


def compute_srank(model, buffer):
    pass
