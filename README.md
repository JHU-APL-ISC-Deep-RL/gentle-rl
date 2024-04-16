# Gentle RL

##  Roadmap

- ```configs``` folder holds configuration files
- ```logs``` folder will store output Tensorboard log files
- ```output``` folder will hold output .pt model files
- ```gentle``` folder contains source code

## Installation
This code has been tested with conda and multiple versions of python and MuJoCo. [Safety Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) and [DeepMind Control](https://github.com/google-deepmind/dm_control) require the newer (DeepMind) MuJoCo.  Safety Gymnasium requires python < 3.11 because of its dependency on PyGame. [Safety Gym](https://github.com/openai/safety-gym) uses mujoco-py and should be run from a different conda environment.  The on-policy code is currently only tested for Safety Gym, with the exception of the Conservative Safety Critic.

Given all of this, the recommended version of python to use is 3.9. Key installation steps:

(1) Clone this repository

(2) Create, activate conda python environment

(3) Install correct version of [PyTorch](https://pytorch.org/get-started/locally/)

(4) Install conda version of mpi4py

(5) Navigate to ```gentle``` and type ```pip install -e .```


##  Sample Configuration Files
Sample unconstrained and constrained learning configuration files are provided for the Safety Gymnasium and 
DeepMind Control environments.  More are coming.

## Running the code
On-policy code is only tested for Safety Gym (will get to gymnasium soon).  To run these approaches (PPO, TRPO, COPG), go to
`gentle/rl` and type something like

```commandline
mpiexec -n 5 python policy_optimizer.py --config ../../configs/on_policy/Unconstr/CarButton2/ppo.json
```

Note that these on-policy approaches are designed to run on multiple cores coordinated by MPI.

To run off-policy code (OPAC<sup>2</sup> works best, SAC and TD3 with resetting are usually competent), go `gentle/rl` 
and type something like

```commandline
python opac2.py --config ../../configs/safety_gymnsasium/off_policy/opac/CarGoal2/tanh_2.json
```

Note that these off-policy algorithms are designed to run single-thread. They can be run on either CPUs or GPUs.  
The convention in the off-policy configs is that the `2` in the config name indicates a higher penalty weight.

The constrained versions of both on- and off- policy methods run similarly, except from `gentle/constr`.

## Contact Info
For questions please contact [Jared.Markowitz@jhuapl.edu](mailto:Jared.Markowitz@jhuapl.edu)