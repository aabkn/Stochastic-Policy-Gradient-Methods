# Stochastic Policy Gradient Methods: Improved Sample Complexity for Fisher-non-degenerate Policies

The repo hosts the code for the experiments section in paper ["Stochastic Policy Gradient Methods: Improved Sample Complexity for Fisher-non-degenerate Policies"](https://proceedings.mlr.press/v202/fatkhullin23a/fatkhullin23a.pdf) by Ilyas Fatkhullin, Anas Barakat, Anastasia Kireeva, Niao He (2023).

This code contains implementations for N-PG-IGT, (N)-HARPG, and Vanilla-PG.

### Prerequisites

This code is based on the [garage repository](https://github.com/rlworkgroup/garage). To install the code with our algorithms implementation, navigate to the directory containing the code and install garage as an editable package:
```bash
pip install -e '.[all,dev]'
```

To run experiments on `mujoco` environments (e.g., `humanoid`, `hopper`, `halfcheetah`), you additionally need to install it. The installation guide for `mujoco` can be found in [mujoco-py repository](https://github.com/openai/mujoco-py).

### Running the Experiments

You can find the main experiment file `run_PG.py` in directory `examples_PolicyGradient`. You can specify various parameters to control the experiment settings.

 - `seed`: the random seed for reproducibility.
 - `batch_size`: the number of samples per batch at each iteration.
 - `gamma_0`: the initial stepsize.
 - `method`: optimization method (`sgd`=Vanilla SGD, `nigt`=N-PG-IGT, `nsgdm`=Normalized SGD, `nstormhess`=N-HARPG, `stormhess`=HARPG).
 - `eta_0`: the initial momentum parameter
 - `env`: the environment (`walker`, `acrobot`, `cartpole`, `halfcheetah`, `hopper`, `humanoid`, `reacher`, `swimmer`)
 - `logdir`: the path to directory for logs

#### Example Command

To run an experiment with a specific configuration, navigate to your project's root directory and use the following command:

```bash
python examples/run_experiments.py -seed 11 -batch_size 5000 -gamma_0 0.1 -method sgd -env cartpole -epochs 100```
