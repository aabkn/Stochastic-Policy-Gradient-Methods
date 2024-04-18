"""Utilities for ensuring that experiments are deterministic."""
import random
import sys
import warnings

import numpy as np

seed_ = None
seed_stream_ = None


def set_seed(seed, torch_seed=None):
    """Set the process-wide random seed.

    Args:
        seed (int): A positive integer

    """
    seed %= 4294967294
    # pylint: disable=global-statement
    global seed_
    global seed_stream_
    seed_ = seed
    random.seed(seed)
    np.random.seed(seed)
    if 'tensorflow' in sys.modules:
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        tf.compat.v1.set_random_seed(seed)
        try:
            # pylint: disable=import-outside-toplevel
            import tensorflow_probability as tfp
            seed_stream_ = tfp.util.SeedStream(seed_, salt='garage')
        except ImportError:
            pass
    if 'torch' in sys.modules:
        warnings.warn(
            'Enabeling deterministic mode in PyTorch can have a performance '
            'impact when using GPU.')
        import torch  # pylint: disable=import-outside-toplevel
        if torch_seed is None:
            torch_seed = seed
        torch.manual_seed(torch_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_seed():
    """Get the process-wide random seed.

    Returns:
        int: The process-wide random seed

    """
    return seed_


def get_tf_seed_stream():
    """Get the pseudo-random number generator (PRNG) for TensorFlow ops.

    Returns:
        int: A seed generated by a PRNG with fixed global seed.

    """
    if seed_stream_ is None:
        set_seed(0)
    return seed_stream_() % 4294967294