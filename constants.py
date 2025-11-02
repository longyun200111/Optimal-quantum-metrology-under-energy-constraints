import numpy as np
from utils import fock_dm


def get_H(dim):
    return sum(
        i * fock_dm(dim, i) for i in range(dim)
    )