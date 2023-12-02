from gradflow.tensor import *

import numpy as np

def test_forward_add():
    L = [[1, 2, 3], [4, 5, 6], [7 , 8, 9]]
    A = np.array(L) + 1
    C = Tensor(L) + 1
    assert np.all((A==C.numpy()))