from gradflow.tensor import *

import numpy as np

def test_forward_add():
    L = [[1, 2, 3], [4, 5, 6], [7 , 8, 9]]
    A = np.array(L) + 1
    C = Tensor(L) + 1
    assert np.all((A==C.numpy()))

def test_forward_multiplication():
    L = [[1, 2, 3], [4, 5, 6], [7 , 8, 9]]
    A = np.array(L) * 7
    C = Tensor(L) * 7
    assert np.all((A==C.numpy()))

    A2 = A * A
    C2 = C * C
    assert np.all((A2==C2.numpy()))

def test_forward_power():
    L = [[1, 2, 3], [4, 5, 6], [7 , 8, 9]]
    A = np.array(L) ** 3.2
    C = Tensor(L) ** 3.2
    assert np.all(A==C.numpy())

def test_negation():
    L = [[1, 2, 3], [4, 5, 6], [7 , 8, 9]]
    A = np.array(L)
    C = Tensor(L)
    A,C = -A,-C
    assert np.all((A==C.numpy()))

def test_forward_sub():
    L = [[1, 2, 3], [4, 5, 6], [7 , 8, 9]]
    A = np.array(L)
    B = A * 0.5
    C = Tensor(L)
    D = C * 0.5
    assert np.all((A-B)==(C-D).numpy())
    assert np.all((B-A)==(D-C).numpy())

def test_forward_div():
    L = [[1, 2, 3], [4, 5, 6], [7 , 8, 9]]
    A = np.array(L)
    B = A + 8.0
    A1 = B/A
    C = Tensor(L)
    D = C + 8.0
    C1 = D/C
    assert abs(np.var(A1 - C1.numpy())) < 1e-8
    assert abs(np.var((A/30.0)-(D/30.0).numpy())) < 1e-8
    assert abs(np.var(1/A1 - (1/C1).numpy())) < 1e-8

def test_forward_matmul():
    L = [[1, 2, 3], [4, 5, 6], [7 , 8, 9]]
    A = np.array(L)
    C = Tensor(L)
    res = A @ A
    C0 = C @ C
    C1 = C @ A
    # C2 = A @ C       -> not allowed as numpy doesn't know how to handle Tensor
    assert np.all(res==C0.numpy())
    assert np.all(res==C1.numpy())
