from gradflow.tensor import *
import tensorflow as tf

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

def test_forward_indexing():
    L = [[1, 2, 3], [4, 5, 6], [7 , 8, 9]]
    A = np.array(L)
    B = A[:2, :2]
    C = Tensor(L)
    D = C[:2, :2]
    assert np.all(B==D.numpy())
    A[1:, 1:] = B
    C[1:, 1:] = D
    assert np.all(A==C.numpy())

def test_backward_add():
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[4.0, 5.0], [3.0, 1.0]]
    a = Tensor(A, "A")
    b = Tensor(B, "B")
    x = Tensor(1.0, "X")
    c = a + b
    c += c + 7
    c += x
    c.backward()
    da, db, dx = a.grad(), b.grad(), x.grad()
    assert np.all(da == db)

    a = tf.Variable(A)
    b = tf.Variable(B)
    x = tf.Variable(1.0)
    with tf.GradientTape() as tape:
        c = a + b
        c += c + 7
        c += x
    grads = tape.gradient(c, [a,b,x])
    da_tf = grads[0].numpy()
    db_tf = grads[1].numpy()
    dx_tf = grads[2].numpy()

    assert np.all(da == da_tf)
    assert np.all(db == db_tf)
    assert np.all(dx == dx_tf)

def test_backward_mult():
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[4.0, 5.0], [3.0, 1.0]]
    a = Tensor(A, "A")
    b = Tensor(B, "B")
    x = Tensor(1.0, "X")
    c = a * b
    c *= c * 7
    c *= x
    c.backward()
    da, db, dx = a.grad(), b.grad(), x.grad()
    C = c.numpy()

    a = tf.Variable(A)
    b = tf.Variable(B)
    x = tf.Variable(1.0)
    with tf.GradientTape() as tape:
        c = a * b
        c *= c * 7
        c *= x
    grads = tape.gradient(c, [a,b,x])
    da_tf = grads[0].numpy()
    db_tf = grads[1].numpy()
    dx_tf = grads[2].numpy()

    assert np.all(C == c.numpy())

    print(da,'\n----------\n',da_tf)
    assert np.all(da == da_tf)
    assert np.all(db == db_tf)
    assert np.all(dx == dx_tf)

def test_pow_backward():
    A = [[1.0, 2.0], [3.0, 4.0]]
    a = Tensor(A, "A")
    z = a ** 3.3
    z.backward()
    da = a.grad()

    a = tf.Variable(A)
    with tf.GradientTape() as tape:
        z = a ** 3.3
    da_tf = tape.gradient(z, a).numpy()

    print(da,'\n----------\n',da_tf)
    assert abs(np.var(da - da_tf)) < 1e-8

def test_matmul_back():
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[4.0, 5.0], [3.0, 1.0]]
    a = Tensor(A, "A")
    b = Tensor(B, "B")
    M = a @ b
    M.backward()
    da, db = a.grad(), b.grad()

    a = tf.Variable(A)
    b = tf.Variable(B)
    with tf.GradientTape() as tape:
        M = a @ b
    grads = tape.gradient(M, [a, b])
    da_tf = grads[0].numpy()
    db_tf = grads[1].numpy()
    assert np.all(da == da_tf)
    assert np.all(db == db_tf)

def test_backward_indexing():
    L = [[1, 2, 3], [4, 5, 6], [7 , 8, 9]]
    A1 = Tensor(L)
    A1 += 7
    A2 = Tensor(L)
    a = A2[0] + 7
    b = A2[1] + 7
    c = A2[2] + 7
    B = Tensor(np.zeros((3, 3), dtype=float))
    B[0] = a
    B[1] = b
    B[2] = c
    matrix = np.array(L) + 1
    C1 = A1 @ matrix
    C2 = B @ matrix
    C1.backward()
    C2.backward()
    d1 = A1.grad()
    d2 = A2.grad()
    assert np.all(C1.numpy() == C2.numpy())
    assert np.all(d1 == d2)

def test_ops1():
    '''
    Test operations: reduce_sum, reduce_mean, exp, log
    '''
    L = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0 , 8.0, 9.0]]
    A = Tensor(L)
    A1 = A.log()
    A2 = A1 - 1
    A3 = A2.exp()
    b = A3.reduce_sum()
    c = A3.reduce_mean()
    d = b - c
    d.backward()
    dA = A.grad()
    dA1 = A1.grad()
    dA2 = A2.grad()
    dA3 = A3.grad()
    db = b.grad()
    dc = c.grad()

    A_tf = tf.Variable(L)
    with tf.GradientTape() as tape:
        A1_tf = tf.math.log(A_tf)
        A2_tf = A1_tf - 1
        A3_tf = tf.math.exp(A2_tf)
        b_tf = tf.reduce_sum(A3_tf)
        c_tf = tf.reduce_mean(A3_tf)
        d_tf = b_tf - c_tf
    grads = tape.gradient(d_tf, [A_tf, A1_tf, A2_tf, A3_tf, b_tf, c_tf])
    dA_tf = grads[0].numpy()
    dA1_tf = grads[1].numpy()
    dA2_tf = grads[2].numpy()
    dA3_tf = grads[3].numpy()
    db_tf = grads[4].numpy()
    dc_tf = grads[5].numpy()

    def comp(x, y):
        print(x,'\n<------->\n',y)
        assert abs(np.var(x - y)) < 1e-8 

    comp(A.numpy(), A_tf.numpy())
    comp(A1.numpy(), A1_tf.numpy())
    comp(A2.numpy(), A2_tf.numpy())
    comp(A3.numpy(), A3_tf.numpy())
    comp(b.numpy(), b_tf.numpy())
    comp(c.numpy(), c_tf.numpy())
    comp(d.numpy(), d_tf.numpy())
    
    comp(dc, dc_tf)    
    comp(db, db_tf)
    comp(dA3, dA3_tf)
    comp(dA2, dA2_tf)
    comp(dA1, dA1_tf)
    comp(dA, dA_tf)
