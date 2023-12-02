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