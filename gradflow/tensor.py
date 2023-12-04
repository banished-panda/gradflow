import numpy as np

_count = -1
def counter() -> int:
    global _count
    _count += 1
    return _count

class Tensor:

    def __init__(self, data, name=None) -> None:
        count = counter()
        self._data = np.array(data)
        self._name = f'tensor{count}' if name == None else name
        self._operands = set()  # list of tensor that are used to calcualte this tensor
        self._next = set()      # list of tensor that use this tensor for calculations
        self._grad = np.zeros_like(self._data, dtype=float)
        self._flowgrad = lambda : None  # function to flow grad to operands
    
    def __repr__(self) -> str:
        return f"Tensor('{self._name}',{self._data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self._data + other._data)
        result._operands = set((self, other))
        self._next.add(result)
        other._next.add(result)

        def flowgrad():
            grad = result._grad
            self._grad += Tensor._reverse_brodcast_grads(grad, self._grad.ndim)
            other._grad += Tensor._reverse_brodcast_grads(grad, other._grad.ndim)
        result._flowgrad = flowgrad

        return result
    
    def __radd__(self, other):
        return other + self
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self._data * other._data)
        result._operands = set((self, other))
        self._next.add(result)
        other._next.add(result)

        def flowgrad():
            grad_self = other._data * result._grad
            grad_other = self._data * result._grad
            self._grad += Tensor._reverse_brodcast_grads(grad_self, self._grad.ndim)
            other._grad += Tensor._reverse_brodcast_grads(grad_other, other._grad.ndim)
        result._flowgrad = flowgrad

        return result
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Support powers of int/float only"
        result = Tensor(self._data ** other)
        result._operands = set((self,))
        self._next.add(result)

        def flowgrad():
            local_grad = other * self._data**(other-1) 
            grad = result._grad * local_grad
            self._grad += Tensor._reverse_brodcast_grads(grad, self._grad.ndim)
        result._flowgrad = flowgrad

        return result
    
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other
    
    def __truediv__(self, other):
        return self * other**-1.0
    
    def __rtruediv__(self, other):
        return other * self**-1.0
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self._data @ other._data)
        result._operands = set((self, other))
        self._next.add(result)
        other._next.add(result)

        def flowgrad():
            self._grad += result._grad @ np.transpose(other._data)
            other._grad += np.transpose(self._data) @ result._grad
        result._flowgrad = flowgrad
        return result
    
    def __getitem__(self, index):
        result = Tensor(self._data[index])
        result._operands = set((self, ))
        self._next.add(result)

        def flowgrad():
            self._grad[index] += result._grad
        result._flowgrad = flowgrad

        return result
    
    def __setitem__(self, key, new_value):
        new_value = new_value if isinstance(new_value, Tensor) else Tensor(new_value)
        # Prevent formation of Cyclic graphs
        # ==> create a separate Tensor for old version of self
        old_self = Tensor(self._data)
        old_self._name += (':'+self._name+'OLD')
        old_self._flowgrad = self._flowgrad
        old_self._operands = self._operands
        old_self._next = self._next.copy()
        for v in old_self._next:
            v._operands.remove(self)
            v._operands.add(old_self)
        # completely reform self
        self._data[key] = new_value._data
        self._operands=set((old_self, new_value))
        self._next = set()
        old_self._next.add(self)
        new_value._next.add(self)

        def flowgrad():
            new_value._grad += self._grad[key]
            tmp = self._grad.copy()
            tmp[key] = 0
            old_self._grad += tmp
        self._flowgrad = flowgrad
    
    def reduce_sum(self):
        result = Tensor(self._data.sum())
        result._operands = set((self, ))
        self._next.add(result)

        def flowgrad():
            self._grad += np.ones_like(self.grad) * result._grad
        result._flowgrad = flowgrad

        return result
    
    def reduce_mean(self):
        sum = self.reduce_sum()
        return sum / self._data.size
    
    def exp(self):
        result = Tensor(np.exp(self._data))
        result._operands = set((self, ))
        self._next.add(result)

        def flowgrad():
            self._grad += result._data * result._grad
        result._flowgrad = flowgrad

        return result
    
    def log(self):
        result = Tensor(np.log(self._data))
        result._operands = set((self, ))
        self._next.add(result)

        def flowgrad():
            self._grad += (1 / self._data) * result._grad
        result._flowgrad = flowgrad

        return result
    
    def tanh(self):
        result = Tensor(np.tanh(self._data))
        result._operands = set((self, ))
        self._next.add(result)

        def flowgrad():
            local_grad = 1 - result._data ** 2
            self._grad += result._grad * local_grad
        result._flowgrad = flowgrad

        return result
    
    def relu(self):
        positive = np.array(self._data > 0, dtype=int)
        result = Tensor(self._data * positive)
        result._operands = set((self, ))
        self._next.add(result)

        def flowgrad():
            self._grad += result._grad * positive
        result._flowgrad = flowgrad

        return result
    
    def reset(self):
        '''Resets gradient and references to other Tensors'''
        self._grad = np.zeros_like(self._data, dtype=float)
        self._operands = set()
        self._next = set()
        self._flowgrad = lambda : None
    
    def numpy(self) -> np.ndarray:
        return self._data
    
    def grad(self):
        return self._grad
    
    def backward(self):
        # Perform topological sort to get linear ordering of vertices
        topo = []
        visited = set()
        def build_topo(vertex: Tensor):
            if vertex not in visited:
                visited.add(vertex)
                for operand in vertex._operands:
                    build_topo(operand)
                topo.append(vertex)
        build_topo(self)

        self._grad = np.ones_like(self._data, dtype=float)
        for v in reversed(topo):
            v._flowgrad()

    
    def _reverse_brodcast_grads(grad: np.ndarray, to_dim:int):
        assert grad.ndim >= to_dim, "This wasn't broadcast"
        while grad.ndim != to_dim:
            grad = grad.sum(axis=0)
        return grad
