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
        self._operands = set()
        self._grad = np.zeros_like(self._data, dtype=float)
        self._flowgrad = lambda : None  # function to flow grad to operands
    
    def __repr__(self) -> str:
        return f"Tensor('{self._name}',{self._data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self._data + other._data)
        result._operands = set((self, other))

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
        return result
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Support powers of int/float only"
        result = Tensor(self._data ** other)
        result._operands = set((self, other))
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
        return result
    
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
    