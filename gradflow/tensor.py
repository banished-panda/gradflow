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
    
    def __repr__(self) -> str:
        return f"Tensor('{self._name}',\n{self._data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self._data + other._data)
        return result
    
    def __radd__(self, other):
        return other + self
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self._data * other._data)
        return result
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Raise to powers of int/float only"
        result = Tensor(self._data ** other)
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
        return result
    
    def numpy(self) -> np.ndarray:
        return self._data