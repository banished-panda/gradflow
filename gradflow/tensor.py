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
    
    def _radd_(self, other):
        return self + other
    
    def numpy(self) -> np.ndarray:
        return self._data