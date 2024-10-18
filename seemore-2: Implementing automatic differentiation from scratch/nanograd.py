class Value:
    def __init__(self, data, deps=(), op=''):
        self.data = data
        self.grad = 0
        self._deps = set(deps)
        self._op = op
        self._backward = lambda: None

    def __repr__(self):
        test = [x.data for x in self._deps]
        return f"Value(data={self.data}" + \
            f", data_in_deps={test}, op={self._op})" if self._op else ")"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data + other.data, (self, other), '+')

        def backward():
            self.grad += result.grad
            other.grad += result.grad

        result._backward = backward
        return result
    
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result._backward = _backward
        return result
    
    def __rmul__(self, other):
        return self * other
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._deps:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for value in reversed(topo):
            value._backward()
