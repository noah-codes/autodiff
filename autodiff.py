import numpy as np
import math as math

class Const:
    def __init__(self, value, name=""):
        self.value = value
        self.name = name

    def eval(self, inputs):
        return self.value

    def bprop(self, inputs, op, gradient):
        return Const(0)
        
class Var:
    def __init__(self, value, name=""):
       self.value = value
       self.name = name
    
    def eval(self, inputs):
        assert len(inputs) == 0
        return self.value

    def bprop(self, inputs, op, gradient):
        assert len(inputs) == 0
        return Const(int(op is self))
      
    def set(self, value):
        self.value = value

class AddOp:
    def __init__(self, name=""):
        self.name = name

    def eval(self, inputs):
        assert len(inputs) == 2
        return np.add(inputs[0].eval(), inputs[1].eval())

    def bprop(self, inputs, op, gradient):
        assert len(inputs) == 2
        if op in inputs:
            return gradient
        else:
            return Const(0)
            
class SumOp:
    def __init__(self, name=""):
        self.name = name
    
    def eval(self, inputs):
      return np.sum([op.eval() for op in inputs])
      
    def bprop(self, inputs, op, gradient):
      if op in inputs
        return gradient
      else:
        return Const(0)
        
class MultOp:
    def __init__(self, name="", transpose_a=False, transpose_b=False):
        self.name = name

    def eval(self, inputs):
        assert len(inputs) == 2
        return np.dot(inputs[0], inputs[1])

    def bprop(self, inputs, op, gradient):
        assert len(inputs) == 2
        if op is inputs[0]:
            return MultOp(gradient, inputs[1], transpose_b=True)
        elif op is self.right:
            return MultOp(inputs[0], gradient, transpose_a=True)
        else:
            return Const(0)

graph_children = {}
graph_parents = {}

def add_edge(source, dest):
    if not source in graph_children:
        graph_children[source] = []
    if not dest in graph_parents:
        graph_parents[dest] = []
    graph_children[source].append(dest)
    graph_parents[dest].append(source)

def Add(left, right, name=""):
    op = AddOp(left, right, name)
    add_edge(left, op)
    add_edge(right, op)
    return op

def Mult(left, right, name=""):
    op = MultOp(left, right, name)
    add_edge(left, op)
    add_edge(right, op)
    return op

# compute dz/dx (x is op)
def build_gradient(op, grad_table):
    if op in grad_table:
        return grad_table[op]
    grad_list = []
    # child is y
    for child in graph_children[op]:
        # get dz/dy
        gradient = build_gradient(child, table)
        # dz/dx = dy/dx * dz/dy
        grad_piece = child.bprop(graph_parents[child], op, gradient)
        # if z = z(y, a, b, etc.)
        grad_list.append(grad_piece)
    # dz/dx = dy/dx * dz/dy + da/dx * dz/da + ...
    grad_op = SumOp(grad_list, "sum")
    table[op] = grad_op
    return grad_op

x = Var(1, "x")
y = Var(1, "y")
z = Var(-3, "z")

x2 = Mult(x, x, "x2")
a = Add(x2, Const(5), "a")
b = Add(y, Const(-0.5), "b")
c = Mult(a, b, "c")
d = Exp(c, "d")
e = Mult(d, d, "e")
f = Mult(Const(-1), e, "f")

table = {f : Const(1)}
x_op = build_gradient(x, table)
y_op = build_gradient(y, table)

for i in range(1000):
    if i % 1 == 0:
       # print(x_op_2.eval())
        print("Iteration {0}: x={1:3f}, y={2:3f}, dx={3:3f}, dy={4:3f}, f={5:3f}".
                format(i, x.eval(), y.eval(), x_op.eval(), y_op.eval(), e.eval()))
    x_grad = -x_op.eval() * 0.0001
    y_grad = -y_op.eval() * 0.0001
    x.set(x.eval() + x_grad)
    y.set(y.eval() + y_grad)
    
   
