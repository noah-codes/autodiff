import numpy as np
import math as math

class UnaryOp:
    def __init__(self, parent, name=""):
        self.parent = parent
        self.name = name

class BinaryOp:
    def __init__(self, left, right, name=""):
        self.left = left
        self.right = right
        self.name = name
        
class MultiOp:
    def __init__(self, inputs, name=""):
        self.inputs = inputs
        self.name = name

class Const:
    def __init__(self, value, name=""):
        self.value = value
        self.name = name

    def eval(self):
        return self.value

    def bprop(self, op, gradient):
        return Const(0)
        
class Var:
    def __init__(self, value, name=""):
       self.value = value
       self.name = name
    
    def eval(self):
        return self.value

    def bprop(self, op, gradient):
        return Const(0)
      
    def set(self, value):
        self.value = value

class AddOp(BinaryOp):
    def __init__(self, left, right, name=""):
        BinaryOp.__init__(self, left, right, name)

    def eval(self):
        return self.left.eval() + self.right.eval()

    def bprop(self, op, gradient):
        if op is self.left or op is self.right:
            return gradient
        else:
            return Const(0)
            
class SumOp(MultiOp):
    def __init__(self, inputs, name=""):
        MultiOp.__init__(self, inputs, name)
    
    def eval(self):
      return np.sum([input.eval() for input in self.inputs])
      
    def bprop(self, op, gradient):
      if op in self.inputs:
        return gradient
      else:
        return Const(0)
        
class MultOp(BinaryOp):
    def __init__(self, left, right, name=""):
        BinaryOp.__init__(self, left, right, name)

    def eval(self):
        return self.left.eval() * self.right.eval()

    def bprop(self, op, gradient):
        if op is self.left:
            return MultOp(self.right, gradient)
        elif op is self.right:
            return MultOp(self.left, gradient)
        else:
            return Const(0)

class ExpOp(UnaryOp):
    def __init__(self, parent, name=""):
        UnaryOp.__init__(self, parent, name)

    def eval(self):
        return math.exp(self.parent.eval())

    def bprop(self, op, gradient):
        if op is self.parent:
            return MultOp(gradient, ExpOp(self.parent))
        else:
            return Const(0)

graph_children = {}

def add_edge(source, dest):
    if not source in graph_children:
        graph_children[source] = []
    graph_children[source].append(dest)

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

def Exp(parent, name=""):
    op = ExpOp(parent, name)
    add_edge(parent, op)
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
        grad_piece = child.bprop(op, gradient)
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
    
   
