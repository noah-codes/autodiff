import numpy as np
import math as math

class UnaryOp:
    def __init__(self, parent):
        self.parent = parent

class BinaryOp:
    def __init__(self, left, right):
        self.left = left
        self.right = right

class Const:
    def __init__(self, value):
        self.value = value

    def eval(self):
        return self.value

    def bprop(self, op, gradient):
        return Const(0)

    def set(self, value):
        self.value = value

class AddOp(BinaryOp):
    def __init__(self, left, right):
        BinaryOp.__init__(self, left, right)

    def eval(self):
        return self.left.eval() + self.right.eval()

    def bprop(self, op, gradient):
        if op is self.left or op is self.right:
            return gradient
        else:
            return Const(0)

class MultOp(BinaryOp):
    def __init__(self, left, right):
        BinaryOp.__init__(self, left, right)

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
    def __init__(self, parent):
        UnaryOp.__init__(self, parent)

    def eval(self):
        return math.exp(self.parent.eval())

    def bprop(self, op, gradient):
        if op is self.parent:
            return MultOp(gradient, ExpOp(self.parent))
        else:
            return Const(0)

graph_children = {}
names = {}

def add_edge(source, dest):
    if not source in graph_children:
        graph_children[source] = []
    graph_children[source].append(dest)

def Add(left, right):
    op = AddOp(left, right)
    add_edge(left, op)
    add_edge(right, op)
    return op

def Mult(left, right):
    op = MultOp(left, right)
    add_edge(left, op)
    add_edge(right, op)
    return op

def Exp(parent):
    op = ExpOp(parent)
    add_edge(parent, op)
    return op

def build_gradient(op, table):
    print("Build gradient for " + names[op])
    if op in table:
        print("Gradient for " + names[op] + " was in table")
        return table[op]
    grad_list = []
    print("Children: " + str([names[x] for x in graph_children[op]]))
    for child in graph_children[op]:
        gradient = build_gradient(child, table)
        grad_piece = child.bprop(op, gradient)
        grad_list.append(grad_piece)
    grad_op = None
    if len(grad_list) == 1:
        grad_op = grad_list[0]
    elif len(grad_list) == 2:
        grad_op = AddOp(grad_list[0], grad_list[1])
    else:
        raise ValueError
    table[op] = grad_op
    return grad_op

x = Const(1)
y = Const(1)
z = Const(-3)

a = Add(x, Const(5))
b = Add(y, Const(-0.5))
c = Mult(a, b)
d = Exp(c)
e = Mult(d, d)

names[x] = "x"
names[y] = "y"
names[z] = "z"
names[a] = "a"
names[b] = "b"
names[c] = "c"
names[d] = "d"
names[e] = "e"

print(d.eval())
table = {e : Const(1)}
x_op = build_gradient(x, table)
print(x_op.eval())
#y_op = build_gradient(y, table)

"""for i in range(5000):
    if i % 10 == 0:
        print("Iteration {0}: x={1:3f}, y={2:3f}, dx={3:3f}, dy={4:3f}, f={5:3f}".
                format(i, x.eval(), y.eval(), x_op.eval(), y_op.eval(), e.eval()))
    x_grad = -x_op.eval() * 0.0001
    y_grad = -y_op.eval() * 0.0001
    x.set(x.eval() + x_grad)
    y.set(y.eval() + y_grad)"""
    
   
