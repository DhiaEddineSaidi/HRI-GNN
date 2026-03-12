# 2.1 Logic Embedding Approaches

Idea:

Represent **logical formulas inside continuous vector spaces**.

Example:

human(X) → mortal(X)

becomes a differentiable constraint.

Major example:

- **Logic Tensor Networks**
    
---

# 2.2 Differentiable Logic Systems

Another approach:

Make **logic operations differentiable** so they can be optimized with gradient descent.

Examples:

- differentiable theorem proving
    
- neural logic programming
    

Examples of systems:

- **Neural Theorem Provers**
    
- **DeepProbLog**
    

Idea:

logic reasoning → differentiable computation graph

---

# 2.3 Neural Logic Programming

Neural networks learn **logical rules directly from data**.

Example:

Input facts:

parent(A,B)  
parent(B,C)

Network learns rule:

parent(X,Y) ∧ parent(Y,Z) → grandparent(X,Z)

Key paper:

- **Neural Logic Programming**
    

This combines:

- symbolic relations
    
- neural training
    

---

# 2.4 Tensorized Logic Representations

This is mentioned in the paragraph.

Idea:

Represent logic relations as **tensors**.

Example relation:

R(x,y)

represented as matrix/tensor operations.

Key idea:

logic = tensor algebra

Which allows:

- GPU acceleration
    
- neural training.
    

---

# 2.5 Graph-Based Reasoning (GNN approaches)

This is where **Graph Neural Networks** appear.

Idea:

Represent symbolic knowledge as a **graph**.

Example:

Socrates → human  
human → mortal

Graph:

Socrates --human--> node  
node --mortal--> property

A **Graph Neural Networks** performs reasoning through **message passing**.

Nodes exchange information.

Eventually the network can infer:

Socrates → mortal

So reasoning emerges from **graph propagation**.

This is a very promising path for TYPE-6 systems.