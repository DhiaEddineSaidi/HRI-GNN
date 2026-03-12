---
tags:
  - GNN
  - Foundations
  - MustRead
  - "#INNOVATION"
date: 2026-02-28
status: Reading
---

# The Graph Neural Network Model
**Authors:** Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, Gabriele Monfardini
**Year:** 2009
**Journal:** IEEE Transactions on Neural Networks

## 1. The Claim (Pass 1)
- **Problem solved:** Traditional machine learning approaches usually require graph data to be "squashed" (flattened into vectors) before processing, dropping crucial topological information. Existing recursive neural networks (RNNs) only worked on directed acyclic graphs.
- **Main Innovation:** They introduce the "Graph Neural Network" (GNN), which extends neural networks to directly process most types of graphs (acyclic, cyclic, directed, undirected). They use an information diffusion mechanism to reach an equilibrium state.
- **Why it matters for my PhD:** This is the absolute foundation of my Year 1 baseline. Before I can inject Neuro-Symbolic logic, I need to understand how standard GNNs exchange messages (Message Passing) between nodes (e.g., the panels of a Raven matrix) to create representations.

## 2. The Evidence (Pass 2)
*(To be completed by DhiaEddine during detailed reading)*
- What do their architectures look like (e.g., the encoding network)?
- How do they prove the convergence of the transition function?

## 3. The Methodology (Pass 3)
*(To be completed by DhiaEddine during detailed reading)*
- **Transition Function ($f_w$):** How does a node update its state based on its neighbors?
- **Output Function ($g_w$):** How does the node produce the final prediction?
- **Backpropagation:** How does the gradient descent work here using the Almeida-Pineda algorithm?

## Notes & Thoughts for `stat_of_the_art.tex`
- The core limitation they address is the loss of structured/relational information in traditional models. This is highly relevant when justifying *why* we don't just use a simple CNN to solve Raven puzzles.
- *Insert more thoughts here while reading...*


## Important ideas in the paper : 
- Applications to a graphical domain can generally be divided into two broad classes : 
	- Called graph-focused 
	- Node-focused applications
[[the_third_ai_summer__Kautz_2020]]