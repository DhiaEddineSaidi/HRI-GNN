# Monte-Carlo Tree Search (MCTS) and Neural Networks in AlphaGo

## Overview

**Monte-Carlo Tree Search (MCTS)** is a search algorithm used for decision-making in large state spaces such as board games.

It was a core component of **AlphaGo**, where it was combined with neural networks to efficiently evaluate board states and guide search.

Instead of exploring all possible moves, MCTS:

- explores promising moves
    
- simulates games
    
- builds statistics about outcomes
    

This makes it suitable for games with extremely large search spaces such as Go.

---

# The Challenge: Search in Go

The board game **Go** has an enormous search space:

```
≈ 10^170 possible board configurations
```

Classical search algorithms (like minimax) become infeasible.

MCTS solves this by exploring **only a fraction of the game tree**, focusing on promising areas.

---

# The Four Steps of Monte-Carlo Tree Search

Each iteration of MCTS follows four phases.

## 1. Selection

Starting from the root (current state), the algorithm selects a path in the tree according to a rule that balances:

- exploration (trying new moves)
    
- exploitation (choosing moves known to be good)
    

A common formula used:

```
UCB / PUCT
```

This determines which node to explore next.

---

## 2. Expansion

When a previously unexplored state is reached:

- a new node is added to the search tree
    
- the tree grows incrementally.
    

---

## 3. Simulation (Rollout)

From the new state, the algorithm simulates a game until the end.

Traditionally:

```
play random moves until win or loss
```

Result:

```
win = 1
loss = 0
```

---

## 4. Backpropagation

The outcome of the simulation is propagated back through the tree.

Each node updates statistics:

- visit count
    
- estimated win probability
    

Over thousands of simulations, the best move becomes clear.

---

# Limitations of Pure MCTS

Random simulations are inefficient in complex games.

Problems:

- many simulations explore bad moves
    
- rollouts are slow
    
- the search tree grows inefficiently.
    

AlphaGo solved this by combining MCTS with neural networks.

---

# Neural Networks in AlphaGo

AlphaGo introduced two neural networks.

## Policy Network

Predicts promising moves.

```
P(move | board_state)
```

Example output:

|Move|Probability|
|---|---|
|A|0.40|
|B|0.25|
|C|0.20|

Instead of exploring every move, MCTS focuses on high-probability ones.

This reduces the branching factor.

---

## Value Network

Estimates the probability of winning from a given state.

```
V(board_state) = probability of winning
```

Example:

```
V(state) = 0.72
```

Meaning a **72% chance of winning** from this position.

This replaces long random simulations.

---

# Loose Coupling Between MCTS and Neural Networks

The neural networks **do not replace the search**.

Instead they **guide it**.

MCTS remains the main decision process.

Neural networks provide heuristics.

---

## Where Neural Networks Are Used

### Expansion

The policy network proposes promising moves.

Instead of expanding all children:

```
expand only high-probability moves
```

---

### Simulation

Instead of random rollouts:

```
value = neural_network(board_state)
```

This directly estimates the outcome.

---

### Selection

The policy network provides a **prior probability** used in the search formula.

Example:

```
PUCT = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

Where:

- Q(s,a) = average value of the move
    
- P(s,a) = policy probability
    
- N(s) = number of visits
    

---

# Intuition

The system combines:

- **neural intuition**
    
- **symbolic search planning**
    

Analogy with humans:

|Human Player|AlphaGo|
|---|---|
|intuition|neural network|
|calculation|MCTS|

---

# Key Idea

Neural networks provide **evaluation and move suggestions**.

MCTS provides **structured planning and search**.

The combination makes the system dramatically stronger than either component alone.