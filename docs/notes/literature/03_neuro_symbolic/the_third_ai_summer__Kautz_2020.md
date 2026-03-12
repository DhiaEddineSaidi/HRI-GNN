---
type: Journal article
authors: Henry A. Kautz
first_published: 2022-03-31
start_reading: 2026-03-20
end_reading:
status: Reading
tags:
  - GNN
  - MustRead
  - NSC
  - SURVEY
---
# The third AI summer: AAAI Robert S. Engelmore Memorial Lecture
This article summarizes the author's Robert S. Englemore Memorial Lecture presented at the Thirty-Fourth AAAI Conference on Artificial Intelligence on February 10, 2020. It explores recurring themes in the history of AI, real and imagined dangers from AI, and the future of the field.
## 1. The Claim (Pass 1)
### 1.1 Problem solved:
Professor Kautz talked about history, future and risks of AI. The problem here is the chronological state-of-the-art of AI and Pr. Kautz somehow solved this problem by regrouping and highlighting main salient aspects of each part. 
### 1.2 Main Innovation:
The main innovation brought by Pr. Kautz is the ==[[#Kautz Taxonomy for NSC neural-symbolic computing|taxonomy]] of the neuro-symbolic reasoning==.
### 1.3 Why it matters for my PhD: 
This article is mentioned in the [[gnn_meets_nsc__lamb_2021| GNNs meets NSC survey]]  which i'm asked to read and summarize. It details in the **PART III : Future of AI** the taxonomy of the Neuro-Symbolic Reasoning, which i'm interested in to locate the GNN in one of the 6 types mentioned. The GNN is the first type.

## 2. The Evidence (Pass 2)
Professor Kautz 

## 3. The Methodology (Pass 3)
After a brief introduction in which Professor Kautz introduced the three major AI subdomain they are in conflict Symbolic AI neural networks and deep learning  , the author divide his article to three main parts :
- **Part I**
	- The first AI summer  : Irrational exuberance, 1948-1966 
		- Artificial neural networks 
		- knowledge representation
		- Heuristic search
	-  The First AI winter : crushed dreams, 1667-1977
	- Second summer : Knowledge is power, 19\[67\]8-1987
	- The second ai winter : The disrespected science, 1988-2011
		- Probabilistic reasoning 
		- Machine learning
	- The Third summer : Deep learning 201\[26\]-?
		- Hierarchical representation learning
		- similarity
	- Why winter might not return
- **Part II**
	- AI for bad
		- keeping your (face | mind) private
		- AI-created fake (news | friends)
		- Weapons of (environmental) mass destruction
- **Part III : Future of AI**
- **Summary : The State of AI**
## Notes & Thoughts for `stat_of_the_art.tex`
- The core limitation they address is the loss of structured/relational information in traditional models. This is highly relevant when justifying *why* we don't just use a simple CNN to solve Raven puzzles.
- *Insert more thoughts here while reading...*

## Important ideas in the paper : 

### Kautz Taxonomy for NSC neural-symbolic computing
- Henry Kautz introduced a 6 type taxonomy for NSC  :

	- **Type 1; SYMBOLIC NEURO SYMBOLIC :** Currently the deep learning SOP (Standard Operating Procedure) for NLP (Natural Language Processing). 
	 Symbolic input (in the case of language: sequences of word) are each converted to vectors by **[[word2vec]], GloVe**, or similar and passed to a neural network. The network's output units convert the previous layer to a symbolic category or sequence of symbols via **[[softmax]]** operation
	 ![[Pasted image 20260309024517.png|641]]
	
	-  **Type 2; SYMBOLIC\[NEURO\]**: this architecture employs a Neural pattern recognition subroutine within a symbolic problem solver. AlphaGo is a prototypical example of this design. The problem solver is the [[monte-carlo_tree_search_algorithm]]  and its heuristic evaluation function is a neural network? Most robots and autonomous vehicles are symbolic\[Neuro\]                                 
	![[Pasted image 20260309032342.png|294]]
	
	- **Type 3; NEURO;SYMBOLIC:**
	-  **Type 4; NEURO:SYMBOLIC:**
	-  **Type 5; NEUROsymbolic:**
	-  **Type 6; NEURO \[SYMBOLIC \]**

### **1. Logic Tensor Networks (LTN) [Serafini and d'Avila Garcez, 2016]**
**What it does:**
- LTN takes **First-Order Logic formulas** (like ∀x, ∃y, etc.) and gives them a **real-valued truth value** (between 0 and 1) using **fuzzy logic semantics**.
- It then uses **neural networks** to learn the best interpretation of the predicates.

**Simple Example:**
- You give the rule: `∀x (Student(x) → Smart(x))`
- The neural network learns what `Student` and `Smart` mean by looking at data.
- A person named "Alice" might be a Student with truth 0.95 and Smart with truth 0.85.
- The system checks if the rule holds (it does, because 0.95 → 0.85 is valid in fuzzy logic).

**Why it's "fuzzy FOL":** It keeps the quantifiers and structure of FOL, but the truth values are fuzzy.

### **2. Tensor Product Representations [Huang et al., 2017]**
**What it does:**
- It uses a mathematical operation called the **"tensor product"** to combine symbolic representations (like "John" and "loves" and "Mary") into a single vector that represents the whole sentence or fact.
- This is a way to **embed symbolic structures into continuous vector spaces**.
**Connection to Fuzzy:**
- Once symbols become vectors, you can measure **similarities** between them.
- "John loves Mary" and "John likes Mary" might have similar vector representations (fuzzy similarity), even if the symbols are different.
### **3. Fuzzy Methods in FOL [Diligenti et al., 2017]**
**What they do:**
- They explicitly use **T-norms** (the mathematical operations that define fuzzy logic AND, OR, and IMPLIES) to turn logical formulas into **continuous loss functions** for neural networks
**Example:**
- Suppose you have a rule: `∀x (Cat(x) → Mammal(x))`
- You turn this into a **loss function**: `loss = 1 - (Cat(x) → Mammal(x))` for all x.
- If the network predicts a picture is a Cat (0.9) but not a Mammal (0.1), the fuzzy implication gives a low value (e.g., 0.2), so the loss is high (0.8).
- The network is trained to minimize this loss, thus learning to respect the logical rule.