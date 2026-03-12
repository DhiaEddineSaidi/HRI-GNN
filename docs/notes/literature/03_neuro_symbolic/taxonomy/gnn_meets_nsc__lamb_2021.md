---
type: conference survey paper
authors: Luís C. Lamb, Artur d’Avila Garcez, Marco Gori, Marcelo O. R. Prates, Pedro H. C. Avelar, Moshe Y. Vardi
first_published: 2021-06-12
start reading: 2026-03-20
end_reading:
status: Reading
tags:
  - GNN
  - MustRead
  - NSC
  - SURVEY
---
# Graph Neural Networks Meet Neural-Symbolic Computing: A Survey and Perspective
This paper was first introduced in the International Joint Conference on Artificial Intelligence 2020 IJCAI 2020 as a survey paper where the authors reviewed the state-of-the-art on the use of GNNs as a model of neural-symbolic computing. This survey includes the application of GNNs in several domains as well as its relationship to current developments in neural-symbolic computing.
## 1. The Claim (Pass 1)
### 1.1 Problem solved: 
Authors reviewed the state-of-the-art on the use of GNNs as a model of neural-symbolic computing.
### 1.2 Main Innovation:
Recent work have surveyed **[[gnn_foundations__scarselli_2009| Graph Neural Networks]]** and **[[neural-symbolic computing]]**, but to the best of our knowledge, no survey has reviewed and analysed the recent results on the ==specific relationship between **GNNs** and **NSC**==.
### 1.3 Why it matters for my PhD: 
This paper will help me to know the main approaches used to rich knowledge representation and reasoning within deep learning and the approach of GNN for learning and reasoning about problems that require [[relational structures]] or [[symbolic learning]]

## 2. The Evidence (Pass 2)
Based on Pr. Kautz taxonomy the paper 
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
- Henry Kautz introduced a [[the_third_ai_summer__Kautz_2020#Kautz Taxonomy for NSC neural-symbolic computing|neural-symbolic computing taxonomy]], 
- **TYPE 1 neural-symbolic system** :  GNNs, this type is a standard DL which some may argue is a stretch to refer to as neural-symbolic (But which is included here to note that the input and output of a NN can be made of symbols e.g. in the case of language translation or question answering apps) 
- GNNs were recently combined with CNN in novel ways which have produced impressive results on data efficiency in parallel 
- **TYPE 2 neural-symbolic system** : like DeepMind's [[AlphaGo]], where the **core neural network** is loosely-coupled with a **symbolic problem solver** such as [[Monte Carlo tree search]] 
- **TYPE 3 neural-symbolic system** : hybrid system whereby NN focusing on one task (e.g. Object Detection) <=interacts via input/output  =>  with a symbolic system specializing in a complementary task.
- **TYPE 4 neural-symbolic system** symbolic knowledge is compiled into the training set of a neural network using the SOP architecture but with  a special training regime based on symbolic rules
$$
\textcolor{blue}{input: }\int_{}^{} x^n \,dx \hspace{1.5cm}\textcolor{blue}{output} : (\frac{1}{n+1}) x^{n+1}
$$
 - **TYPE 5 neural-symbolic system**:  Are those tightly-coupled neural-symbolic systems where a symbolic logic rule is mapped onto a distributed representation (an embedding) and acts as a soft-constraint (a regularizer) on the network's loss function. Examples of these are :  
	-  **Logic Tensor Networks** [[Serafini and d’Avila Garcez, 2016]]
	- **Tensor Product Representations** [[Huang et al., 2017]]
	These examples shoes that NSC has focused on the learning of adequate embeddings (**vectorization**) for the purpose of symbolic computation. 
	This branch of NSC has been called  **[[neural-symbolic computing an effective methodology for principled integration of machine learning and reasoning.pdf#page=9&selection=5,0,7,13|tensorizations methods]]** includes :
	it uses [[fuzzy_logic]] methods in representing first-order logic.
- **TYPE 6 NEURO\[SYMBOLIC\]** : ![[Pasted image 20260311005115.png]]
based on Daniel Kahneman’s theory of “thinking fast and slow” which states that the brain implements two distinct mechanisms for reasoning. System 1 operates automatically and quickly, with little or no effort and no sense of voluntary control, and is based on similarity. System 2, by contrast, allocates attention to the effortful mental activities. You consciously and often painfully work your way through a tree of choices and imagined outcomes


TYPE 6 systems are considered to be systems that should be capable of combinatorial reasoning (The ability to combine known concepts into **new combinations never seen before**), suggesting using an attention schema to achieve it effectively 