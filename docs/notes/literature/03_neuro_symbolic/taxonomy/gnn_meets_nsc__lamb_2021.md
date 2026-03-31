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

## The Article Structure : 
### Abstract 
### 1 Introduction
### 2 Neural-symbolic Computing Taxonomy
### 3 Graph Neural Networks Meet Neural-Symbolic Computing : 
#### 3.1 Logic Tensor Networks
#### 3.2 Pointer Networks 
#### 3.3 Convolutions as Self-Attention
#### 3.4 Graph Convolutional Networks
#### 3.5 The Graph Neural Network Model
#### 3.6 Message-passing Neural Networks
#### 3.7 Graph Attention Networks
### 4 Perspectives and Applications of GNNs to Neural-Symbolic Computing
#### 4.1 Relational Learning and Reasoning
#### 4.2 Combinatorial Optimization and Constraint Satisfaction Problems
### 5 Conclusions

## Important ideas in the paper : 
### IDEA 1 : 
The origine of GNNs can be traced back to neural-symbolic computing (NSC) in that both try to enrich the vector representations in the inputs of NN, first by accepting tree structures and then graphs more generally. In parallel : 
	- **NSC : has focused on the learning of adequate embedding** for the purpose of symbolic computation. This bra,ch of NSC , which includes [[Serafini and d’Avila Garcez, 2016| Logic Tensor Networks]] , [[Huang et al., 2017|Tensor Product Representations]], has been called in [[d’Avila Garcez et al., 2019]] **tensorization methods** and draw similarities with [[Diligenti et al., 2017]] that use **[[fuzzy logic|fuzzy methods]]** in representing **[[real logic|first-order logic]]**.
	- **GNN : were recently combined with convolutional networks** [[Battaglia et al., 2018]] in novel ways which have produced impressive results on data efficiency.
### IDEA 2 : 
- Henry Kautz introduced a [[the_third_ai_summer__Kautz_2020#Kautz Taxonomy for NSC neural-symbolic computing|neural-symbolic computing taxonomy]], 
- **TYPE 1 neural-symbolic system** :  GNNs, this type is a standard DL which some may argue is a stretch to refer to as neural-symbolic (But which is included here to note that the input and output of a NN can be made of symbols e.g. in the case of language translation or question answering apps) 
- GNNs were recently combined with CNN in novel ways which have produced impressive results on data efficiency in parallel 
- **TYPE 2 neural-symbolic system** : like DeepMind's [[AlphaGo]], where the **core neural network** is loosely-coupled with a **symbolic problem solver** such as [[Monte Carlo tree search]] 
- **TYPE 3 neural-symbolic system** : hybrid system whereby NN focusing on one task (e.g. Object Detection) <=interacts via input/output  =>  with a symbolic system specializing in a complementary task.
- **TYPE 4 neural-symbolic system** symbolic knowledge is compiled into the training set of a neural network using the SOP (standard operating procedure) architecture but with  a special training regime based on symbolic rules
$$
\textcolor{blue}{input: }\int_{}^{} x^n \,dx \hspace{1.5cm}\textcolor{blue}{output} : (\frac{1}{n+1}) x^{n+1}
$$
 - **TYPE 5 neural-symbolic system**:  Are those tightly-coupled neural-symbolic systems where a symbolic logic rule is mapped onto a distributed representation (an embedding) and acts as a soft-constraint (a regularizer) on the network's loss function. Examples of these are :  
	-  **Logic Tensor Networks** [[Serafini and d’Avila Garcez, 2016]]
	- **Tensor Product Representations** [[Huang et al., 2017]]
	These examples shoes that NSC has focused on the learning of adequate embeddings (**vectorization**) for the purpose of symbolic computation. 
	This branch of NSC has been called  **[[neural-symbolic computing an effective methodology for principled integration of machine learning and reasoning.pdf#page=9&selection=5,0,7,13|tensorizations methods]]** includes :
	it uses [[fuzzy logic]] methods in representing first-order logic.
- **TYPE 6 NEURO\[SYMBOLIC\]** :  The fully integrated system ![[Pasted image 20260311005115.png]]
based on Daniel Kahneman’s theory of “thinking fast and slow” which states that the brain implements two distinct mechanisms for reasoning. System 1 operates automatically and quickly, with little or no effort and no sense of voluntary control, and is based on similarity. System 2, by contrast, allocates attention to the effortful mental activities. You consciously and often painfully work your way through a tree of choices and imagined outcomes


TYPE 6 systems are considered to be systems that should be capable of ==combinatorial reasoning (The ability to combine known concepts into **new combinations never seen before**)==, suggesting using an [[attention schema]] to achieve it effectively.

### IDEA 3 : 
In machine learning the priors or inductive biases (the set of assumptions that a learner uses to compute predictions on test data). In the context of DL (Deep learning) the design of neural building blocks.
#### Feed-forward layers :
- Encourage the learner to combine features additively  
- example : 
$$ 
\begin{align}
X_i &= \{x_1 , x_2, ...,x_n\}  \hspace{1cm}\textcolor{blue}{features = \{price, age, pixels ....\}}  \\ 

Y &= w_1 * x_1 + w_2 * x_2 + ... + w_n*x_n
\end{align}


$$
The model learn how every feature contribute.

#### Dropout :
- Discourage the learner to overfit 
#### Multi-task learning 
- Encourage the learner to prefer sets of parameters that explain more than one task 
#### Convolutional architecture (CNN) :
- Successful for tasks defined over Euclidian signals
- Enforce **equivariance to spacial translation**
#### Recurrent Layers (RNN) :
- Over sequential data
- Enforce **equivariance in time** 
#### Attention mechanisms through Transformer Networks : 
- Sequential tasks like NLP (Natural Language Processing) & Symbolic Reasoning (such as solving math equations and integrals )
- encourage the learner to **combine representations additively**
	- example
	$$ 
\begin{aligned}
NLP :
phrase &: \textcolor{blue}{\text{"The bank is safe"}} \\ 
Embedding \Rightarrow& :  
\left\{
\begin{array}{ll}
\textcolor{blue}{bank} \rightarrow V_1 \\
\textcolor{blue}{safe} \rightarrow V_2
\end{array}
\right. \hspace{1cm}V_1\hspace{0.1cm}and\hspace{0.1cm}V_2\hspace{0.1cm}\text{are }\textcolor{red}{Representations} \\
\text{Attention combines the vectors like} &: W_1* V_1 + W_2*V_2 \text{ (context representation)}
\end{aligned}
$$
- enforce **permutation invariance**