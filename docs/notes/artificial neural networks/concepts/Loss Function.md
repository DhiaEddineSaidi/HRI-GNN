A loss function (or cost function) in machine learning measures the discrepancy between a model's predicted output and the actual target value, acting as a guide to evaluate performance. It computes a scalar value representing the error, which optimization algorithms like gradient descent minimize to update model parameters and improve accuracy.

## Types of Loss Functions

| Loss Function                        | mathematical equation                                                                                    | classification | regression |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------- | -------------- | ---------- |
| Mean Square Error (MSE) / L2 Loss    | $$MSE=\frac{1}{N}.\sum_{i=1}^{N}(y_i-\hat{y}_i)^2$$                                                      | ✖️             | ✔️         |
| Mean Absolute Error (MAE) / L1 Loss  | $$MAE=\frac{1}{N}.\sum_{i=1}^{N}\mid y_i-\hat{y}_i\mid ^2$$                                              | ✖️             | ✔️         |
| Binary Cross-Entropy Loss / Log Loss | $$\mathcal{L}(y, f(x)) = -\frac{1}{N}\sum_{i=1}^{N}(y_i.log(\hat{y}_i) + (1 - y_i).log(1 - \hat{y}_i))$$ | ✔️             | ✖️         |

![[Pasted image 20260328013939.png]]

![[Pasted image 20260328014052.png]]
That's why Log Loss or cross entropy is suited for classification 

### Log Loss Function :
in classification context and using a sigmoid activation function the output of the activation function follows Bernoulli's law : 
![[Pasted image 20260328035300.png]]

Likelihood 
![[Pasted image 20260328041906.png]]

![[Pasted image 20260328041956.png]]

$$
\begin{aligned}
\text{For inputs } \forall i \in [1,n], n\in \mathbb{N} \text{ vector }X^i \in \mathbb{R}^d , X^i = \begin{bmatrix} x_1 \\ \vdots \\ x_n\end{bmatrix}\\ \\
\text{ The perceptron or the node } Z^i \text{ with inputs } X^k \text{ and } X^l ,\\ Z^i = W^i_kX^k+W^i_lX^l + \beta^i  \\

\text{and given the activation function Sigmoid  : } 
f(x) &= \frac{1}{1+e^{-x}} \forall x \in \mathbb{R}  \\ 
\forall f(Z^i) = \frac{1}{1+e^{-Z^i}} \\
\text{Bernoulli's law with parameter } \theta=y  \\
 \text{Consider }
P(Y=y) &= f(z)^y(1-f(z))^{1-y}


\end{aligned}
$$