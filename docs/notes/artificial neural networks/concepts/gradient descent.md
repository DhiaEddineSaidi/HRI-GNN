The gradient descent algorithm is an unconstrained mathematical optimization method. It is a first-order iterative algorithm for minimizing a differentiable multivariate function. The idea is to perform repeated steps in the direction **opposite** $(-)$  to the gradient $\nabla{f}=\frac{df}{dr}\text{ (coordinate free terms)}$ of the function at the current point $p$ , as this is the steepest descent direction. 
Conversely, moving in the direction of the gradient will lead to a path that maximizes the function; this procedure is then known as gradient ascent.
- #### The Gradient :
	- coordinate free terms  : $$ \nabla{f} = \frac{df}{dr} $$
	- Coordinate system : $$ 
	  \begin{aligned} 
	  \text{ for } & f:\mathbb{R}^n \rightarrow \mathbb{R} \\ \\ 
	  & \nabla{f}: \mathbb{R}^n \rightarrow \mathbb{R}^n \text{, is defined at the point }p = (x_1,...,x_n) \\ &\text{ in n-dimentional space as the vector } \\
	  & \nabla{f}(p)= \begin{bmatrix}  
						\frac{\partial f}{\partial x_1}(p) \\
						\vdots \\
						\frac{\partial f}{\partial x_n}(p)   
					\end{bmatrix} \\ 
	  \end{aligned}$$
 - #### The Gradient Decent formula : 
   ou function $f$ here is the [[Loss Function]] $\mathcal{L}$, and the **$\alpha$ is the learning rate**
$$\begin{aligned} 
& W_{t+1} = W_t - \alpha \nabla{f} \text{ with our function f here is the Log Loss function } \mathcal{L} \\ \\ 
&\mathcal{L}(Y, f(X)) = -\frac{1}{N}\sum_{i=1}^{N}Y^ilog(\hat{Y}^ii) + (1 - Y^i)log(1 - \hat{Y}^i)  \\ 
&\text{ where } Y = \{ Y^1,...,Y^n\} \text{ The Set of Obseerved values and } Y^i = \begin{bmatrix} y_1^i \\ \vdots \\ y_n^i \end{bmatrix} \\
&\text{ and } \hat{Y} = \{ \hat{Y}^1,...,\hat{Y}^n\} \text{ The Set of Outputted values and } \hat{Y}^i = \begin{bmatrix} \hat{y}_1^i \\ \vdots \\ \hat{y}_n^i \end{bmatrix} \\ 
& \text{ and } X = \{ X^1,...,X^n\} \text{ The Set of Input values and } X^i = \begin{bmatrix} x_1^i \\ \vdots \\ x_n^i \end{bmatrix} \\

\end{aligned}$$