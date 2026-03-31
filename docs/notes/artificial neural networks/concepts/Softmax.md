The **softmax function,** also known as **softargmax**  or **normalized exponential function**  converts a tuple $z=(z_1,z_2,...,z_k)\in \mathbb{R^k}$ of K real numbers into a probability distribution over K possible outcomes. 
It is a generalization of the logistic function **$f(x)=\frac{L}{1 + e^{-k(x-x_0)}}$** to multiple dimensions, and is used in multinomial logistic regression. 
The softmax function is often used as the last activation function of a neural network to normalize the output of a network to a probability distribution over predicted output classes.
## Formula 
$$ \begin{gather*} 
\sigma : \mathbb{R^k} \rightarrow (0,1)^k , \text{ where } k>1 , \\ \text{ takes a tuple } z=(z_1,...,z_k) \in \mathbb{R^k} \text{ and comutes each component of vector } \sigma(z) \in (0,1)^k \text{ with } \\ \sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{k}e^{z_i}} \\ \sum_{i=1}^{k} \sigma(z)_i = 1 
\end{gather*}$$

 ![[Pasted image 20260326010735.png|693]]