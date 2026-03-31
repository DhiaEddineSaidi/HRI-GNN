## Definition

A **tensor** is a mathematical object that generalizes **scalars, vectors, and matrices** to higher dimensions.  
In practice (especially in machine learning), a tensor is a **multi-dimensional array of numbers** used to represent data.

|Object|Tensor order|Example|
|---|---|---|
|Scalar|0D tensor|temperature, loss value|
|Vector|1D tensor|list of numbers|
|Matrix|2D tensor|table of values|
|Tensor|3D+ tensor|images, videos, batches of data|

---

## Representation

A tensor can be written with indices:

$$  
T_{i,j,k}  
$$

Example of a **3D tensor**:

$$  
T =  
\begin{bmatrix}  
\begin{bmatrix}1 , \frac{2}{3} , 4\end{bmatrix},  
\begin{bmatrix}5, \frac{6}{7}, 8\end{bmatrix}  
\end{bmatrix}  
$$

For an **RGB image**:

$$  
I \in \mathbb{R}^{H \times W \times 3}  
$$

- (H): height
    
- (W): width
    
- (3): color channels (R,G,B)
    

---

## Tensor in Python

Using **NumPy**:

```python
import numpy as np

tensor = np.array([
    [[1,2],[3,4]],
    [[5,6],[7,8]]
])

print(tensor.shape)
```

Using **PyTorch**:

```python
import torch

t = torch.tensor([
    [[1,2],[3,4]],
    [[5,6],[7,8]]
])
```

---

## Main Properties

**1. Shape**

Number of elements in each dimension.

Example:

```
(2, 2, 2)
```

means:

```
depth × rows × columns
```

---

**2. Rank (or order)**

Number of dimensions.

Examples:

|Shape|Rank|
|---|---|
|()|0|
|(5,)|1|
|(3,4)|2|
|(3,4,5)|3|

---

**3. Data type**

Type of values stored:

- int
    
- float
    
- bool
    

Example:

```python
tensor.dtype
```

---

**4. Device (in deep learning)**

Where the tensor is stored:

- CPU
    
- GPU
    

Example:

```python
tensor.device
```

---

## Why tensors are important

Tensors are the **core data structure in deep learning frameworks** such as:

- PyTorch
    
- TensorFlow
    

They allow efficient **vectorized computation on CPUs and GPUs**.

Examples of tensor data:

|Data type|Tensor shape|
|---|---|
|grayscale image|H × W|
|RGB image|H × W × 3|
|video|frames × H × W × 3|
|batch of images|N × H × W × 3|