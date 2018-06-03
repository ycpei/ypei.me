---
template: post
date: 2018-06-03
title: Automatic differentiation
---
This post is meant as a documentation of my understanding of autodiff. I
benefited a lot from [Toronto CSC321
slides](http://www.cs.toronto.edu/%7Ergrosse/courses/csc321_2018/slides/lec10.pdf)
and the [autodidact](https://github.com/mattjj/autodidact/) project
which is a pedagogical implementation of
[Autograd](https://github.com/hips/autograd). That said, any mistakes in
this note are mine (especially since some of the knowledge is obtained
from interpreting slides!), and if you do spot any I would be grateful
if you can let me know.

Automatic differentiation (AD) is a way to compute derivatives. It does
so by traversing through a computational graph using the chain rule.

There are the forward mode AD and reverse mode AD, which are kind of
symmetric to each other and understanding one of them results in little
to no difficulty in understanding the other.

In the language of neural networks, one can say that the forward mode AD
is used when one wants to compute the derivatives of functions at all
layers with respect to input layer weights, whereas the reverse mode AD
is used to compute the derivatives of output functions with respect to
weights at all layers. Therefore reverse mode AD (rmAD) is the one to
use for gradient descent, which is the one we focus in this post.

Basically rmAD requires the computation to be sufficiently decomposed,
so that in the computational graph, each node as a function of its
parent nodes is an elementary function that the AD engine has knowledge
about.

For example, the Sigmoid activation $a' = \sigma(w a + b)$ is quite
simple, but it should be decomposed to simpler computations:

-   $a' = 1 / t_1$
-   $t_1 = 1 + t_2$
-   $t_2 = \exp(t_3)$
-   $t_3 = - t_4$
-   $t_4 = t_5 + b$
-   $t_5 = w a$

Thus the function $a'(a)$ is decomposed to elementary operations like
addition, subtraction, multiplication, reciprocitation, exponentiation,
logarithm etc. And the rmAD engine stores the Jacobian of these
elementary operations.

Since in neural networks we want to find derivatives of a single loss
function $L(x; \theta)$, we can omit $L$ when writing derivatives and
denote, say $\bar \theta_k := \partial_{\theta_k} L$.

In implementations of rmAD, one can represent the Jacobian as a
transformation $j: (x \to y) \to (y, \bar y, x) \to \bar x$. $j$ is
called the *Vector Jacobian Product* (VJP). For example,
$j(\exp)(y, \bar y, x) = y \bar y$ since given $y = \exp(x)$,

$\partial_x L = \partial_x y \cdot \partial_y L = \partial_x \exp(x) \cdot \partial_y L = y \bar y$

as another example, $j(+)(y, \bar y, x_1, x_2) = (\bar y, \bar y)$ since
given $y = x_1 + x_2$, $\bar{x_1} = \bar{x_2} = \bar y$.

Similarly,

1.  $j(/)(y, \bar y, x_1, x_2) = (\bar y / x_2, - \bar y x_1 / x_2^2)$
2.  $j(\log)(y, \bar y, x) = \bar y / x$
3.  $j((A, \beta) \mapsto A \beta)(y, \bar y, A, \beta) = (\bar y \otimes \beta, A^T \bar y)$.
4.  etc\...

In the third one, the function is a matrix $A$ multiplied on the right
by a column vector $\beta$, and $\bar y \otimes \beta$ is the tensor
product which is a fancy way of writing $\bar y \beta^T$. See
[numpy\_vjps.py](https://github.com/mattjj/autodidact/blob/master/autograd/numpy/numpy_vjps.py)
for the implementation in autodidact.

So, given a node say $y = y(x_1, x_2, ..., x_n)$, and given the value of
$y$, $x_{1 : n}$ and $\bar y$, rmAD computes the values of
$\bar x_{1 : n}$ by using the Jacobians.

This is the gist of rmAD. It stores the values of each node in a forward
pass, and computes the derivatives of each node exactly once in a
backward pass.

It is a nice exercise to derive the backpropagation in the fully
connected feedforward neural networks (e.g. [the one for MNIST in Neural
Networks and Deep
Learning](http://neuralnetworksanddeeplearning.com/chap2.html#the_four_fundamental_equations_behind_backpropagation))
using rmAD.

AD is an approach lying between the extremes of numerical approximation
(e.g. finite difference) and symbolic evaluation. It uses exact formulas
(VJP) at each elementary operation like symbolic evaluation, while
evaluates each VJP numerically rather than lumping all the VJPs into an
unwieldy symbolic formula.

Things to look further into: the higher-order functional currying form
$j: (x \to y) \to (y, \bar y, x) \to \bar x$ begs for a functional
programming implementation.
