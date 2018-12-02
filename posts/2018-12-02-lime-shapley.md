---
title: Shapley, LIME and SHAP
date: 2018-12-02
template: post
comments: true
---

In this post I explain LIME (Ribeiro et. al. 2016), the Shapley values
(Shapley, 1953) and the SHAP values (Lundberg-Lee, 2017).

__Acknowledgement__. Thanks to Josef Lindman Hörnlund for bringing the LIME
and SHAP papers to my attention. The research is done while working at KTH 
mathematics department.

_If you are reading on a mobile device, you may need to "request desktop site"
for equations to be properly displayed. This post is licensed under CC BY-SA._

Shapley values
--------------

A coalitional game $(v, N)$ of $n$ players involves

-   The set $N = \{1, 2, ..., n\}$ that represents the players.
-   A function $v: 2^N \to \mathbb R$, where $v(S)$ is the worth of
    coalition $S \subset N$.

The Shapley values $\phi_i(v)$ of such a game specify a fair way to
distribute the total worth $v(N)$ to the players. It is defined as (in
the following, for a set $S \subset N$ we use the convention $s = |S|$
to be the number of elements of set $S$ and the shorthand
$S - i := S \setminus \{i\}$ and $S + i := S \cup \{i\}$)

$$\phi_i(v) = \sum_{S: i \in S} {(n - s)! (s - 1)! \over n!} (v(S) - v(S - i)).$$

$\phi_i(v)$ is an expectation:

$$\phi_i(v) = \mathbb E_{S \sim \nu_i} (v(S) - v(S - i))$$

where $\nu_i(S) = n^{-1} {n - 1 \choose s - 1}^{-1} 1_{i \in S}$, that
is, first pick the size $s$ uniformly from $\{1, 2, ..., n\}$, then pick
$S$ uniformly from the subsets of $N$ that has size $s$ and contains
$i$.

The Shapley values satisfy some nice properties which are readily
verified, including:

-   **Efficiency**.
    $\sum_i \phi_i(v) = v(N) - v(\emptyset)$.
-   **Symmetry**. If for some $i, j \in N$, for all
    $S \subset N$, we have $v(S + i) = v(S + j)$, then
    $\phi_i(v) = \phi_j(v)$.
-   **Null player**. If for some $i \in N$, for all
    $S \subset N$, we have $v(S + i) = v(S)$, then $\phi_i(v) = 0$.
-   **Linearity**. $\phi_i$ is linear in games. That is
    $\phi_i(v) + \phi_i(w) = \phi_i(v + w)$, where $v + w$ is defined by
    $(v + w)(S) := v(S) + w(S)$.

In the literature, an added assumption $v(\emptyset) = 0$ is often
given, in which case the Efficiency property is defined as
$\sum_i \phi_i(v) = v(N)$. Here I discard this assumption to avoid minor
inconsistencies across different sources. For example, in the LIME
paper, the local model is defined without an intercept, even though the
underlying $v(\emptyset)$ may not be $0$. In the SHAP paper, an
intercept $\phi_0 = v(\emptyset)$ is added which fixes this problem when
making connections to the Shapley values.

Conversely, according to Strumbelj-Kononenko (2010), it was shown in
Shapley\'s original paper (Shapley, 1953) that these four properties
together with $v(\emptyset) = 0$ defines the Shapley values.

LIME
----

LIME (Ribeiro et. al. 2016) is a model that offers a way to explain
feature contributions of supervised learning models locally.

Let $f: X_1 \times X_2 \times ... \times X_n \to \mathbb R$ be a
function. We can think of $f$ as a model, where $X_j$ is the space of
$j$th feature. For example, in a language model, $X_j$ may be the count
of the $j$th word in the vocabulary.

The output may be something like housing price, or log-probability of
something.

LIME tries to assign a value to each feature *locally*. By locally, we
mean that given a specific sample $x \in X := \prod_{i = 1}^n X_i$, we
want to fit a model around it.

More specifically, let $h_x: 2^N \to X$ be a function defined by

$$(h_x(S))_i = 
\begin{cases}
x_i, & \text{if }i \in S; \\
0, & \text{otherwise.}
\end{cases}$$

That is, $h_x(S)$ masks the features that are not in $S$, or in other
words, we are perturbing the sample $x$. Specifically, $h_x(N) = x$.
Alternatively, the $0$ in the \"otherwise\" case can be replaced by some
kind of default value (see the last section of this post).

For a set $S \subset N$, let us denote $1_S \in \{0, 1\}^n$ to be an
$n$-bit where the $k$th bit is $1$ if and only if $k \in S$.

Basically, LIME samples $S_1, S_2, ..., S_m \subset N$ to obtain a set
of perturbed samples $x_i = h_x(S_i)$ in the $X$ space, and then fits a
linear model $g$ using $1_{S_i}$ as the input samples and $f(h_x(S_i))$
as the output samples:

**Problem**(LIME). Find $w = (w_1, w_2, ..., w_n)$ that
minimises

$$\sum_i (w \cdot 1_{S_i} - f(h_x(S_i)))^2 \pi_x(h_x(S_i))$$

where $\pi_x(x')$ is a function that penalises $x'$s that are far away
from $x$. In the LIME paper the Gaussian kernel was used:

$$\pi_x(x') = \exp\left({- \|x - x'\|^2 \over \sigma^2}\right).$$

Then $w_i$ represents the importance of the $i$th feature.

The LIME model has a more general framework, but the specific model
considered in the paper is the one described above, with a Lasso for
feature selection.

Shapley values and LIME
-----------------------

The connection between the Shapley values and LIME is noted in
Lundberg-Lee (2017), but the underlying connection goes back to 1988
(Charnes et. al.).

To see the connection, we need to modify LIME a bit.

First, we need to make LIME less efficient by considering *all* the
$2^n$ subsets instead of the $m$ samples $S_1, S_2, ..., S_{m}$.

Then we need to relax the definition of $\pi_x$. It no longer needs to
penalise samples that are far away from $x$. In fact, we will see later
than the choice of $\pi_x(x')$ that yields the Shapley values is high
when $x'$ is very close or very far away from $x$, and low otherwise. We
further add the restriction that $\pi_x(h_x(S))$ only depends on the
size of $S$, thus we rewrite it as $q(s)$ instead.

We also denote $v(S) := f(h_x(S))$ and $w(S) = \sum_{i \in S} w_i$.

Finally, we add the Efficiency property as a constraint:
$\sum_{i = 1}^n w_i = f(x) - f(h_x(\emptyset)) = v(N) - v(\emptyset)$.

Then the problem becomes a weighted linear regression:

**Problem**. minimises
$\sum_{S \subset N} (w(S) - v(S))^2 q(s)$ over $w$ subject to
$w(N) = v(N) - v(\emptyset)$.

**Claim** (Charnes et. al. 1988). The solution to this problem
is

$$w_i = {1 \over n} (v(N) - v(\emptyset)) + \left(\sum_{s = 1}^{n - 1} {n - 2 \choose s - 1} q(s)\right)^{-1} \sum_{S \subset N: i \in S} \left({n - s \over n} q(s) v(S) - {s - 1 \over n} q(s - 1) v(S - i)\right). \qquad (-1)$$

Specifically, if we choose

$$q(s) = c {n - 2 \choose s - 1}^{-1}$$

for any constant $c$, then $w_i = \phi_i(v)$ are the Shapley values.

**Remark**. Don\'t worry about this specific choice of $q(s)$
when $s = 0$ or $n$, because $q(0)$ and $q(n)$ do not appear on the
right hand side of (-1). Therefore they can be defined to be of any
value. A common convention of the binomial coefficients is to set
${\ell \choose k} = 0$ if $k < 0$ or $k > \ell$, in which case
$q(0) = q(n) = \infty$.

In Lundberg-Lee (2017), $c$ is chosen to be $1 / n$, see Theorem 2
there.

**Proof**. The Lagrangian is

$$L(w, \lambda) = \sum_{S \subset N} (v(S) - w(S))^2 q(s) - \lambda(w(N) - v(N) + v(\emptyset)).$$

and by making $\partial_{w_i} L(w, \lambda) = 0$ we have

$${1 \over 2} \lambda = \sum_{S \subset N: i \in S} (w(S) - v(S)) q(s). \qquad (0)$$

Summing (0) over $i$ and divide by $n$, we have

$${1 \over 2} \lambda = {1 \over n} \sum_i \sum_{S: i \in S} (w(S) q(s) - v(S) q(s)). \qquad (1)$$

We examine each of the two terms on the right hand side.

Counting the terms involving $w_i$ and $w_j$ for $j \neq i$, and using
$w(N) = v(N) - v(\emptyset)$ we have

$$\begin{aligned}
&\sum_{S \subset N: i \in S} w(S) q(s) \\
&= \sum_{s = 1}^n {n - 1 \choose s - 1} q(s) w_i + \sum_{j \neq i}\sum_{s = 2}^n {n - 2 \choose s - 2} q(s) w_j \\
&= q(1) w_i + \sum_{s = 2}^n q(s) \left({n - 1 \choose s - 1} w_i + \sum_{j \neq i} {n - 2 \choose s - 2} w_j\right) \\
&= q(1) w_i + \sum_{s = 2}^n \left({n - 2 \choose s - 1} w_i + {n - 2 \choose s - 2} (v(N) - v(\emptyset))\right) q(s) \\
&= \sum_{s = 1}^{n - 1} {n - 2 \choose s - 1} q(s) w_i + \sum_{s = 2}^n {n - 2 \choose s - 2} q(s) (v(N) - v(\emptyset)). \qquad (2)
\end{aligned}$$

Summing (2) over $i$, we obtain

$$\begin{aligned}
&\sum_i \sum_{S: i \in S} w(S) q(s)\\
&= \sum_{s = 1}^{n - 1} {n - 2 \choose s - 1} q(s) (v(N) - v(\emptyset)) + \sum_{s = 2}^n n {n - 2 \choose s - 2} q(s) (v(N) - v(\emptyset))\\
&= \sum_{s = 1}^n s{n - 1 \choose s - 1} q(s) (v(N) - v(\emptyset)). \qquad (3)
\end{aligned}$$

For the second term in (1), we have

$$\sum_i \sum_{S: i \in S} v(S) q(s) = \sum_{S \subset N} s v(S) q(s). \qquad (4)$$

Plugging (3)(4) in (1), we have

$${1 \over 2} \lambda = {1 \over n} \left(\sum_{S \subset N} s q(s) v(S) - \sum_{s = 1}^n s {n - 1 \choose s - 1} q(s) (v(N) - v(\emptyset))\right). \qquad (5)$$

Plugging (5)(2) in (0) and solve for $w_i$, we have

$$w_i = {1 \over n} (v(N) - v(\emptyset)) + \left(\sum_{s = 1}^{n - 1} {n - 2 \choose s - 1} q(s) \right)^{-1} \left( \sum_{S: i \in S} q(s) v(S) - {1 \over n} \sum_{S \subset N} s q(s) v(S) \right). \qquad (6)$$

By splitting all subsets of $N$ into ones that contain $i$ and ones that
do not and pair them up, we have

$$\sum_{S \subset N} s q(s) v(S) = \sum_{S: i \in S} (s q(s) v(S) + (s - 1) q(s - 1) v(S - i)).$$

Plugging this back into (6) we get the desired result. $\square$

SHAP
----

The SHAP paper (Lundberg-Lee 2017) is not clear in its definition of the
\"SHAP values\" and its relation to LIME, so the following is my
interpretation of their interpretation model.

Recall that we want to calculate feature contributions to a model $f$ at
a sample $x$.

Let $\mu$ be a probability density function over the input space
$X = X_1 \times ... \times X_n$. A natural choice would be the density
that generates the data, or one that approximates such density (e.g.
empirical distribution).

The feature contribution (SHAP value) is thus defined as the Shapley
value $\phi_i(v)$, where

$$v(S) = \mathbb E_{z \sim \mu} (f(z) | z_S = x_S). \qquad (7)$$

So it is a conditional expectation where $z_i$ is clamped for $i \in S$.

One simplification is to assume the $n$ features are independent, thus
$\mu = \mu_1 \times \mu_2 \times ... \times \mu_n$. In this case, (7)
becomes

$$v(S) = \mathbb E_{z_{N \setminus S} \sim \mu_{N \setminus S}} f(x_S, z_{N \setminus S}) \qquad (8)$$

For example, Strumbelj-Kononenko (2010) considers this where $\mu$ is
the uniform distribution over $X$, see Definition 4 there.

A further simplification is model linearity, which means $f$ is linear.
In this case, (8) becomes

$$v(S) = f(x_S, \mathbb E_{\mu_{N \setminus S}} z_{N \setminus S}). \qquad (9)$$

It is worth noting that to make the modified LIME model considered in
the previous section fall under the linear SHAP framework (9), we need
to make a further specialisation, that is, change the definition of
$h_x(S)$ to

$$(h_x(S))_i = 
\begin{cases}
x_i, & \text{if }i \in S; \\
\mathbb E_{\mu_i} z_i, & \text{otherwise.}
\end{cases}$$

References
----------

-   Charnes, A., B. Golany, M. Keane, and J. Rousseau. "Extremal
    Principle Solutions of Games in Characteristic Function Form: Core,
    Chebychev and Shapley Value Generalizations." In Econometrics of
    Planning and Efficiency, edited by Jati K. Sengupta and Gopal K.
    Kadekodi, 123--33. Dordrecht: Springer Netherlands, 1988.
    <https://doi.org/10.1007/978-94-009-3677-5_7>.
-   Lundberg, Scott, and Su-In Lee. "A Unified Approach to Interpreting
    Model Predictions." ArXiv:1705.07874 \[Cs, Stat\], May 22, 2017.
    <http://arxiv.org/abs/1705.07874>.
-   Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. "'Why
    Should I Trust You?': Explaining the Predictions of Any Classifier."
    ArXiv:1602.04938 \[Cs, Stat\], February 16, 2016.
    <http://arxiv.org/abs/1602.04938>.
-   Shapley, L. S. "17. A Value for n-Person Games." In Contributions to
    the Theory of Games (AM-28), Volume II, Vol. 2. Princeton: Princeton
    University Press, 1953. <https://doi.org/10.1515/9781400881970-018>.
-   Strumbelj, Erik, and Igor Kononenko. "An Efficient Explanation of
    Individual Classifications Using Game Theory." J. Mach. Learn. Res.
    11 (March 2010): 1--18.
