---
date: '2019-03-15'
title: List of Notations
template: default
---

Here I list meanings of notations that may have not been explained
elsewhere.

-   $\text{ty}$: type. Given a word $w \in [n]^\ell$,
    $\text{ty} w = (m_1, m_2, ..., m_n)$ where $m_i$ is the number of
    $i$\'s in $w$. For example
    $\text{ty} (1, 2, 2, 1, 4, 2) = (2, 3, 0, 1)$. The definition of
    $\text{ty} T$ for a tableau $T$ is similar.
-   $[n]$: for $n \in \mathbb N_{>0}$, $[n]$ stands for the set
    $\{1, 2, ..., n\}$.
-   $i : j$: for $i, j \in \mathbb Z$, $i : j$ stands for the set
    $\{i, i + 1, ..., j\}$, or the sequence $(i, i + 1, ..., j)$,
    depending on the context.
-   $k = i : j$: means $k$ iterates over $i$, $i + 1$,\..., $j$. For
    example $\sum_{k = 1 : n} a_k := \sum_{k = 1}^n a_k$.
-   $x_{i : j}$: stands for the set $\{x_k: k = i : j\}$ or the sequence
    $(x_i, x_{i + 1}, ..., x_j)$, depending on the context. So are
    notations like $f(i : j)$, $y^{i : j}$ etc.
-   $\mathbb N$: the set of natural numbers / nonnegative integer
    numbers $\{0, 1, 2,...\}$, whereas
-   $\mathbb N_{>0}$ or $\mathbb N^+$: Are the set of positive integer
    numbers.
-   $x^w$: when both $x$ and $w$ are tuples of objects, this means
    $\prod_i x_{w_i}$. For example say $w = (1, 2, 2, 1, 4, 2)$, and
    $x = x_{1 : 7}$, then $x^w = x_1^2 x_2^3 x_4$.
-   $LHS$, LHS, $RHS$, RHS: left hand side and right hand side of a
    formula
-   $e_i$: the $i$th standard basis in a vector space:
    $e_i = (0, 0, ..., 0, 1, 0, 0, ...)$ where the sequence is finite or
    infinite depending on the dimension of the vector space and the $1$
    is the $i$th entry and all other entries are $0$.
-   $1_{A}(x)$ where $A$ is a set: an indicator function, which
    evaluates to $1$ if $x \in A$, and $0$ otherwise.
-   $1_{p}$: an indicator function, which evaluates to $1$ if the
    predicate $p$ is true and $0$ otherwise. Example: $1_{x \in A}$,
    same as $1_A(x)$.
-   $\xi \sim p$: the random variable $xi$ is distributed according to
    the probability density function / probability mass function /
    probability measure $p$.
-   $\xi \overset{d}{=} \eta$: the random variables $\xi$ and $\eta$
    have the same distribution.
-   $\mathbb E f(\xi)$: expectation of $f(\xi)$.
-   $\mathbb P(A)$: probability of event $A$.
-   $a \wedge b$: $\min\{a, b\}$.
-   $a \vee b$: $\max\{a, b\}$.
-   $(\alpha)_+$: the positive part of $\alpha$, i.e. $\alpha \vee 0$.
-   $(\alpha)_-$: the negative part of $\alpha$, i.e. $(- \alpha)_+$.
