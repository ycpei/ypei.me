---
template: oldpost
title: A \(q\)-weighted Robinson-Schensted algorithm
date: 2013-06-01
comments: true
tags: RS, \(q\)-Whittaker_functions, Macdonald_polynomials
archive: false
---
In [this paper](https://projecteuclid.org/euclid.ejp/1465064320) with [Neil](http://www.bristol.ac.uk/maths/people/neil-m-oconnell/) we construct a \\(q\\)-version of the Robinson-Schensted
algorithm with column insertion. Like the [usual RS
correspondence](http://en.wikipedia.org/wiki/Robinson–Schensted_correspondence)
with column insertion, this algorithm could take words as input. Unlike
the usual RS algorithm, the output is a set of weighted pairs of
semistandard and standard Young tableaux \\((P,Q)\\) with the same
shape. The weights are rational functions of indeterminant \\(q\\).

If \\(q\\in\[0,1\]\\), the algorithm can be considered as a randomised
RS algorithm, with 0 and 1 being two interesting cases. When
\\(q\\to0\\), it is reduced to the latter usual RS algorithm; while
when \\(q\\to1\\) with proper scaling it should scale to directed random
polymer model in [(O'Connell 2012)](http://arxiv.org/abs/0910.0069).
When the input word \\(w\\) is a random walk:

\\begin{align\*}\\mathbb
P(w=v)=\\prod\_{i=1}^na\_{v\_i},\\qquad\\sum\_ja\_j=1\\end{align\*}

the shape of output evolves as a Markov chain with kernel related to
\\(q\\)-Whittaker functions, which are Macdonald functions when
\\(t=0\\) with a factor.


