---
template: oldpost
title: A \(q\)-Robinson-Schensted-Knuth algorithm and a \(q\)-polymer
date: 2016-10-13
comments: true
archive: false
---
(Latest update: 2017-01-12)
In [Matveev-Petrov 2016](http://arxiv.org/abs/1504.00666) a  \\(q\\)-deformed Robinson-Schensted-Knuth algorithm (\\(q\\)RSK) was introduced. In this article we give reformulations of this algorithm in terms of Noumi-Yamada description, growth diagrams and local moves. We show that the algorithm is symmetric, namely the output tableaux pair are swapped in a sense of distribution when the input matrix is transposed. We also formulate a  \\(q\\)-polymer model based on the \\(q\\)RSK and prove the corresponding Burke property, which we use to show a strong law of large numbers for the partition function given stationary boundary conditions and  \\(q\\)-geometric weights. We use the  \\(q\\)-local moves to define a generalisation of the \\(q\\)RSK taking a Young diagram-shape of array as the input. We write down the joint distribution of partition functions in the space-like direction of the  \\(q\\)-polymer in  \\(q\\)-geometric environment, formulate a  \\(q\\)-version of the multilayer polynuclear growth model (\\(q\\)PNG) and write down the joint distribution of the  \\(q\\)-polymer partition functions at a fixed time. 

This article is available at [arXiv](https://arxiv.org/abs/1610.03692).
It seems to me that one difference between arXiv and Github is that on arXiv each preprint has a few versions only.
In Github many projects have a "dev" branch hosting continuous updates, whereas the master branch is where the stable releases live.

[Here]({{ site.url }}/assets/resources/qrsklatest.pdf) is a "dev" version of the article, which I shall push to arXiv when it stablises. Below is the changelog.

* 2017-01-12: Typos and grammar, arXiv v2.
* 2016-12-20: Added remarks on the geometric \\(q\\)-pushTASEP. Added remarks on the converse of the Burke property. Added natural language description of the \\(q\\)RSK. Fixed typos.
* 2016-11-13: Fixed some typos in the proof of Theorem 3.
* 2016-11-07: Fixed some typos. The \\(q\\)-Burke property is now stated in a more symmetric way, so is the law of large numbers Theorem 2.
* 2016-10-20: Fixed a few typos. Updated some references. Added a reference: [a set of notes titled "RSK via local transformations"](http://web.mit.edu/~shopkins/docs/rsk.pdf).
It is written by [Sam Hopkins](http://web.mit.edu/~shopkins/) in 2014 as an expository article based on MIT combinatorics preseminar presentations of Alex Postnikov.
It contains some idea (applying local moves to a general Young-diagram shaped array in the order that matches any growth sequence of the underlying Young diagram) which I thought I was the first one to write down.
