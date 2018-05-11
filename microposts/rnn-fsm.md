---
date: 2018-05-11
---
### Some notes on RNN, FSM / FA, TM and UTM

Related to [a previous micropost](#neural-turing-machine).

[These slides from Toronto](http://www.cs.toronto.edu/~rgrosse/csc321/lec9.pdf) is a nice introduction to RNN (recurrent neural network) from a computational point of view. It states that RNN can simulate any FSM (finite state machine, a.k.a. finite automata abbr. FA) with a toy example computing the parity of a binary string.

[Goodfellow et. al.'s book](http://www.deeplearningbook.org/contents/rnn.html) (see page 372 and 374) goes one step further, stating that RNN with a hidden-to-hidden layer can simulate Turing machines, and not only that, but also the *universal* Turing machine abbr. UTM (the book referenced [Siegelmann-Sontag](https://www.sciencedirect.com/science/article/pii/S0022000085710136)), a property not shared by the weaker network where the hidden-to-hidden layer is replaced by an output-to-hidden layer (page 376).

By the way, the RNN with a hidden-to-hidden layer has the same architecture as the so-called linear dynamical system mentioned in [Hinton's video](https://www.coursera.org/learn/neural-networks/lecture/Fpa7y/modeling-sequences-a-brief-overview).

From what I have learned, the universality of RNN and feedforward networks are therefore due to different arguments, the former coming from Turing machines and the latter from an analytical view of approximation by step functions.
