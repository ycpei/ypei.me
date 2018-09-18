---
date: 2018-09-18
---

Just some non-rigorous guess / thought: Feedforward networks are like combinatorial logic, and recurrent networks are like sequential logic (e.g. data flip-flop is like the feedback connection in RNN). Since NAND + combinatorial logic + sequential logic = von Neumann machine which is an approximation of the Turing machine, it is not surprising that RNN (with feedforward networks) is Turing complete (assuming that neural networks can learn the NAND gate).
