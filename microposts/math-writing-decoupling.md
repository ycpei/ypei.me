---
date: 2018-05-10
---
### Writing readable mathematics like writing an operating system

One way to write readable mathematics is to decouple concepts. One idea is the following template. First write a toy example with all the important components present in this example, then analyse each component individually and elaborate how (perhaps more complex) variations of the component can extend the toy example and induce more complex or powerful versions of the toy example. Through such incremental development, one should be able to arrive at any result in cutting edge research after a pleasant journey.

It's a bit like the UNIX philosophy, where you have a basic system of modules like IO, memory management, graphics etc, and modify / improve each module individually (H/t [NAND2Tetris](http://nand2tetris.org/)).

The book [Neutral networks and deep learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen is an example of such approach. It begins the journey with a very simple neutral net with one hidden layer, no regularisation, and sigmoid activations. It then analyses each component including cost functions, the back propagation algorithm, the activation functions, regularisation and the overall architecture (from fully connected to CNN) individually and improve the toy example incrementally. Over the course the accuracy of the example of mnist grows incrementally from 95.42% to 99.67%.
