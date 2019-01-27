---
date: 2019-01-27
---

My take on the [Nature paper _Learning can be undecidable_](https://www.nature.com/articles/s42256-018-0002-3):

Fantastic article, very clearly written.

So it reduces a kind of learninability called estimating the maximum (EMX) to the cardinality of real numbers which is undecidable.

When it comes to the relation between EMX and the rest of machine learning framework, the article mentions that EMX belongs to "extensions of PAC learnability include Vapnik’s statistical learning setting and the equivalent general learning setting by Shalev-Shwartz and colleagues" (I have no idea what these two things are), but it does not say whether EMX is representative of or reduces to common learning tasks. So it is not clear whether its undecidability applies to ML at large.

Another condition to the main theorem is the union bounded closure assumption. It seems a reasonable property of a family of sets, but then again I wonder how that translates to learning.

The article says "By now, we know of quite a few independence [from mathematical axioms] results, mostly for set theoretic questions like the continuum hypothesis, but also for results in algebra, analysis, infinite combinatorics and more. Machine learning, so far, has escaped this fate." but the description of the EMX learnability makes it more like a classical mathematical / theoretical computer science problem rather than machine learning.

An insightful conclusion: "How come learnability can neither be proved nor refuted? A closer look reveals that the source of the problem is in defining learnability as the existence of a learning function rather than the existence of a learning algorithm. In contrast with the existence of algorithms, the existence of functions over infinite domains is a (logically) subtle issue."

In relation to practical problems, it uses an example of ad targeting. However, A lot is lost in translation from the main theorem to this ad example.

The EMX problem states: given a domain X, a distribution P over X which is unknown, some samples from P, and a family of subsets of X called F, find A in F that approximately maximises P(A).

The undecidability rests on X being the continuous [0, 1] interval, and from the insight, we know the problem comes from the cardinality of subsets of the [0, 1] interval, which is "logically subtle".

In the ad problem, the domain X is all potential visitors, which is finite because there are finite number of people in the world. In this case P is a categorical distribution over the 1..n where n is the population of the world. One can have a good estimate of the parameters of a categorical distribution by asking for sufficiently large number of samples and computing the empirical distribution. Let's call the estimated distribution Q. One can choose the from F (also finite) the set that maximises Q(A) which will be a solution to EMX.

In other words, the theorem states: EMX is undecidable because not all EMX instances are decidable, because there are some nasty ones due to infinities. That does not mean no EMX instance is decidable. And I think the ad instance is decidable. Is there a learning task that actually corresponds to an undecidable EMX instance? I don't know, but I will not believe the result of this paper is useful until I see one.

h/t Reynaldo Boulogne
