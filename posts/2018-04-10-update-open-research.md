---
template: post
date: 2018-04-28
title: Updates on open research
---

It has been 9 months since I last wrote about open (maths) research. Since then two things happened which prompted me to write an update.

As always I discuss open research only in mathematics, not because I think it should not be applied to other disciplines, but simply because I do not have experience nor sufficient interests in non-mathematical subjects.

First, I read about Richard Stallman the founder of the free software movement, in [his biography by Sam Williams](http://shop.oreilly.com/product/9780596002879.do) and his own collection of essays [_Free software, free society_](https://shop.fsf.org/books-docs/free-software-free-society-selected-essays-richard-m-stallman-3rd-edition), from which I learned a bit more about the context and philosophy of free software and open source software.
For anyone interested in open research, I highly recommend having a look at these two books.
I am also reading Levy's [Hackers](http://www.stevenlevy.com/index.php/books/hackers), which documented the development of the hacker culture predating Stallman.
I can see the connection of ideas from the hacker ethic to free software to the open source philosophy.
My guess is that the software world is fortunate to have pioneers who advocated for freedom and openness from the beginning, whereas for academia which has a much longer history, credit protection has always been a bigger concern.

Also a month ago I attended a workshop called [Open research: rethinking scientific collaboration](https://www.perimeterinstitute.ca/conferences/open-research-rethinking-scientific-collaboration). That was the first time I met a group of people (mostly physicists) who also want open research to happen, and we had some stimulating discussions.

From both of these I feel like I should write an updated post on open research.

### Freedom and community
Ideals matter. Stallman's struggles stemmed from the frustration of denied request of source code (a frustration I shared in academia except source code is replaced by maths knowledge), and revolved around two things that underlie the free software movement: freedom and community.
That is, the freedom to use, modify and share a work, and by sharing, to help the community.

Likewise, as for open research, apart from the utilitarian view that open research is more efficient / harder for credit theft, we should not ignore the ethical aspect that open research is right and fair.
In particular, I think freedom and community can also serve as principles in open research.
One way to make this argument more concrete is to describe what I feel are the central problems: NDAs (non-disclosure agreements) and reproducibility.

__NDAs__. It is assumed that when establishing a research collaboration, or just having a discussion, all those involved own the joint work in progress, and no one has the freedom to disclose any information e.g. intermediate results without getting permission from all collaborators. In effect this amounts to signing an NDA.
NDAs are harmful because they restrict people's freedom from sharing information that can benefit their own or others' research.
Considering that in contrast to the private sector, the primary goal of academia is knowledge but not profit, NDAs in research are unacceptable.

__Reproducibility__. Research papers written down are not necessarily reproducible, even though they appear on peer-reviewed journals.
This is because the peer-review process is opaque and the proofs in the papers may not be clear to everyone.
To make things worse, there are no open channels to discuss results in these papers and one may have to rely on interacting with the small circle of the informed. 
One example is folk theorems. Another is trade secrets required to decipher published works.

I should clarify that freedom works both ways. One should have the freedom to disclose maths knowledge, but they should also be free to withhold any information that does not hamper the reproducibility of published works (e.g. results in ongoing research yet to be published), even though it may not be nice to do so when such information can help others with their research.

Similar to the solution offered by the free software movement, we need a community that promotes and respects free flow of maths knowledge, in the spirit of the [four essential freedoms](https://www.gnu.org/philosophy/), a community that rejects NDAs and upholds reproducibility.

Here are some ideas on how to tackle these two problems and build the community:

1. Free licensing. It solves NDA problem - free licenses permit redistribution and modification of works, so if you adopt them in your joint work, then you have the freedom to modify and distribute the work; it also helps with reproducibility - if a paper is not clear, anyone can write their own version and publish it. Bonus points with the use of copyleft licenses like [Creative Commons Share-Alike](https://creativecommons.org/licenses/by-sa/4.0/) or the [GNU Free Documentation License](https://www.gnu.org/licenses/fdl.html).
2. A forum for discussions of mathematics. It helps solve the reproducibility problem - public interaction may help quickly clarify problems. By the way, Math Overflow is not a forum.
3. An infrastructure of mathematical knowledge. Like the GNU system, a mathematics encyclopedia under a copyleft license maintained in the Github-style rather than Wikipedia-style by a "Free Mathematics Foundation", and drawing contributions from the public (inside or outside of the academia). To begin with, crowd-source (again, Github-style) the proofs of say 1000 foundational theorems covered in the curriculum of a bachelor's degree. Perhaps start with taking contributions from people with some credentials (e.g. having a bachelor degree in maths) and then expand the contribution permission to the public, or taking advantage of existing corpus under free license like Wikipedia.
4. Citing with care: if a work is considered authorative but you couldn't reproduce the results, whereas another paper which tries to explain or discuss similar results makes the first paper understandable to you, give both papers due attribution (something like: see [1], but I couldn't reproduce the proof in [1], and the proofs in [2] helped clarify it). No one should be offended if you say you can not reproduce something - there may be causes on both sides, whereas citing [2] is fairer and helps readers with a similar background.

### Tools for open research

The open research workshop revolved around how to lead academia towards a more open culture.
There were discussions on open research tools, improving credit attributions, the peer-review process and the path to adoption.

During the workshop many efforts for open research were mentioned, and afterwards I was also made aware by more of them, like the following:

- [OSF](https://osf.io), an online research platform. It has a clean and simple interface with commenting, wiki, citation generation, DOI generation, tags, license generation etc. Like Github it supports private and public repositories (but defaults to private), version control, with the ability to fork or bookmark a project.
- [SciPost](https://scipost.org/), physics journals whose peer review reports and responses are public (peer-witnessed refereeing), and allows comments (post-publication evaluation). Like arXiv, it requires some academic credential (PhD or above) to register.
- [Knowen](https://knowen.org/), a platform to organise knowledge in directed acyclic graphs. Could be useful for building the infrastructure of mathematical knowledge.
- [Fermat's Library](https://fermatslibrary.com/), the journal club website that crowd-annotates one notable paper per week released a Chrome extension [Librarian](https://fermatslibrary.com/librarian) that overlays a commenting interface on arXiv. As an example Ian Goodfellow did an [AMA (ask me anything) on his GAN paper](https://fermatslibrary.com/arxiv_comments?url=https://arxiv.org/pdf/1406.2661.pdf).
- [The Polymath project](https://polymathprojects.org/), the famous massive collaborative mathematical project. Not exactly new, the Polymath project is the only open maths research project that has gained some traction and recognition. However, it does not have many active projects ([currently only one active project](http://michaelnielsen.org/polymath1/index.php?title=Main_Page)).
- [The Stacks Project](https://stacks.math.columbia.edu/). I was made aware of this project by [Yiting](https://people.kth.se/~yitingl/). Its data is hosted on github and accepts contributions via pull requests and is licensed under the GNU Free Documentation License, ticking many boxes of the free and open source model.

### An anecdote from the workshop

In a conversation during the workshop, one of the participants called open science "normal science", because reproducibility, open access, collaborations, and fair attributions are all what science is supposed to be, and practices like treating the readers as buyers rather than users should be called "bad science", rather than "closed science".

To which an organiser replied: maybe we should rename the workshop "Not-bad science".
