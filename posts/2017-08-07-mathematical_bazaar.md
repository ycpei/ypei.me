---
template: oldpost
title: The Mathematical Bazaar
date: 2017-08-07
comments: true
archive: false
---

In this essay I describe some problems in academia of mathematics and
propose an open source model, which I call open research in mathematics.

This essay is a work in progress - comments and criticisms are welcome!
[^feedback]

Before I start I should point out that

1.  Open research is *not* open access. In fact the latter is a
    prerequisite to the former.
2.  I am not proposing to replace the current academic model with the
    open model - I know academia works well for many people and I am
    happy for them, but I think an open research community is long
    overdue since the wide adoption of the World Wide Web more than two
    decades ago. In fact, I fail to see why an open model can not run in
    tandem with the academia, just like open source and closed source
    software development coexist today.

problems of academia
--------------------

Open source projects are characterised by publicly available source
codes as well as open invitations for public collaborations, whereas closed
source projects do not make source codes accessible to the public. How
about mathematical academia then, is it open source or closed source? The
answer is neither.

Compared to some other scientific disciplines, mathematics does not
require expensive equipments or resources to replicate results; compared
to programming in conventional software industry, mathematical findings
are not meant to be commercial, as credits and reputation rather than
money are the direct incentives (even though the former are commonly
used to trade for the latter). It is also a custom and common belief
that mathematical derivations and theorems shouldn\'t be patented.
Because of this, mathematical research is an open source activity in the
sense that proofs to new results are all available in papers, and thanks
to open access e.g. the arXiv preprint repository most of the new
mathematical knowledge is accessible for free.

Then why, you may ask, do I claim that maths research is not open
sourced? Well, this is because 1. mathematical arguments are not easily
replicable and 2. mathematical research projects are mostly not open for
public participation.

Compared to computer programs, mathematical arguments are not written in
an unambiguous language, and they are terse and not written in maximum
verbosity (this is especially true in research papers as journals
encourage limiting the length of submissions), so the understanding of a
proof depends on whether the reader is equipped with the right
background knowledge, and the completeness of a proof is highly
subjective. More generally speaking, computer programs are mostly
portable because all machines with the correct configurations can
understand and execute a piece of program, whereas humans are subject to
their environment, upbringings, resources etc. to have a brain ready to
comprehend a proof that interests them. (these barriers are softer than
the expensive equipments and resources in other scientific fields
mentioned before because it is all about having access to the right
information)

On the other hand, as far as the pursuit of reputation and prestige
(which can be used to trade for the scarce resource of research
positions and grant money) goes, there is often little practical
motivation for career mathematicians to explain their results to the
public carefully. And so the weird reality of the mathematical academia
is that it is not an uncommon practice to keep trade secrets in order to
protect one\'s territory and maintain a monopoly. This is doable because
as long as a paper passes the opaque and sometimes political peer review
process and is accepted by a journal, it is considered work done,
accepted by the whole academic community and adds to the reputation of
the author(s). Just like in the software industry, trade secrets and
monopoly hinder the development of research as a whole, as well as
demoralise outsiders who are interested in participating in related
research.

Apart from trade secrets and territoriality, another reason to the
nonexistence of open research community is an elitist tradition in the
mathematical academia, which goes as follows:

-   Whoever is not good at mathematics or does not possess a degree in
    maths is not eligible to do research, or else they run high risks of
    being labelled a crackpot.
-   Mistakes made by established mathematicians are more tolerable than
    those less established.
-   Good mathematical writings should be deep, and expositions of
    non-original results are viewed as inferior work and do not add to
    (and in some cases may even damage) one\'s reputation.

All these customs potentially discourage public participations in
mathematical research, and I do not see them easily go away unless an
open source community gains momentum.

To solve the above problems, I propose a open source model of
mathematical research, which has high levels of openness and
transparency and also has some added benefits listed in the last section
of this essay. This model tries to achieve two major goals:

-   Open and public discussions and collaborations of mathematical
    research projects online
-   Open review to validate results, where author name, reviewer name,
    comments and responses are all publicly available online.

To this end, a Github model is fitting. Let me first describe how open
source collaboration works on Github.

open source collaborations on Github
------------------------------------

On [Github](https://github.com), every project is publicly available in
a repository (we do not consider private repos). The owner can update
the project by \"committing\" changes, which include a message of what
has been changed, the author of the changes and a timestamp. Each
project has an issue tracker, which is basically a discussion forum
about the project, where anyone can open an issue (start a discussion),
and the owner of the project as well as the original poster of the issue
can close it if it is resolved, e.g. bug fixed, feature added, or out of
the scope of the project. Closing the issue is like ending the
discussion, except that the thread is still open to more posts for
anyone interested. People can react to each issue post, e.g. upvote,
downvote, celebration, and importantly, all the reactions are public
too, so you can find out who upvoted or downvoted your post.

When one is interested in contributing code to a project, they fork it,
i.e. make a copy of the project, and make the changes they like in the
fork. Once they are happy with the changes, they submit a pull request
to the original project. The owner of the original project may accept or
reject the request, and they can comment on the code in the pull
request, asking for clarification, pointing out problematic part of the
code etc and the author of the pull request can respond to the comments.
Anyone, not just the owner can participate in this review process,
turning it into a public discussion. In fact, a pull request is a
special issue thread. Once the owner is happy with the pull request,
they accept it and the changes are merged into the original project. The
author of the changes will show up in the commit history of the original
project, so they get the credits.

As an alternative to forking, if one is interested in a project but has
a different vision, or that the maintainer has stopped working on it,
they can clone it and make their own version. This is a more independent
kind of fork because there is no longer intention to contribute back to
the original project.

Moreover, on Github there is no way to send private messages, which
forces people to interact publicly. If say you want someone to see and
reply to your comment in an issue post or pull request, you simply
mention them by `@someone`.

open research in mathematics
----------------------------

All this points to a promising direction of open research. A maths
project may have a wiki / collection of notes, the paper being written,
computer programs implementing the results etc. The issue tracker can
serve as a discussion forum about the project as well as a platform for
open review (bugs are analogous to mistakes, enhancements are possible
ways of improving the main results etc.), and anyone can make their own
version of the project, and (optionally) contribute back by making pull
requests, which will also be openly reviewed. One may want to add an
extra \"review this project\" functionality, so that people can comment
on the original project like they do in a pull request. This may or may
not be necessary, as anyone can make comments or point out mistakes in
the issue tracker.

One may doubt this model due to concerns of credits because work in
progress is available to anyone. Well, since all the contributions are
trackable in project commit history and public discussions in issues and
pull request reviews, there is in fact *less* room for cheating than the
current model in academia, where scooping can happen without any
witnesses. What we need is a platform with a good amount of trust like
arXiv, so that the open research community honours (and can not ignore)
the commit history, and the chance of mis-attribution can be reduced to
minimum.

Compared to the academic model, open research also has the following
advantages:

-   Anyone in the world with Internet access will have a chance to
    participate in research, whether they are affiliated to a
    university, have the financial means to attend conferences, or are
    colleagues of one of the handful experts in a specific field.
-   The problem of replicating / understanding maths results will be
    solved, as people help each other out. This will also remove the
    burden of answering queries about one\'s research. For example, say
    one has a project \"Understanding the fancy results in \[paper
    name\]\", they write up some initial notes but get stuck
    understanding certain arguments. In this case they can simply post
    the questions on the issue tracker, and anyone who knows the answer,
    or just has a speculation can participate in the discussion. In the
    end the problem may be resolved without the authors of the paper
    being bothered, who may be too busy to answer.
-   Similarly, the burden of peer review can also be shifted from a few
    appointed reviewers to the crowd.

related readings
----------------

-   [The Cathedral and the Bazaar by Eric Raymond](http://www.catb.org/esr/writings/cathedral-bazaar/)
-   [Doing sience online by Michael Nielson](http://michaelnielsen.org/blog/doing-science-online/)
-   [Is massively collaborative mathematics possible? by Timothy Gowers](https://gowers.wordpress.com/2009/01/27/is-massively-collaborative-mathematics-possible/)

[^feedback]: Please send your comments to my email address - I am still looking for ways to add a comment functionality to this website.
