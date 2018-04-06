---
template: oldpost
title: jst
date: 2015-04-02
comments: true
archive: false
---
jst = juggling skill tree

If you have ever played a computer role playing game, you may have
noticed the protagonist sometimes has a skill "tree" (most of the time
it is actually a directed acyclic graph), where certain skills leads to
others. For example,
[here](http://hydra-media.cursecdn.com/diablo.gamepedia.com/3/37/Sorceress_Skill_Trees_%28Diablo_II%29.png?version=b74b3d4097ef7ad4e26ebee0dcf33d01)
is the skill tree of sorceress in [Diablo
II](https://en.wikipedia.org/wiki/Diablo_II).

Now suppose our hero embarks on a quest for learning all the possible
juggling patterns. Everyone would agree she should start with cascade,
the simplest nontrivial 3-ball pattern, but what afterwards? A few other
accessible patterns for beginners are juggler's tennis, two in one and
even reverse cascade, but what to learn after that? The encyclopeadic
[Library of Juggling](http://libraryofjuggling.com/) serves as a good
guide, as it records more than 160 patterns, some of which very
aesthetically appealing. On this website almost all the patterns have a
"prerequisite" section, indicating what one should learn beforehand. I
have therefore written a script using [Python](http://python.org),
[BeautifulSoup](http://www.crummy.com/software/BeautifulSoup/) and
[pygraphviz](http://pygraphviz.github.io/) to generate a jst (graded by
difficulties, which is the leftmost column) from the Library of Juggling
(click the image for the full size): 

[![The juggling skill tree](../assets/resources/juggling.png){width="38em"}](../assets/resources/juggling.png)
