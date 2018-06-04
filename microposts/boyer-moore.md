---
date: 2018-06-04
---

The [Boyer-Moore algorithm for finding the majority of a sequence of elements](https://en.wikipedia.org/wiki/Boyer–Moore_majority_vote_algorithm) falls in the category of "very clever algorithms".

    int majorityElement(vector<int>& xs) {
        int count = 0;
        int maj = xs[0];
        for (auto x : xs) {
            if (x == maj) count++;
            else if (count == 0) maj = x;
            else count--;
        }
        return maj;
    }

