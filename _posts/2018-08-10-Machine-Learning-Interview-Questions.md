---
title: "Machine Learning Interview Questions"
mathjax: true
---

1. Why does logistic regression use sigmoid function:
- It is differentiable
- Map the output to [0,1]
- In section 4.2 of Pattern Recognition and Machine Learning (Springer 2006), Bishop shows that the logit arises naturally as the form of the posterior probability distribution in a Bayesian treatment of two-class classification. He then goes on to show that the same holds for discretely distributed features, as well as a subset of the family of exponential distributions. For multi-class classification the logit generalizes to the normalized exponential or softmax function.

2. Newton Methods v.s. Gradient Descent
