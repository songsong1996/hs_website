---
title: "DNN Foundation"
description: "Reading notes on Deep Neural Network fundamentals"
summary: "Notes on DNN foundation concepts from Fundamentals of Deep Learning book"
date: 2026-07-05T00:00:00Z
lastmod: 2026-07-05T00:00:00Z
draft: false
weight: 30
toc: true
categories: ["machine-learning", "deep-learning"]
tags: ["dnn", "regularization", "dropout", "overfitting"]
contributors: []
params:
  seo:
    title: "DNN Foundation Notes"
    description: "Deep learning fundamentals and techniques"
---

## Overview

This document contains notes on DNN foundation concepts while reading *Fundamentals of Deep Learning*. Rather than covering all classic ML books, this focuses on quickly recapping essential DNN knowledge.

## Preventing Overfitting in DNNs

### Regularization

Regularization adds a penalty term to the loss function to discourage large weights: $\lambda \cdot f(w)$ added to the error term. As the components of $w$ grow larger, $f(w)$ increases. The most commonly used approaches are L1 and L2 regularization.

#### L1 Regularization

- Formula: $\lambda |w|$
- Encourages weights to become exactly zero
- Useful for feature selection to understand which features contribute to decisions

#### L2 Regularization

- Formula: $\lambda \frac{1}{2} w^2$
- During gradient descent, weights decay linearly toward zero (asymptotically approaching zero without collapsing)
- **Why weight decay occurs:**

$$\begin{align}
L &= \text{Error} + \lambda \frac{1}{2} w^2 \\
\frac{\partial L}{\partial w} &= \frac{\partial \text{Error}}{\partial w} + \lambda w \\
\text{Backpropagation: } w &= w - \eta \left(\frac{\partial \text{Error}}{\partial w} + \lambda w\right) \\
&= (1- \eta \lambda) w - \eta \frac{\partial \text{Error}}{\partial w}
\end{align}$$

With L2 regularization, minimizing the loss function requires both finding the error minimum (the bowl) and keeping weights small.

**Reference:** [Regularization explained](https://explained.ai/regularization/)

### Max Norm Constraints

Restricts the magnitude of $\theta$ to prevent weights from becoming too large by enforcing an absolute upper bound.

### Dropout

Dropout is a regularization technique that randomly deactivates neurons during training:

- During training, neurons remain active with probability $p$
- Forces the network to be accurate even with missing information
- Prevents over-dependence on any single neuron

**Inverted Dropout:** To ensure test-time outputs match expected training-time outputs, we need to consider scaling. If dropout rate is $p$:

$$E[\text{output}] = px + (1-p) \cdot 0 = px$$

Without adjustment, this causes a mismatch at test time. **Inverted dropout** solves this by scaling active neuron outputs by $\frac{1}{p}$ during training:

$$E[\text{output}] = \frac{x}{p} \cdot p + (1-p) \cdot 0 = x$$

This ensures outputs remain consistent between training and testing without post-hoc scaling. 