---
title: "Sequential Recommender Systems Walk-Through"
description: "Detailed walkthrough of SRS, from before DNN era, to LLM Wave"
summary: "By modeling the same example, compare the difference of all SRS related methods"
date: 2025-08-10T16:27:22+02:00
lastmod: 2025-08-10T16:27:22+02:00
draft: false
weight: 50
categories: []
tags: []
contributors: []
pinned: false
homepage: false
seo:
  title: "" # custom title (optional)
  description: "" # custom description (recommended)
  canonical: "" # custom canonical URL (optional)
  noindex: false # false (default) or true
---

# Introduction
I am blessed to have a cat baby, and I enjoy baking a lot, however recently I started to lose weight. So my amazon records looks like . But question for MLE is, how model could model this. 
- Months 1–6: lots of cat stuff (litter mat, fountain, scratching post, treats, toys, carrier).
- Months 7–9: baking accessories (sheet pans, silicone mats, piping bags, cake turntable, mixer bowl).
- Last 2 weeks: workout kick (Theragun → whey protein → lifting gloves).

Appearantly my purchase have sequntial dependencies, which is unable to be captured by conventional recommendation systems, including collaborative filtering and content-based filtering, as they model/depict consumers by their interaction to the items and it's order-agnostic and it's also just pair-wise correlation between the consumer and items based on engagements (clicks, conversions etc).

And to model such sequential dependencies, there are a lot of different models from non-DNN, DNN, to latest LLM. This blog post summarizes how each model works for my user behaviors. For difficulties/challenges/characteristics of sequential recommender, please refer to [Sequential Recommender Systems: Challenges, Progress and Prospects](https://arxiv.org/pdf/2001.04830) 

# Traditional method
## Collaborative Filtering - Matrix-factorization CF
Matrix-Factorization CF learns a latent vector $p_u$ for each user. Although it's not frequency by category, but in practice, $p_u$ ends up to be:
$$p_u \approx (Q^⊤C_uQ+λI)^{−1} Q^⊤C_ur_u​$$

Here:
* $|I|$ is the size of items; $k$ is item factor dimensions.
* $Q \in \mathbb{R}^{|I|\times k}$: Item factor matrix. Row $i$ is the $k$-dim vector for item $i$. 
* $C_u \in \mathbb{R}^{|I|\times |I|}$: A **diagnol** confidence matrix (all non-diagnol = 0) of user $u$ for items. $c_{ui}$ is the condifence score of user $u$ with item $i$. It's normally derived from engagement data $r_{ui}$ (the interaction #clicks, #conversions): 
$$c_{ui} = 1 + \alpha r_{ui}$$

The term $(Q^⊤C_uQ+λI)^{−1}$ is some sort of normalization, while $C_u$ and $r_u$ are both related to consumer engagement, therefore the learned consumer representations end up engagement (clicks/conversions) weighted item vectors. So in my case, my representation will be closest to cat supplies embeddings. Therefore when predict the next items, it will likely end up **cat supplies**. 

# Before DNN Early Stage Sequence models
## Sequential pattern mining
Mine frequent patterns on sequence data, and then utlize the patterns for subsequent recommendations. Although simple and straightforward, but the patterns could be redundant. e.g. I am buying cat supplies monthly, while sometimes buying bake supplies in between. The pattern could be something like `cat_food` -> `cat_litter` -> `baking_supplies`. 



## Markov Chain Based approaches
### Basic Markov Chain
The hypothesis is future purchase depends only on previous $k$ presents. 




User sequence modeling has become common sense for recommendation system to model users. The reason to have it is very straightforward and intuitive, which is consumer's future behavior could be affected by their previous behaviors, aka sequential dependencies. And sequential dependencies is about preferences over time, which is unable to be captured by conventional recommendation systems, including collaborative filtering and content-based filtering, as traditional methods think <user, item> interaction as 


Well-thought-through product announcements will help increase feature awareness and engage users with new functionality. Just like [sharing your public roadmap](https://canny.io/blog/should-you-have-a-public-roadmap/), it's also a great way to let potential customers see that you're constantly improving.

## Further reading

- Read [How to announce product updates and features](https://canny.io/blog/announce-product-updates-features/)
