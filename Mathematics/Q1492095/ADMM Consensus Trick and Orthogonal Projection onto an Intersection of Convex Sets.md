---
# YAML Page - https://hackmd.io/yaml-metadata
title: ADMM Consensus Trick and Orthogonal Projection onto an Intersection of Convex Sets
description: Using the ADMM Consensus Trick for the calculation of the projection onto an intersection of convex sets
tags: Math, Optimization, Convex Optimization, Convex Analysis
# disqus: hackmd
---

# ADMM Consensus Trick and Orthogonal Projection onto an Intersection of Convex Sets

## The ADMM Consensus Optimization

Given a problem in the form:

$$\begin{aligned}
\arg \min_{ \boldsymbol{x} } \quad & \sum_{i}^{n} {f}_{i} \left( \boldsymbol{x} \right) \\
\end{aligned}$$

By invoking auxiliary variables one could rewrite it as:

$$\begin{aligned}
\arg \min_{ \boldsymbol{x} } \quad & \sum_{i}^{n} {f}_{i} \left( \boldsymbol{x}_{i} \right) \\
\text{subject to} \quad & \boldsymbol{x}_{i} = \boldsymbol{z}, \; i = 1, 2, \ldots, n \\
\end{aligned}$$

Namely, it is a separable form with equality constraint on each variable.  
In matrix form it can be written as:

$$\begin{aligned}
\arg \min_{ \boldsymbol{x} } \quad & \sum_{i}^{n} {f}_{i} \left( \boldsymbol{x}_{i} \right) \\
\text{subject to} \quad & \boldsymbol{u} := \begin{bmatrix} \boldsymbol{x}_{1} \\ \boldsymbol{x}_{2} \\ \vdots \\ \boldsymbol{x}_{n} \end{bmatrix} = \begin{bmatrix} I \\ I \\ \vdots \\ I \end{bmatrix} \boldsymbol{z}
\end{aligned}$$

By defining $ {E}_{i} $ to be the matrix such that $ \boldsymbol{x}_{i} = {E}_{i} \boldsymbol{u} $, namely the selector of the appropriate sub vector form $ \boldsymbol{u} $ we can write the problem in the ADMM form (The Scaled ADMM form):

$$\begin{aligned}
\boldsymbol{u}^{k + 1} & = \arg \min_{ \boldsymbol{u} } \sum_{i}^{n} {f}_{i} \left( {E}_{i} \boldsymbol{u} \right) + \frac{\rho}{2} {\left\| {E}_{i} \boldsymbol{u} - \boldsymbol{z}^{k} + \boldsymbol{\lambda}^{k} \right\|}_{2}^{2} \\
\boldsymbol{z}^{k + 1} & = \arg \min_{ \boldsymbol{z} } \frac{\rho}{2} {\left\| \boldsymbol{u}^{k + 1} - \begin{bmatrix} I \\ I \\ \vdots \\ I \end{bmatrix} \boldsymbol{z} + \boldsymbol{\lambda}^{k} \right\|}_{2}^{2} \\
\boldsymbol{\lambda}^{k + 1} & = \boldsymbol{\lambda}^{k} + \boldsymbol{u}^{k + 1} - \begin{bmatrix} I \\ I \\ \vdots \\ I \end{bmatrix} \boldsymbol{z}^{k + 1} \\
\end{aligned}$$

Since the form is block separable it can be written in an element form:

$$\begin{aligned}
\boldsymbol{x}_{i}^{k + 1} & = \arg \min_{ \boldsymbol{x}_{i} } {f}_{i} \left( \boldsymbol{x}_{i} \right) + \frac{\rho}{2} {\left\| \boldsymbol{x}_{i} - \boldsymbol{z}^{k} + \boldsymbol{\lambda}_{i}^{k} \right\|}_{2}^{2}, & i = 1, 2, \ldots, n \\
\boldsymbol{z}^{k + 1} & = \arg \min_{ \boldsymbol{z} } \frac{\rho}{2} \sum_{i = 1}^{n} {\left\| \boldsymbol{x}_{i}^{k + 1} - \boldsymbol{z} + \boldsymbol{\lambda}_{i}^{k} \right\|}_{2}^{2} \\
\boldsymbol{\lambda}_{i}^{k + 1} & = \boldsymbol{\lambda}_{i}^{k} + \boldsymbol{x}_{i}^{k + 1} - \boldsymbol{z}^{k + 1}, & i = 1, 2, \ldots, n \\
\end{aligned}$$

**Remark**: Pay attention that $ \boldsymbol{z}_{k + 1} = \frac{1}{n} \sum_{i = 1}^{n} \boldsymbol{x}_{i}^{k + 1} + \boldsymbol{\lambda}_{i}^{k} $. Namely the mean value of the set $ { \left\{ \boldsymbol{x}_{i}^{k + 1} + \boldsymbol{\lambda}_{i}^{k} \right\} }_{i = 1}^{n} $.

The _Proximal Operator_ is given by $ \operatorname{Prox}_{\lambda f \left( \cdot \right)} \left( \boldsymbol{y} \right) = \arg \min_{\boldsymbol{x}} \frac{1}{2} {\left\| \boldsymbol{x} - \boldsymbol{y} \right\|}_{2}^{2} + \lambda f \left( \boldsymbol{x} \right) $ and simplifying the optimization for $ \boldsymbol{z}_{k + 1} $ one can write the above as:

$$\begin{aligned}
\boldsymbol{x}_{i}^{k + 1} & = \operatorname{Prox}_{\frac{1}{\rho} {f}_{i} \left( \cdot \right)} \left( \boldsymbol{z}^{k} - \boldsymbol{\lambda}_{i}^{k} \right), & i = 1, 2, \ldots, n \\
\boldsymbol{z}^{k + 1} & = \frac{1}{n} \sum_{i = 1}^{n} \boldsymbol{x}_{i}^{k + 1} + \boldsymbol{\lambda}_{i}^{k} \\
\boldsymbol{\lambda}_{i}^{k + 1} & = \boldsymbol{\lambda}_{i}^{k} + \boldsymbol{x}_{i}^{k + 1} - \boldsymbol{z}^{k + 1}, & i = 1, 2, \ldots, n \\
\end{aligned}$$

## Orthogonal Projection onto an Intersection of Convex Sets

The problem is given by:

$$\begin{aligned}
\arg \min_{ \boldsymbol{x} } \quad & \frac{1}{2} {\left\| \boldsymbol{x} - \boldsymbol{y} \right\|}_{2}^{2} \\
\text{subject to} \quad & \boldsymbol{x} \in \bigcap_i \mathcal{C}_{i}, \; i = 1, 2, \ldots n \\
\end{aligned}$$

Namely we're looking for orthogonal projection of $ \boldsymbol{y} $ onto the intersection of the sets $ {\left\{ \mathcal{C}_{i} \right\}}_{i = 1}^{n} $.

The problem could be rewritten as:

$$\begin{aligned}
\arg \min_{ \boldsymbol{x} } \quad & \frac{1}{2} {\left\| \boldsymbol{x} - \boldsymbol{y} \right\|}_{2}^{2} + \sum_{i}^{n} {\delta}_{\mathcal{C}_{i}} \left( \boldsymbol{x} \right)
\end{aligned}$$

Where $ {\delta}_{\mathcal{C}_{i}} (x) = \begin{cases} 0 & x \in \mathcal{C}_{i} \\ \infty & x \notin \mathcal{C}_{i}
\end{cases} $.

So we can use the ADMM Consensus Optimization by setting $ {f}_{1} \left( \boldsymbol{x} \right) = \frac{1}{2} {\left\| \boldsymbol{x} - \boldsymbol{y} \right\|}_{2}^{2} $ and $ {f}_{i} \left( \boldsymbol{x} \right) = {\delta}_{\mathcal{C}_{i - 1}} \left( \boldsymbol{x} \right), \; i = 2, 3, \ldots, n + 1 $.

In order to solve it we need the _Proximal Operator_ of each function. For $ {f}_{1} \left( \cdot \right) $ it is given by:

$$ \operatorname{Prox}_{ \frac{1}{\rho} {f}_{1} \left( \cdot \right)} \left( \boldsymbol{v} \right) = \arg \min_{ \boldsymbol{x} } \frac{1}{2} {\left\| \boldsymbol{x} - \boldsymbol{y} \right\|}_{2}^{2} + \frac{\rho}{2} {\left\| \boldsymbol{x} - \boldsymbol{v} \right\|}_{2}^{2} = \frac{\rho \boldsymbol{v} + \boldsymbol{y}}{1 + \rho} $$

For the rest the _Proximal Operator_ is the orthogonal projection:

$$ \operatorname{Prox}_{ \frac{1}{\rho} {f}_{i} \left( \cdot \right)} \left( \boldsymbol{v} \right) = \operatorname{Proj}_{ \mathcal{C}_{i - 1} } \left( \boldsymbol{v} \right), \; i = 2, 3, \ldots, n + 1 $$


## Reference

 *  [ELE 522 - Large Scale Optimization for Data Science - Alternating Direction Method of Multipliers](https://docdro.id/eqJh0qJ).  
    Slides 21-23 defines the Consensus Optimization.
 *  [Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers](https://ieeexplore.ieee.org/document/8186925).  
    Page 15 at equations 3.5-3.7 defines the Scaled ADMM form.
 *  

