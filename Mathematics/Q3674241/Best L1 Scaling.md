# Best L1 Scaling

## Problem Formulation

The problem is given by:

$$\begin{aligned}
\arg \min_{a} {\left\| a \boldsymbol{x} - \boldsymbol{y} \right\|}_{1}
\end{aligned}$$

Where $ a \in \mathbb{R} $ and $ \boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^{n} $.

## Solution

Since the argument is scales the problem could easily be solved by 1D methods. Yet it could also be solved analytically.

Defining $ f \left( a \right) = {\left\| a \boldsymbol{x} - \boldsymbol{y} \right\|}_{1} $ then $ {f}^{'} \left( a \right) = \sum {x}_{i} \operatorname{sign} \left( a {x}_{i} - {y}_{i} \right) $. By defining $ {s}_{i} \left( a \right) = \operatorname{sign} \left( a {x}_{i} - {y}_{i} \right) $ we can write $ {f}^{'} \left( a \right) = \boldsymbol{s}^{T} \left( a \right) \boldsymbol{x} $ which a linear function of $ \boldsymbol{x} $. Moreover, the function is piece wise constant with *breaking points* / *junction points* at $ a = \frac{ {y}_{i} }{ {x}_{i} } $.

Then there exist $ k $ such that $ \boldsymbol{s}^{T} \left( {a}_{k} - \epsilon \right) \boldsymbol{x} < 0 $ and $ \boldsymbol{s}^{T} \left( {a}_{k} + \epsilon \right) \boldsymbol{x} > 0 $ which is the minimizer of $ f \left( a \right) = {\left\| a \boldsymbol{x} - \boldsymbol{y} \right\|}_{1} $.

![](./Figure0001.png)

In the above figure one could see that each *junction point* has a vertical range associated with it. The optimal argument is the junction point which contains zero in its range.

![](./Figure0002.png)

In the above figure one could see the value of the objective value. Indeed its minimum is attained on the junction point marked above.