# Crypto-Trade 📈
Crypto-Trade is a GitHub repository dedicated to analyzing the potential returns in the dynamic and ever-evolving cryptocurrency market

## Abstract
In this manuscript is explore a quantitative framework to maximize the amount of **USDT** in a given portfolio by taking advantage of it price variations of the **BTC** in the Binance exchange.

## Market dynamics

The value of one BTC could be model as a discrete time serie $`\left \{ X_t \right \}`$ of its price with respect the USDT over time. The price moves up and down fluctuating over time, these variations let us to buy BTC  are precisely what help us to maximize the amount of USDT available.
```math
{X}_{t_0}, {X}_{t_1}, \cdots, {X}_{t_i}, \cdots, {X}_{t_j}, \cdots ,{X}_{t_i}
```
We can take advantage of these variations to exchange [**buy**] BTC at a given price in USDT and then, after a while, exchange the BTC [**sell**] it for USDT at a much higher price to obtain a profit.

## Percentage of change

Suppose we want to use an amount $`{\beta}_{i}`$ of USDT to purchase a quantity of BTC. At the moment in which it is bought, at time $`t_i`$, the value of one BTC with respect to the USDT is $`{X}_{t_i}`$. Now, after some time $`t_j = t_i + \delta t`$, we want to sell the BTC's purchased, the price could change and have a different value $`{X}_{t_j}`$. calculating the percentage change $`phi_i`$ of the price variations at the point of purchase $`i`$ and point of sale $`j`$, it is estimated the value returned by investing $`\beta_i`$ USDT as the following product
```math
{\beta}_{i}{\phi}_{i} = {\beta}_{i} \left [ \frac {X_{t_j}}{X_{t_i}} - 1 \right ]_{{t}_{i}<{t}_{j}}
    %\geq \alpha \quad \alpha \in \left ( 0,1 \right ]
```
From the expression it can be seen that: a positive return $`{\phi}_{i} > 0`$ occurs when $`X_{t_j} > X_{t_i}`$, a negative return $`{\phi}_{i} < 0`$ occurs when $`X_{t_j} > X_{t_i}`$ and no return $`{\phi}_{i} = 0`$ occurs when $`X_{t_j} = X_{t_i}`$.

## Cumulative returns
Following an operation like the previous one, but now given the total amount $`\beta_0`$ of USDT available, we only invest a fraction $`{f}_{i} \in \left ( 0,1 \right ]`$ of it. The total amount in our portfolio after one operation is given by $`\beta_{i+1} = \beta_{i-1}\left ( 1 + {f}_{i} \phi_i \right )`$. After successive trades the amount of USDT in our portfolio is calculated as a successive product of the trade results.
```math
\beta_{m} = \beta_{0}\left ( 1 + f_1 \phi_1 \right )\left ( 1 + f_2 \phi_2 \right ) \cdots \left ( 1 + f_i \phi_i \right ) \cdots \left ( 1 + f_m \phi_m \right )
```
```math
\beta_{m} = \beta_{0} \prod_{i=1}^{m} \left ( 1 + f_i \phi_i \right )
```
This expression (\ref{eq:resources}) computes the total amount available of USDT after m trades. The parameters that can be controlled in the model are the fraction $`f_i`$ of the invest resources $`\beta_*`$ in each trade and although it's not obvious, the time $`\delta t`$ between buy and sell actions. On the other hand, the percentage change $`\phi_i`$ will be considered as a random variable, which in principle can be biased by the way in which positions are placed.

## Growth metric
As a metric to evaluate growth, the natural logarithm (a strictly increasing function) of the ratio $`{\beta_m}/{\beta_0}`$ between the final and initial amounts of the portfolio multiplied by a normalization constant $`{m}^{-1}`$ is given by
```math
G = \frac{1}{m} ln \left ( \frac{\beta_m}{\beta_0} \right )
```
Replacing the expression of the cumulative returns in the last equation we get that the metric ends up being the expected value of the fraction of resources available during each trade:
```math
G = \frac{1}{m} \sum_{i=1}^{m} ln \left ( 1 + f_i \phi_i \right )
```
```math
G = E\left [ ln \left ( 1 + f_i \phi_i \right ) \right ]
```

## Maximize growth
To maximize the growth metric we will consider a fixed fraction $f$ on each trade as a first approximation to the problem. We will calculate the value of with respect the moments of the probability distribution of the $`\phi_i`$ random variable. First we are going to expand as a Taylor series:

```math
\ln(1+x) = \sum_{n=1}^{\infty} (-1)^{n+1} \frac{x^n}{n}, \quad for \quad |x| < 1.
```

```math
G = E \left [ \sum_{n=1}^{\infty} (-1)^{n+1} \frac{\left ( f \phi \right )^n}{n}\right ], \quad for \quad |f \phi| < 1.
```
The optimal value for $`f`$ that maximizes $G$ must satisfies $`{dG}/{df}=0`$, Deriving and equating to zero, we obtain:
```math
E \left [ \phi_i \right ] - f E\left [ \phi_{i}^{2} \right ] + \cdots \approx 0
```
solving for f in the previous expression, we can identify that the optimal value of the fraction is the ratio between the first moment and the second moment of the distribution of the return percentage:
```math
f \approx \frac{E \left [ \phi_i \right ]}{E\left [ \phi_{i}^{2} \right ]}
```
Replacing the value of $`f`$ we compute the value of the growth metric for the optimal value:
```math
G = \frac{E \left [ \phi_i \right ]}{E\left [ \phi_{i}^{2} \right ]} E\left [ \phi_i \right ] - \frac{1}{2} {\left( \frac{E \left [ \phi_i \right ]}{E\left [ \phi_{i}^{2} \right ]} \right)}^{2} E\left [ \phi_{i}^{2} \right ] + \cdot \cdot \cdot
```
```math
G = \frac{{E \left [ \phi_i \right ]}^{2}}{E\left [ \phi_{i}^{2} \right ]} - \frac{1}{2} \frac{{E \left [ \phi_i \right ]}^{2}}{E\left [ \phi_{i}^{2} \right ]} + \cdot \cdot \cdot
```
```math
G \approx \frac{1}{2} {\left( \frac{E \left [ \phi_i \right ]}{ \sqrt{E\left [ \phi_{i}^{2} \right ]}}\right)}^{2}
```
