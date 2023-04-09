# Trapezoid pdf calculation

$$ X \sim \mathcal{T}(s,e,\theta)$$

$$p(X=x) = \begin{cases}something & s<x\le e \\ 0 & \text{otherwise}\end{cases}$$


$p(x) = (x-s) \theta + \text{left height}$

$\int_s^e p(x)\; dx = 1 $

$\iff \int_s^e (x-s) \theta + \text{left height} \;dx = 1$

$\iff \int_s^e \theta x-s\theta + \text{left height} \;dx  = 1$

$\iff (0.5\theta x^2-s\theta x + \text{left height}x)|_{x=s}^{e}  = 1$

$\iff (0.5\theta e^2-s\theta e + \text{left height}e) - (0.5\theta s^2-s\theta s + \text{left height}s)  = 1$

$\iff 0.5\theta e^2-s\theta e + \text{left height}e - (-0.5\theta s^2 + \text{left height}s)  = 1$

$\iff 0.5\theta e^2-s\theta e + \text{left height}e + 0.5\theta s^2 - \text{left height}s  = 1$

$\iff 0.5\theta e^2-s\theta e + \text{left height}(e-s) + 0.5\theta s^2 -   = 1$

$\iff  \text{left height}(e-s)  = 1 - 0.5\theta e^2 + s\theta e - 0.5\theta s^2 = 1 - 0.5\theta(e-s)^2$

$\iff \text{left height} = \frac{1 - 0.5\theta(e-s)^2}{e-s}$

For what values of $\theta$ is the pdf valid? We know it must p(s) > 0 and p(e) > 0 must be non-negative.

$p(s) = (x - s) \theta + \frac{1 - 0.5\theta(e-s)^2}{e-s} > 0$

$p(e) = (e - s) \theta + \frac{1 - 0.5\theta(e-s)^2}{e-s} > 0$

This gives us that $|\theta|\leq 2/(e-s)^2$