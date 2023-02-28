# Contains different explorative notebooks I've used to learn this stuff.

*ok maybe it won't contain anything but images - I've never forked a repository before, which I think I should've done instead of just cloning it like I did. It will be messy adding all the files here at this point. https://github.com/SPFlow/SPFlow*

Here's a discretized pdf of a normal distribution, constructed as a mixture of non-overlapping uniform distributions, created with `fit_spn_to_gaussian_eps(mean = 0, sd = 1, eps = 0.05, scope = 0)`.

![alt text](iterative%20subdivision.png)

This can be used again to construct multivariate distributions, by taking the products and weighted sums of the pdfs of the individual variables.

Below is a plot of increasingly discretized pdfs of a normal distribution, using no discretization, discretization of one axis, and discretization of both axes, created with some variant of `fit_spn_to_2d_gaussian(np.array([0.3, -0.5]), np.array([[1, 0.4],[0.4, 2]]), 0.03, [0, 1])`.

![alt text](increasingly%20discretized%20pdf.png)

They produce increasingly inaccurate approximations of the original pdf, but marginalization, conditioning, map estimates, moments, KL divergece, (conditional) likelihood and other operations can be performed on them easily with zero knowledge of gaussians, which might be useful in some cases.