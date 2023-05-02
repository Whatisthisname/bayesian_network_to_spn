# Contains different explorative notebooks I've used to learn and build this extension

Defining a PGM is pretty easy. Here's a simple example. 

```python	
A = "A" @ noise  # noise is an i.i.d. gaussian with mean 0 and variance 1. The @ operator assigns a name to the variable.
B = "B" @ (A + 1.2*noise + 1) # They can be composed with other variables, and more noise can be added.
C = "C" @ (A + noise + 4)
D = "D" @ (B + 0.3*C + noise - 5)

K = "K" @ (noise - 6)
J = "J" @ (0.5 * K - 3 + 0.2*noise)
A & K # Here, we join the two pgms, which have no connecting edges between them.
display(A.get_graph(detailed=False))
display(A.get_graph(detailed=True))
```

![alt text](pgm%20graph.png)

We can construct an SPN approximating this multivariate normal with the following code.

```python
spn = spnhelp.pgm_to_spn(A, eps = 0.1) # eps can be lowered to get a better approximation
spnhelp.plot_marginals(spn, A)
print(get_number_of_nodes(spn), "nodes, with depth of", get_depth(spn))
```

![alt text](marginals.png)

Now, let's try and sample from it to see how well it captures the covariance:

```python	
scope = A.get_scope(across_factors=True)
print(K.get_Σ()) # printing the true covariance 
print(A.get_Σ()) # printing the true covariance
samples = spnhelp.sample_from_spn(spn, 10000) # sampling from the spn
cov = np.cov(samples, rowvar=False).round(1) # computing covariance of the samples
print(pd.DataFrame(cov, index=scope, columns=scope))
```

![alt text](covariance_sampled.png)

The covariance is pretty close to the true covariance. It would be better with a lower epsilon, but this is just a toy example and it grows exponentially in size with decreasing epsilons. The mean will be captured perfectly (not shown here).





# Approximating a gaussian

One can approximate gaussians by a mixture of uniforms or slopyforms, both with disjoint support. The library allows one to give error bounds that the approximation should fall within, here are two kinds shown below. A slopyform is a uniform distribution with a likelihood proportional to the input.

![alt text](approximation%20error%20bounds.png)

Multiplying two of these approximations together gives us our first simple SPN of a factorized model, using the following Bayesian network:
```python
A = "A" @ noise
B = "B" @ noise
A & B
```

![alt text](2d%20approximations.png)

The last three are constructed with the slopyform approximation, which can be seen to give a lot better results.

Let's introduce a dependence between the two variables in the following way: we can see that more components might be needs
```python
A = "A" @ noise
B = "B" @ (0.5*A + noise)
```

![alt text](2d%20approximations%2C%20dependent%20and%20bad.png)
And with more components:
![alt text](2d%20approximations%2C%20dependent%20and%20good.png)


We can also measure how well the approximation fit their target distribution with the KL divergence. Here, we see that the slopyform approximation is a better fit than the uniform approximation, requiring much fewer components to achieve even lower KL divergence.

![alt text](kl%20divs.png)
