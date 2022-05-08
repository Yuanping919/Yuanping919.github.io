---
layout: post
title: "Control Variate Sampling Analysis"
subtitle: "Basic Ideas on Control Variates"
date: 2020-01-26 23:45:13 -0400
background: '/img/stratified_sampling/stock2.jpg'
---


# ORIE 5582: Monte Carlo Methods in Financial Engineering, Spring 2022
## Project 3 Contral Variate Sampling



```python
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy.matlib as mat
%matplotlib inline
```

We know how complex SDEs can be approximately simulated via the
Euler method, which discretizes time and simulates a stochastic process as it evolves using
the dynamics described by the SDE at the current time. In this question, we will use this
approach to simulate

### Part 1 
Consider a *square-root interest-rate model*, where the interest rate follows the SDE:
\begin{equation*}
dr(t) = a(b-r(t))dt + \sigma \sqrt{r(t)}dW(t),
\end{equation*}
where $\{W(t): t \geq 0\}$ is a standard Brownian motion.
The process is *mean-reverting* in the sense that $r(t)$ is pulled toward $b$. 
The constant $b$ can be thought of as the long-run interest rate and $a$ can be thought of as the speed at which $r(t)$ reverts to $b$.
Choose the parameters as:
$$ T = 1, \: a = 0.2, \: b = 0.1, \: \sigma = 0.05, \: r(0) = 0.05. $$
We'll use standard Monte Carlo simulation with an Euler approximation to estimate the price of a
bond with payoff 1 and maturity $T$, i.e.,
$$ B_0 = \mathbb{E}(e^{-\int_{0}^{T}r(u)\,du}). $$
We'll use 1,000 macroreplications with a discretized time step size of 0.01 and provide a 95\% confidence interval for the bond price.



```python
# Initialize parameters.
T = 1; a = 0.2; b = 0.1; sigma = 0.05; r0 = 0.05; reps = 1000;

# Define estimation function.
def estimate_bond_price(T, a, b, sigma, r0, reps, step_size):
    
    # Initialize
    m = int(T/step_size); # number of timeintervals
    R = np.zeros((reps,m+1)); R[:,0] = r0; # paths of interest rates
    
    # Simulate interest rates paths
    Z = np.random.normal(0, 1, [reps, m]); # standard normal RVs
    for i in range(m):
        R[:,i+1] = R[:,i] + (a*b - a*R[:,i])*step_size + sigma*np.sqrt(np.abs(R[:,i]))*np.sqrt(step_size)*Z[:,i]
    discounted = np.prod(np.exp(-R[:,1:]*step_size), axis = 1)
    
    # Estimate of the bond price
    z_alpha = 1.96 # quantile for 95% confidence
    bond_price = np.mean(discounted)
    bond_std = np.std(discounted, ddof = 1)/np.sqrt(reps)
    bond_lb = bond_price - z_alpha*bond_std # lower bound of CI
    bond_ub = bond_price + z_alpha*bond_std # upper bound of CI
    
    # Print results
    print("The point estimator of the bond price is %.3f." % bond_price)
    print("The approximate 95%% confidence interval is [%.3f, %.3f]." % (bond_lb, bond_ub))
    print("The approximate variance of the estimator is %.6f\n" % (bond_std**2))
    return bond_price, discounted, R

# Test for different step sizes
print('For step size 0.01:')
bond_price1, discounted1, R1 = estimate_bond_price(T, a, b, sigma, r0, reps, 0.01)
print('For step size 0.001:')
bond_price2, discounted2, R2 = estimate_bond_price(T, a, b, sigma, r0, reps, 0.001)
```

    For step size 0.01:
    The point estimator of the bond price is 0.947.
    The approximate 95% confidence interval is [0.946, 0.947].
    The approximate variance of the estimator is 0.000000
    
    For step size 0.001:
    The point estimator of the bond price is 0.947.
    The approximate 95% confidence interval is [0.946, 0.947].
    The approximate variance of the estimator is 0.000000
    


## Part 2
We assume the stock price follows a two-factor stochastic volatility SDE:
\begin{eqnarray*}
dS(t) &=& rS(t)dt + \sqrt{V(t)}S(t) dX_1(t)\\
dV(t) &=& a(b-V(t))dt + \sigma \sqrt{V(t)}dX_2(t)
\end{eqnarray*}
where $X = \{(X_1(t), X_2(t)): t \geq 0\}$ is a two-dimensional Brownian motion with zero drift and covariance matrix
$$ \Sigma = \begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}. $$
Choose the parameters as:
$$ T = 1,\: r = 0.05, \: a = 0.2, \: b = 0.1, \: \sigma = 0.1, \: \rho = 0.75, \: S(0) = 80, \: K = 80. $$
We assume that the initial value of the volatility process is the same as
its long-term mean, i.e., $V(0) = b$.
We'll use standard Monte Carlo simulation with Euler approximations on $S$ and $V$ to estimate the price of a European call option, with strike $K$ and maturity $T$.



```python
# Initialize parameters.
T = 1; r = 0.05; a =0.2; b = 0.1; sigma = 0.1; rho = 0.75; S0 = 80; K = 80; reps = 10000; step_size = 0.05;
m = int(T/step_size) # number of steps
V = np.zeros((reps,m+1)); V[:,0] = b; # paths of the volatility
S = np.zeros((reps,m+1)); S[:,0] = S0; # paths of the volatility

# Simulate paths for stock prices
Z1 = np.random.normal(0, 1, [reps, m]) # standard normal RVs
Z2 = rho*Z1 + np.sqrt(1-rho**2)*np.random.normal(0, 1, [reps, m])
for i in range(m):
    S[:,i+1] = S[:,i]*np.exp((r - V[:,i]/2)*step_size + np.sqrt(np.abs(V[:,i])*step_size)*Z1[:,i])
    V[:,i+1] = V[:,i] + (a*b - a*V[:,i])*step_size + sigma*np.sqrt(np.abs(V[:,i])*step_size)*Z2[:,i]
    
# Estimate the option price
eucall_payoff = np.exp(-r*T)*np.maximum((S[:,m] - K), 0)
eucall_price = np.mean(eucall_payoff)

# Compute the confidence interval
z_alpha = 1.96 # quantile for 95% confidence
eucall_std = np.std(eucall_payoff, ddof=1)/np.sqrt(reps) # estimated standard deviation of the estimator
eucall_lb = eucall_price - z_alpha*eucall_std # lower bound of CI
eucall_ub = eucall_price + z_alpha*eucall_std # upper bound of CI

# Show results
print("The point estimator from the model above is %.3f." % eucall_price)
print("The standard deviation of the estimator is approximately %.3f." % eucall_std)
print("The approximate 95%% confidence interval is [%.3f, %.3f]." % (eucall_lb, eucall_ub))
```

    The point estimator from the model above is 11.818.
    The standard deviation of the estimator is approximately 0.211.
    The approximate 95% confidence interval is [11.405, 12.231].


## Control Variate On Different Cases

We Consider a continuously monitored up-and-out call option as described in Question 2 of Homework 2, with parameters $K=80$, $b = 100$, $T=1$, $S(0) = 75$, $r = 0.045$ and $\sigma = 0.3$.
We then estimate the price of the option using standard Monte Carlo with 5,000 sample paths of the underlying asset $\{S(t): 0 \leq t \leq T\}$ with time discretized into intervals of width $2^{-10}$.

For each of the following control variates (to be used one at a time), We'll provide a point estimate and 95\% confidence interval for the price of the option as well as an estimate of the variance reduction compared to the standard Monte Carlo estimate.


```python
# Initialize parameters.
S0 = 75; r = 0.045; sigma = 0.3; K = 80; T = 1; b = 100; m = 2**10; reps = 5000;

# Simulate paths for stock prices
Z = np.random.normal(0, 1, [reps, m+1]); Z[:,0] = 0; # standard normal RVs
R = sigma*np.sqrt(T/m)*Z + (r - sigma**2/2)*(T/m); R[:,0] = 0 # log returns
S = S0*np.exp(np.cumsum(R, axis=1))# cumulative sum over time

# Estimate the option price
maxS = np.amax(S, axis=1) # maximum for each sample path 
exercise = np.multiply((maxS < b),(S[:,m] > K)) # indicator of whether the payoff is positive 
standard_payoff = np.exp(-r*T)*np.multiply(exercise,(S[:,m] - K))
standard_price = np.mean(standard_payoff)

# Compute the confidence interval
z_alpha = 1.96 # quantile for 95% confidence
standard_std = np.std(standard_payoff, ddof=1)/np.sqrt(reps) # estimated standard deviation of the estimator
standard_lb = standard_price - z_alpha*standard_std # lower bound of CI
standard_ub = standard_price + z_alpha*standard_std # upper bound of CI

# Show results
print("The point estimator from standard Monte Carlo is %.3f." % standard_price)
print("The standard deviation of the estimator is approximately %.3f." % standard_std)
print("The approximate 95%% confidence interval is [%.3f, %.3f]." % (standard_lb, standard_ub))
```

    The point estimator from standard Monte Carlo is 0.864.
    The standard deviation of the estimator is approximately 0.040.
    The approximate 95% confidence interval is [0.787, 0.942].


### 1 The price of the underlying asset at the time of expiration, i.e., $S(T)$.


```python
# Compute the control-variate estimate of the original option price
ST_payoff = S[:,m] # asset price at expiration
the_lambda1 = np.cov(np.vstack((standard_payoff,ST_payoff)))[0,1]/np.var(ST_payoff) # approximate optimal coefficient
cv1_payoff = standard_payoff - the_lambda1*(ST_payoff - S0*np.exp(r*T)) # control-variate estimate
cv1_price = np.mean(cv1_payoff)

# Compute the confidence interval
z_alpha = 1.96 # quantile for 95% confidence
cv1_std = np.std(cv1_payoff, ddof=1)/np.sqrt(reps) # estimated standard deviation of the estimator
cv1_lb = cv1_price - z_alpha*cv1_std # lower bound of CI
cv1_ub = cv1_price + z_alpha*cv1_std # upper bound of CI

# Show results
print("The point estimator with the asset price at expiration as the control variate is %.3f." % cv1_price)
print("The standard deviation of the estimator is approximately %.3f." % cv1_std)
print("The approximate 95%% confidence interval is [%.3f, %.3f]." % (cv1_lb, cv1_ub))
print("The approximate variance is reduced from %.6f to %.6f." % (standard_std**2, cv1_std**2))
```

    The point estimator with the asset price at expiration as the control variate is 0.870.
    The standard deviation of the estimator is approximately 0.039.
    The approximate 95% confidence interval is [0.794, 0.947].
    The approximate variance is reduced from 0.001563 to 0.001524.


## 2 The discounted payoff of a European call option with the same strike, maturity, and dynamics of the underlying asset, i.e., $e^{-rT}[S(T) - K]^+$.



```python
# Compute the European call option price analytically
d1 = (np.log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
call_price = S0*st.norm.cdf(d1) - K*np.exp(-r*T)*st.norm.cdf(d2)

# Compute the control-variate estimate of the original option price
call_payoff = np.exp(-r*T)*np.maximum(S[:,m] - K, 0) # european call payoffs
the_lambda2 = np.cov(np.vstack((standard_payoff,call_payoff)))[0,1]/np.var(call_payoff) # approximate optimal coefficient
cv2_payoff = standard_payoff - the_lambda2*(call_payoff - call_price) # control-variate estimate
cv2_price = np.mean(cv2_payoff)

# Compute the confidence interval
z_alpha = 1.96 # quantile for 95% confidence
cv2_std = np.std(cv2_payoff, ddof=1)/np.sqrt(reps) # estimated standard deviation of the estimator
cv2_lb = cv2_price - z_alpha*cv2_std # lower bound of CI
cv2_ub = cv2_price + z_alpha*cv2_std # upper bound of CI

# Show results
print("The point estimator with the european call price as the control variate is %.3f." % cv2_price)
print("The standard deviation of the estimator is approximately %.3f." % cv2_std)
print("The approximate 95%% confidence interval is [%.3f, %.3f]." % (cv2_lb, cv2_ub))
print("The approximate variance is reduced from %.6f to %.6f." % (standard_std**2, cv2_std**2))
```

    The point estimator with the european call price as the control variate is 0.866.
    The standard deviation of the estimator is approximately 0.040.
    The approximate 95% confidence interval is [0.788, 0.943].
    The approximate variance is reduced from 0.001563 to 0.001561.


## 3. An indicator for whether or not the barrier option has a positive payoff, i.e., 
$$ \mathbf{1}\{\max_{0 \leq t \leq T} S(t) < b \text{ and } S(T) > K \}.$$


```python
# Compute the probability to get positive payoff analytically
b_tilde = np.log(b/S0)/sigma
K_tilde = np.log(K/S0)/sigma
exe_prob = 1-2*st.norm.cdf(-b_tilde)-st.norm.cdf(K_tilde)+st.norm.cdf(K_tilde-2*b_tilde)

# Compute the control-variate estimate of the original option price
the_lambda3 = np.cov(np.vstack((standard_payoff,exercise)))[0,1]/np.var(exercise) # approximate optimal coefficient
cv3_payoff = standard_payoff - the_lambda3*(exercise - exe_prob) # control-variate estimate
cv3_price = np.mean(cv3_payoff)

# Compute the confidence interval
z_alpha = 1.96 # quantile for 95% confidence
cv3_std = np.std(cv3_payoff, ddof=1)/np.sqrt(reps) # estimated standard deviation of the estimator
cv3_lb = cv3_price - z_alpha*cv3_std # lower bound of CI
cv3_ub = cv3_price + z_alpha*cv3_std # upper bound of CI

# Show results
print("The point estimator with the exercise probability as the control variate is %.3f." % cv3_price)
print("The standard deviation of the estimator is approximately %.3f." % cv3_std)
print("The approximate 95%% confidence interval is [%.3f, %.3f]." % (cv3_lb, cv3_ub))
print("The approximate variance is reduced from %.6f to %.6f." % (standard_std**2, cv3_std**2))
```

    The point estimator with the exercise probability as the control variate is 0.798.
    The standard deviation of the estimator is approximately 0.024.
    The approximate 95% confidence interval is [0.751, 0.845].
    The approximate variance is reduced from 0.001563 to 0.000577.


<font color = black>When $\lambda$ is estimated close to its optimal value, the variance of the estimator $Var(Y)$ should be reduced to $Var(Y)(1-\rho^2_{xy})$, where $\rho_{xy}$ denotes the correlation between $Y$ and the control variate.  
Among all the control variates chosen above, the exercise probability in (c) has a higher correlation with the barrier option payoff. As is shown above, the estimated correlation coefficient for control variates in (c) is 0.81, much higher than those in (a) and (b). Thus, the variance should be reduced more.</font>


```python
np.corrcoef(np.vstack((standard_payoff,ST_payoff,call_payoff,exercise)))
```




    array([[ 1.        ,  0.15189176,  0.03695851,  0.78676706],
           [ 0.15189176,  1.        ,  0.89265425,  0.13531766],
           [ 0.03695851,  0.89265425,  1.        , -0.0410022 ],
           [ 0.78676706,  0.13531766, -0.0410022 ,  1.        ]])




```python

```
