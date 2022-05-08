---
layout: post
title: "Stratified Sampling Analysis"
subtitle: "Stratified Sampling is another method to decrease variance when doing MC"
date: 2020-01-26 23:45:13 -0400
background: '/img/stratified_sampling/stock.jpg'
---

# ORIE 5582: Monte Carlo Methods in Financial Engineering, Spring 2022
## Project 4 Stratified Sampling



```python
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
%matplotlib inline
```


Consider a discretely monitored Asian call option, with parameters $S(0) = 40$, $r=0.02$, $\sigma = 0.4$, $K = 35$, and $T = 1$ and time points $0 < t_1 < t_2 < \cdots < t_m = T$ with $m = 100$ and $t_i = iT/m$ for $i = 1, \ldots, m$.

For all parts,we'll use 10,000 replications (sample paths) and report a point estimate and 95\% confidence interval for the option price.

## Part 1. Standard Estimation of the option price



```python
# Initialize parameters.
S0 = 40; r = 0.02; sigma = 0.4; K = 35; T = 1; m = 100; reps = 10000 

# Simulate paths for stock prices
Z = np.random.normal(0, 1, [reps, m+1]); Z[:,0] = 0; # standard normal RVs
R = sigma*np.sqrt(T/m)*Z + (r - sigma**2/2)*(T/m); R[:,0] = 0 # log returns
S = S0*np.exp(np.cumsum(R, axis=1))# cumulative sum over time

# Estimate the option price
standard_payoff = np.exp(-r*T)*np.maximum(np.mean(S[:,1:],axis=1) - K, 0)
standard_price = np.mean(standard_payoff)

# Compute the confidence interval
z_alpha = 1.96 # quantile for 95% confidence
standard_std = np.std(standard_payoff, ddof=1)/np.sqrt(reps) # sample standard deviation
standard_lb = standard_price - z_alpha*standard_std # lower bound of CI
standard_ub = standard_price + z_alpha*standard_std # upper bound of CI

# Show results
print("The point estimator from standard Monte Carlo is %.3f." % standard_price)
print("The approximate standard deviation of the estimator is %.3f." % standard_std)
print("The approximate 95%% confidence interval is [%.3f, %.3f]." % (standard_lb, standard_ub))
```

    The point estimator from standard Monte Carlo is 6.741.
    The approximate standard deviation of the estimator is 0.081.
    The approximate 95% confidence interval is [6.583, 6.900].


## Part 2. Stratified Samplings on $S(T)$

In this part we'll use terminal stratification on $S(T)$ with proportional stratification and $d = 10$ equiprobable strata. To construct the confidence interval, we use the approach in which we calculate the sample variance within each stratum. Meanwhile, we'll plot 8 sample paths, one from each stratum.


```python
# Initialize parameters.
d = 10; all_ts = [i*T/m for i in range(m+1)]

# Define the function to generate brownian bridges
def generate_brownian_bridges(all_ts,W_T):
    m = len(all_ts)-1; T = all_ts[-1]; reps = W_T.size; dt = T/m;
    W = np.zeros((reps,m+1)); W[:,0] = 0; W[:,m] = W_T.reshape(-1)
    # Generate paths using conditional sampling
    for i in range(m-1):
        last_t = all_ts[i]; curr_t = all_ts[i+1];
        Zi = np.random.randn(reps)
        W[:,i+1] = (T - curr_t)/(T - last_t)*W[:,i] + dt/(T - last_t)*W[:,m] + np.sqrt((T - curr_t)*dt/(T - last_t))*Zi
    return W

# Generate the asset prices paths
ni = int(reps/d); S = {}; W = {};
U_T = (np.random.random([ni, d]) + np.tile(range(d),(ni,1)))/d
W_T = np.sqrt(T)*st.norm.ppf(U_T)
drift = (r - sigma**2/2)*np.tile(all_ts,(ni,1))
for i in range(d):
    W[i] = generate_brownian_bridges(all_ts,W_T[:,i])
    S[i] = S0*np.exp(drift + sigma*W[i])

# Estimate the option price
eqprob_payoff = np.zeros((ni,d))
for i in range(d):
    eqprob_payoff[:,i] = np.exp(-r*T)*np.maximum(np.mean(S[i][:,1:],axis=1) - K, 0)
eqprob_price = np.mean(eqprob_payoff)

# Compute the confidence interval
z_alpha = 1.96 # quantile for 95% confidence
eqprob_vari = np.var(eqprob_payoff, ddof=1, axis=0) # sample variance in each group
eqprob_std = np.sqrt(np.mean(eqprob_vari)/reps); # approximate SD of the estimator
eqprob_lb = eqprob_price - z_alpha*eqprob_std # lower bound of CI
eqprob_ub = eqprob_price + z_alpha*eqprob_std # upper bound of CI

# Show results
print("The point estimator from equiprobable stratification is %.3f." % eqprob_price)
print("The approximate standard deviation of the estimator is %.3f." % eqprob_std)
print("The approximate 95%% confidence interval is [%.3f, %.3f]." % (eqprob_lb, eqprob_ub))
print("The approximate variance is reduced by %.3f%%." % (100-100*eqprob_std**2/standard_std**2))
for i in range(d):
    plt.plot(all_ts,S[i][0,:],label=('Stratum {}'.format(i+1)))
plt.xlabel('Time'); plt.ylabel('Stock Price'); plt.legend(loc=2)
plt.title('Simulated Prices Paths from Each Stratum')
plt.show()
```

    The point estimator from equiprobable stratification is 6.621.
    The approximate standard deviation of the estimator is 0.045.
    The approximate 95% confidence interval is [6.533, 6.709].
    The approximate variance is reduced by 68.974%.



    
![png](/img/stratified_sampling/output_6_1.png)
    


## Part 3 Stratified Sampling on S(T) with optimal stratification
Next, we want to again use terminal stratification on $S(T)$, but with the `optimal' stratification. Again consider $d = 10$ equiprobable strata, but now first use $100$ samples in each strata to get an estimate $s_i^2$ for the variance of the option price conditioned on $S(T)\in A_i$. And then we split the remaining $9,000$ samples according the the optimal splitting scheme (based on estimates $s_i$), and recompute the mean and confidence interval.



```python
# Initialize parameters
n_pilot = 1000; n_formal = 9000; 

# Generate the pilot samples
pilot_ni = int(n_pilot/d); pilot_S = {}; pilot_W = {}; pilot_payoff = {}; si = [];
pilot_U_T = (np.random.random([pilot_ni, d]) + np.tile(range(d),(pilot_ni,1)))/d
pilot_W_T = np.sqrt(T)*st.norm.ppf(pilot_U_T)
pilot_drift = (r - sigma**2/2)*np.tile(all_ts,(pilot_ni,1))
for i in range(d):
    pilot_W[i] = generate_brownian_bridges(all_ts,pilot_W_T[:,i])
    pilot_S[i] = S0*np.exp(pilot_drift + sigma*pilot_W[i])
    pilot_payoff[i] = np.exp(-r*T)*np.maximum(np.mean(pilot_S[i][:,1:],axis=1) - K, 0)
    si.append(np.std(pilot_payoff[i], ddof=1))

# Generate formal samples
opt_ni = [int(si[i]/sum(si)*n_formal) for i in range(d)]; opt_ni[-1] = n_formal - sum(opt_ni[:-1]);
S = {}; W = {};
for i in range(d):
    U_T_i = np.random.random(opt_ni[i])/d + i/d
    W_T_i = np.sqrt(T)*st.norm.ppf(U_T_i)
    drift_i = (r - sigma**2/2)*np.tile(all_ts,(opt_ni[i],1))
    W[i] = generate_brownian_bridges(all_ts,W_T_i)
    S[i] = S0*np.exp(drift_i + sigma*W[i])

# Estimate the option price
optspl_payoff = {};
for i in range(d):
    optspl_payoff[i] = np.exp(-r*T)*np.maximum(np.mean(S[i][:,1:],axis=1) - K, 0)
optspl_price = sum([optspl_payoff[i].mean() for i in range(d)])/d

# Compute the confidence interval
optspl_vari = [np.var(optspl_payoff[i], ddof=1) for i in range(d)] # sample variance in each group
optspl_var = sum([optspl_vari[i]/(d**2*opt_ni[i]) for i in range(d)]); # approximate variance of the estimator
optspl_std = np.sqrt(optspl_var)
optspl_lb = optspl_price - z_alpha*optspl_std # lower bound of CI
optspl_ub = optspl_price + z_alpha*optspl_std # upper bound of CI

# Show results
print("The point estimator from the optimal sampling %.3f." % optspl_price)
print("The approximate standard deviation of the estimator is %.3f." % optspl_std)
print("The approximate 95%% confidence interval is [%.3f, %.3f]." % (optspl_lb, optspl_ub))
print("The approximate variance is reduced by %.3f%%." % (100-100*optspl_var/standard_std**2))
```

    The point estimator from the optimal sampling 6.624.
    The approximate standard deviation of the estimator is 0.041.
    The approximate 95% confidence interval is [6.543, 6.705].
    The approximate variance is reduced by 73.597%.


## Part 4 Stratified Sampling on Geometric mean of the sample path

In this part we'll use post-stratification on $\tilde{S}$ where $\tilde{S}$ is the geometric mean of the prices $S(t_1), \ldots, S(t_m)$. Again we use $d=10$ equiprobable strata. 
$$ \tilde{S} = \left(\prod_{i=1}^m S(t_i)\right)^{1/m} \sim S(0)\exp\left((r-\sigma^2/2)\bar{t} + \bar{\sigma}\mathcal{N}(0, \bar{t})\right), $$
where
$$ \bar{t} = \frac{1}{m}\sum_{i=1}^m t_i \quad \text{and} \quad \bar{\sigma}^2 = \frac{\sigma^2}{m^2\bar{t}}\sum_{i=1}^m [2(m-i) + 1] t_i. $$

When constructing our confidence interval, we use the fact that the variance of the post-stratification estimator is (asymptotically) equal to that of the proportional stratification estimator.



```python
# Compute parameters
t_bar = sum(all_ts[1:])/m; 
sigma_bar2 = sigma**2/(m**2)/t_bar*sum([(2*m-2*i-1)*all_ts[i+1] for i in range(m)]); 
sigma_bar = sigma_bar2**0.5;

# Generate samples of asset prices
Z = np.random.normal(0, 1, [reps, m+1]); Z[:,0] = 0; # standard normal RVs
R = sigma*np.sqrt(T/m)*Z + (r - sigma**2/2)*(T/m); R[:,0] = 0 # log returns
S = S0*np.exp(np.cumsum(R, axis=1))# cumulative sum over time

# Post stratification
indicators = np.zeros((reps,d)); stratums = {}; Si = [];
post_payoff = np.exp(-r*T)*np.maximum(np.mean(S[:,1:],axis=1) - K, 0)
S_tilde = st.mstats.gmean(S[:,1:], axis=1)
qt = lambda x: S0*np.exp((r - sigma**2/2)*t_bar+sigma_bar*np.sqrt(t_bar)*x)
for i in range(d):
    x_low = st.norm.ppf(i/d); x_up = st.norm.ppf((i+1)/d);
    indicators[:,i] = ((S_tilde>=qt(x_low))&(S_tilde<=qt(x_up)))
    stratums[i] = post_payoff[indicators[:,i].astype('bool')]
    Si.append(np.sum(indicators[:,i]*post_payoff))
Ni = np.sum(indicators, axis=0)
post_price = np.mean([Si[i]/Ni[i] for i in range(d)])

# Compute the confidence interval
var_mu_i = [np.var(stratums[i], ddof=1)/Ni[i] for i in range(d)]# approximate variance in each stratum
post_var = np.sum(var_mu_i)/(d**2); # approximate variance of the estimator
post_std = np.sqrt(post_var)
post_lb = post_price - z_alpha*post_std # lower bound of CI
post_ub = post_price + z_alpha*post_std # upper bound of CI

# Show results
print("The point estimator from post stratification is %.3f." % post_price)
print("The approximate standard deviation of the estimator is %.3f." % post_std)
print("The approximate 95%% confidence interval is [%.3f, %.3f]." % (post_lb, post_ub))
print("The approximate variance is reduced by %.3f%%." % (100-100*post_var/standard_std**2))
```

    The point estimator from post stratification is 6.681.
    The approximate standard deviation of the estimator is 0.023.
    The approximate 95% confidence interval is [6.637, 6.725].
    The approximate variance is reduced by 92.197%.


