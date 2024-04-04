# %% [markdown]
"""
# Big O Notation
"""
# %%
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
# %%
# Defining the $n$ for the sequence of random variables $X_n$.
n = range(1, 1_000)
# Defining the $\varepsilon$
e = 0.001
# Defining a set of $\delta$ values to test on the given $\varepsilon$.
delta_e = np.linspace(.01, 1, 10)
# %%
def get_p_value(n, e, delta_e, sim=2_000):
    # `sim` realization that later will be use to estimate the prob of
    # $|X_n| > \delta$.
    X = stats.norm.rvs(size=(max(n), sim))
    p_res = np.nan * np.zeros((len(n), len(delta_e)))
    for i in range(len(n)):
        iter_i = n[i]
        # Calculating the sequence of random variables $|X_n|$.
        X_n = np.mean(X[:iter_i, :], axis=0)
        abs_X_n = np.abs(X_n)
        for j in range(len(delta_e)):
            # Calculating the probability of $|X_n| > \delta$ for a delta value.
            # Iterating for different values of $delta$ to find one that
            # satisfies the condition for the Big O notation.
            p = np.mean(abs_X_n > delta_e[j])
            p_res[i, j] = p
    return p_res
# %%
res = get_p_value(n, e, delta_e) 
# %%
fig, ax = plt.subplots()
for i, d in enumerate(delta_e):
    ax.plot(n, res[:, i], label=f'$\delta = {d:0.2f}$')
ax.axhline(e, color='red', linestyle='--', label=f'$\epsilon = {e:0.2f}$')
ax.set_ylim(e - 0.03, e + 0.01)
ax.set_xlabel('n')
ax.set_ylabel('p-value')
ax.legend(loc = 'lower left')
ax.set_title('Testing different values of $\delta$ for the given $\epsilon$')  
plt.show()
