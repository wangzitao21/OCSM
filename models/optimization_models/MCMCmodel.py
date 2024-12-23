import numpy as np
import scipy.stats as st

def genex(range_, N=1):
    Npar = range_.shape[0]
    x = np.empty((Npar, N))
    x[:] = np.nan
    for i in range(N):
        x[:,i] = range_[:,0] + (range_[:,1] - range_[:,0]) * np.random.rand(Npar)
    return x

def likehood(k, obs, forwardmodel):
    y = forwardmodel(k)
    return st.multivariate_normal.pdf(obs, mean=y, cov=np.cov([obs, y], rowvar=False), allow_singular=True)

def prior(x, range_):
    for i in range(x.shape[0]):
        if (x[i] < range_[i, 0]) or (x[i] > range_[i, 1]):
            return 0
    return 1

def acceptance_rule(x,x_new):
    if x_new > x:
        return True
    else:
        accept = np.random.uniform(0, 1)
        return (accept < (x-x_new))

def metropolis_hastings(x, obs, sd, q, range_, forwardmodel, *,
                        prior=prior, likelihood=likehood, acceptance_rule=acceptance_rule,
                       ):
    x = np.array(x)
    m = x.shape[0]
    obs = np.array(obs)

    transition_model = lambda x: np.random.normal(x, sd, (m,))

    accepted, rejected = [], []
    scores = []
    for i in range(q):
        x_new = transition_model(x)
        x_lik = likelihood(x, obs, forwardmodel)
        x_new_lik = likelihood(x_new, obs, forwardmodel)
        if (acceptance_rule(x_lik + np.log(prior(x, range_)), x_new_lik + np.log(prior(x_new, range_)))):            
            x = x_new
            accepted.append(x_new)
            scores.append(x_new_lik)
        else:
            rejected.append(x_new)
        if i % 200 == 0:
            print('已完成:{:.2f}%'.format((i/q) * 100))

    return np.array(accepted), np.array(rejected), np.array(scores)