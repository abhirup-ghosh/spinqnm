import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def find_kde(x):
    # Make super set of samples {x_i} U -{x_i}
    x_full = np.concatenate((x, -x))
    
    # Get the KDE from seaborn's distplot
    x_kde, y_kde = sns.distplot(x_full).get_lines()[0].get_data();
    
    # Create an interpolant
    from scipy.interpolate import InterpolatedUnivariateSpline as ius
    yx_kde = ius(x_kde, y_kde)
    
    # Evaluate only at the positive values 
    x_kde_even     = np.linspace(min(x_kde), max(x_kde), num=len(x_kde) + 1)
    x_kde_even_pos = [x_pos for x_pos in x_kde_even if x_pos >= 0]
    y_kde_even_pos = [yx_kde(x_i) for x_i in x_kde_even_pos]
    
    # Normalise distribution
    # normalisation  = yx_kde.integral(0, max(x_kde_even_pos))
    # print(normalisation)
    
    # return np.array(x_kde_even_pos), np.array(y_kde_even_pos) / normalisation
    
    # Since y_kde is normalised, the half kde is twice 
    plt.plot(np.array(x_kde_even_pos), np.array(y_kde_even_pos), 'C1-.')
    return np.array(x_kde_even_pos), 2.0 * np.array(y_kde_even_pos)

def combined_posterior(x1, x2):
    x1_post, y1_post = find_kde(x1)
    x2_post, y2_post = find_kde(x2)
    
    # Create the interpolants
    from scipy.interpolate import InterpolatedUnivariateSpline as ius
    yx1_kde = ius(x1_post, y1_post)
    yx2_kde = ius(x2_post, y2_post)
    
    # Now we multiply them:
    x_vals = np.linspace(0, max(x1_post), num=len(x1_post))
    
    y_vals = [yx1_kde(x) * yx2_kde(x) for x in x_vals]
    
    yx_combined = ius(x_vals, y_vals)
    
    print(yx_combined.integral(0, max(x1_post)))
    
    return np.array(x_vals), np.array(y_vals) / yx_combined.integral(0, max(x1_post))

def joint_likelihood_and_posterior(df1, df2, pr1max, pr2max):

    # priors ranges
    pr1range = [0, pr1max]
    pr2range = [0, pr2max]

    Nbins = 50
    
    lbins1 = np.linspace(pr1range[0], pr1range[1], Nbins)
    lintp1 = (lbins1[:-1] + lbins1[1:])/2.

    lbins2 = np.linspace(pr2range[0], pr2range[1], Nbins)
    lintp2 = (lbins2[:-1] + lbins2[1:])/2.

    dl1 = np.mean(np.diff(lbins1))
    dl2 = np.mean(np.diff(lbins2))
    
    # prior histograms
    prl1, lbins1 = np.histogram(np.random.uniform(pr1range[0], pr1range[1], 100000), bins=lbins1, density=True)
    prl2, lbins2 = np.histogram(np.random.uniform(pr2range[0], pr2range[1], 100000), bins=lbins1, density=True)
    
    # posterior histograms
    Pl1, lbins1 = np.histogram(df1['alphangr'], bins=lbins1, density=True)
    Pl2, lbins2 = np.histogram(df2['alphangr'], bins=lbins2, density=True)
    
    # likelihoods
    likel1 = Pl1/prl1
    likel2 = Pl2/prl2

    likel1 /= np.sum(likel1) * dl1
    likel2 /= np.sum(likel2) * dl2
    
    # HS: Abhirup, if dl1 \neq dl2, which one should we use?
    # joint likelihood
    likel = likel1*likel2
    likel /= np.sum(likel) * dl1 # chose dl1 because dl1=dl2

    # joint posterior
    Pl = likel * prl1 # chose one of the priors because they were identical
    Pl /= np.sum(Pl) * dl1 # chose dl1 because dl1=dl2
    
    return lintp1, lintp2, likel, Pl