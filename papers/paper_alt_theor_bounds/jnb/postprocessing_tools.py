import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity

def read_relevant_columns(samples):
    """
    Takes a 'posterior_samples.dat' file (found in cbcBayes of each PE run) and 
    return the columns we are mostly interested in.
    """
    alpha_ngr = samples['alphangr']
    m1_source = samples['m2_source']
    m2_source = samples['m2_source']
    mf_source = samples['mf_source_evol']
    z = samples['redshift']
    
    return alpha_ngr, m1_source, m2_source, mf_source, z

def get_dimensionless_coupling_posterior(smp_l, smp_mass, p=4):
    """
    Combines the PE samples of l (smp_l), one of the masses (smp_mass) for 
    a given exponent p as:
    
    gamma = (l / smp_mass)^{p}
    
    """
    # Convert mass from solar masses to km
    Msun_to_km = 1.476

    smp_mass_geo = smp_mass * Msun_to_km
        
    return (smp_l / smp_mass_geo)**p

def get_threshold(smp_mass, thr_coef=1/2.):
    """
    Given the PE samples on a mass (smp_mass) we calculate the associated median,
    convert to km and multiply by a value thr_coed (between 0 and 1). This sets 
    the length scale below which our posterior in the coupling l should be below
    if we want to claim any constraint. The default values is 0.5, which was used 
    in other papers.
    """
    # Convert mass from solar masses to km
    Msun_to_km = 1.476
    
    smp_mass_geo = np.median(smp_mass) * Msun_to_km
    
    return thr_coef * m_in_km

def get_kde_l(smp_l, l_min = 0, l_max = 60, vis=False):
    """
    Get the samples in the dimensionful coupling constant l
    and output a Kernel Denisty Estimator. 
    
    In general, we find that the sharp boundary at l = 0
    causes problems in generating a KDE that reflects the 
    true distribution. To avoid this we use the "mirror-around-zero" 
    approach of 2104.11189.
    
    We follow: https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
    """
    
    from scipy.stats import norm
    from sklearn.neighbors import KernelDensity
    from scipy.interpolate import InterpolatedUnivariateSpline as ius
    
    # Mirror around l = 0
    smp_l_full = np.concatenate((smp_l, -smp_l))
    
    # Create l values
    dl = 1
    l = np.arange(- l_max, l_max + dl, dl)    
    density = sum(norm(li).pdf(l) for li in smp_l_full)
    
    # Instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=2.0, kernel='gaussian')
    kde.fit(smp_l_full[:, None])

    # Score_samples returns the log of the probability density
    logprob = kde.score_samples(l[:, None])
    prob = np.exp(logprob)
    
    # Now we take only the l >= 0 values and the associated KDE values, making a interpolating spline
    # normalize then normalize a final time.

    l_mid = int((len(l) - 1) / 2)
    l_pos = l[l_mid:]
    pdf_pos = prob[l_mid:]
    
    pdf_l_interp = ius(l_pos, pdf_pos)
    
    if vis:
        # Compute the KDE using the same process, but without the mirroring
        # for illustrative purposes
        density_illus = sum(norm(li).pdf(l_pos) for li in smp_l)
        kde_illus = KernelDensity(bandwidth=2.0, kernel='gaussian')
        kde_illus.fit(smp_l[:, None])
        prob_illus = np.exp(kde_illus.score_samples(l_pos[:, None]))
        
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(l_pos, pdf_l_interp(l_pos) / pdf_l_interp.integral(l_min, l_max), 'k-', lw=1, label='KDE')
        ax.plot(l_pos, prob_illus, 'C1:', lw=1, label='KDE (no mirror)')
        ax.hist(smp_l, density=True, histtype='step', color='C3', lw=1)
        ax.set_xlim(l_min, l_max)
        ax.set_xlabel(r'$\ell$ [km]')
        ax.set_ylabel('PDF')
        ax.legend(loc='best', fontsize=8)
            
    return l_pos, pdf_l_interp(l_pos) / pdf_l_interp.integral(l_min, l_max)

def get_cdf(x, y, vis=False):
    """
    Return an interpolating function for the cumulative density function (CDF)
    """
    
    from scipy.interpolate import InterpolatedUnivariateSpline as ius
    
    yx_interp = ius(x, y)
    
    cdf = [yx_interp.integral(0, x_max) for x_max in x]
    
    if vis:
        fig, ax = plt.subplots(figsize=(4, 2))

        ax.plot(x, cdf)
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(0, 1.1)
        ax.set_xlabel(r'$\ell$ [km]')
        ax.set_ylabel('CDF')
    
    return x, cdf

def get_percentage(l, cdf, pct=0.9, vis=False):
    """
    Given values of l and the corresponding CDF, 
    locate the value of l for which CDF = pct.
    
    Default value is pct=0.9
    """
    
    from scipy.interpolate import InterpolatedUnivariateSpline as ius
    from scipy.optimize import brentq
    
    cdf_interp = ius(l, cdf)
    
    def root_finder(x):
        return cdf_interp(x) - pct

    l_at_pct = brentq(root_finder, min(l), max(l))
    
    if vis:
        fig, ax = plt.subplots(figsize=(4, 2))

        ax.plot(l, cdf)
        ax.axhline(pct, c='k', ls='-')
        ax.axvline(l_at_pct, c='k', ls='-')
        ax.set_xlim(min(l), max(l))
        ax.set_ylim(0, 1.1)
        ax.set_xlabel(r'$\ell$ [km]')
        ax.set_ylabel('CDF')        
    
    return l_at_pct

def do_posterior_combine(x1, y1, prior_limits1, 
                         x2, y2, prior_limits2, 
                         vis=False):

    """
    Computes the joint posterior
    """
    
    # Create the interpolants
    from scipy.interpolate import InterpolatedUnivariateSpline as ius

    yx1_kde = ius(x1, y1)
    yx2_kde = ius(x2, y2)
    
    # Now we multiply them:
    x12_vals = np.linspace(0, max(x1), num=len(x1))
    
    y12_vals = [yx1_kde(x) * yx2_kde(x) for x in x12_vals]
    
    yx12_combined = ius(x12_vals, y12_vals)
    yx12_normalisation = yx12_combined.integral(0, max(x1))
    
    if vis:
        fig, ax = plt.subplots(figsize=(4, 2))

        ax.plot(x1, y1, 'C0-', label='Posterior 1')
        ax.plot(x2, y2, 'C1--', label='Posterior 2')
        ax.plot(x1, yx12_combined(np.array(x1)) / yx12_normalisation, 'C2:', label='Combined posterior')

        ax.set_xlim(min(x1), max(x1))
        
        ax.set_xlabel(r'$\ell$ [km]')
        ax.set_ylabel('PDF')
        
        ax.legend(loc='best', fontsize=8)
    
    return x12_vals, yx12_combined(np.array(x1)) / yx12_normalisation

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