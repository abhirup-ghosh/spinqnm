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