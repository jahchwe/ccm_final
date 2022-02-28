#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 13:44:41 2019

@author: jahchwe

FINAL project!!

Steps
* Create simulated discrimination data
* Create distance plots
* Explore category additions
"""

import pandas as pd
import seaborn as sb
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist
from scipy.stats import norm
from scipy.special import logsumexp
sb.set()
#create 'faces' evenly spaced
faces_on_dimension = np.asarray([(i*0.5) + 0.25 for i in range(-10,30)])

#distances in gender dimension
#gender_dist_square = squareform(pdist(faces_on_dimension, metric = 'euclidean'))
#gender_dist_map = sb.heatmap(gender_dist_square)
#plt.title('Pairwise Euclidean Distance between Faces')
#plt.show()


#mimic plot in paper
#adjacent_pair_dist = []
#for i in range(len(gender_dist_square)-1):
#    print(gender_dist_square[i, i+1])

'''
##################
two category model
##################
'''

 
'''
FUNCTIONS FOR LIKELIHOOD
'''

def log_prob_C(category):
    return np.log(category_priors[category])

def log_prob_S_T(stim, targets):
    #stim: single stim value
    #targets: array of hypothetical target values
    return norm.logpdf(stim, targets, noise_sd)

def log_prob_T_C(targets, category):
    #targets: array of hypothetical target values
    #category: int indicating which category 
    return norm.logpdf(targets, category_means[category], category_sd[category])


def log_likelihood_S_C(stim_array, category):
    #stim_array: array with all stim to evaluate
    #category: int indicating which category
    return_array = []
    #Because I can't figure out how to do the integration properly, I'm specifying a 
    #concrete list of potential target sounds
    
    #iterate through all stim, but use vectorized operations for targets
    for stim in stim_array:
        log_s_t = log_prob_S_T(stim, targets)
        log_t_c = log_prob_T_C(targets, category)
        log_mult = log_s_t + log_t_c
        return_array.append(logsumexp(log_mult))
    
    return np.asarray(return_array)

def log_posterior_C_S(stim_array):
    #stim array: array with all stim to evaluate
    #categories: array with list of category integers
    
    #to calculate denominator, which is sum of numerators across categories
    #calculate for all categories at once
    
    #will be a num_stim x num_categories returned array
    
    # num_stim x num_categories
    numerators = []
    #iterate through categories
    for cat in range(len(category_means)): 
        likelihood = log_likelihood_S_C(stim_array, cat)
        prior = log_prob_C(cat)
        numerators.append(likelihood + prior)
        
    denominator = logsumexp(numerators, axis = 0)
    
    return numerators - denominator


#plt.savefig("test.png")

#not in log space
def expected_T_S(stim_array, log_posterior_C_S):
   #stim_num x 1
    first_term = np.multiply((category_var[0]/(category_var[0] + noise_var)), stim_array)
    first_term = np.reshape(first_term, (len(faces_on_dimension),))
    # stim_num x 1
    weighted_cat_means = np.sum(np.transpose(np.exp(log_posterior_C_S)) * category_means, axis = 1)
    #stim_num x 1
    second_term = np.multiply((noise_var/(category_var[0] + noise_var)), weighted_cat_means)
    
    return first_term + second_term

def measure_magnet(stim_array, expected):
    stim_array = np.reshape(stim_array, (len(stim_array),))
    return np.sum(np.abs(stim_array - expected))

def plot_posterior_cat_id(log_posterior, labels, faces, outpath):
    plt.figure()
    posterior = np.exp(log_posterior)
    for it, cat in enumerate(posterior):
        plt.plot(faces, cat, label = labels[it])

    plt.legend()
    plt.title('Posterior Category Identification')
    plt.xlabel('Faces on dimension')
    plt.ylabel('Probability')
    plt.savefig(outpath, dpi = 400)
    
def plot_magnet(expected_values, labels, outpath):
    plt.figure()
    mx = None
    for cat_id in range(len(category_means)):
        dist = np.exp(log_prob_T_C(targets, cat_id))
        mx = np.max(dist)
        plt.plot(targets, dist, label = 'P(%s)' % (labels[cat_id]))
    
    plt.plot([faces_on_dimension, expected_test], [mx, 0], 'k.-', alpha = 0.4)
    plt.legend()
    plt.xlabel('Faces on dimension')
    plt.ylabel('Probability')
    plt.title('Magnet effect')
    plt.savefig(outpath, dpi = 400)
'''
########################
'''        

targets = np.arange(-5, 15, 0.1)

noise_sd = 0.5
noise_var = noise_sd**2
#!!!!!!!! CATEGORIES MUST HAVE EQUAL VARIANCES for formulas to hold
magnet_measures = []
expected_vals = []

'''
########
Narrow 2
########
'''

category_priors = np.asarray([0.5, 0.5])
category_means = np.asarray([0,10])
category_sd = np.asarray([0.5, 0.5])
category_var = category_sd**2

narrow_2_C_S = log_posterior_C_S(faces_on_dimension)
        
plot_posterior_cat_id(narrow_2_C_S, ['Male', 'Female'], faces_on_dimension, 'narrow_2_id.png')
expected_test = expected_T_S(faces_on_dimension, narrow_2_C_S)
plot_magnet(expected_test, labels = ['Male', 'Female'], outpath = 'narrow_2_magnet.png')

expected_vals.append(expected_test)
magnet_measures.append(measure_magnet(faces_on_dimension, expected_test))

'''
########
Narrow 3
########
'''

category_priors = np.asarray([0.4, 0.2, 0.4])
category_means = np.asarray([0, 5, 10])
category_sd = np.asarray([0.5, 0.5, 0.5])
category_var = category_sd**2

narrow_3_C_S = log_posterior_C_S(faces_on_dimension)
  
plot_posterior_cat_id(narrow_3_C_S, ['Male', 'Ambiguous', 'Female'], faces_on_dimension, 'narrow_3_id.png')
expected_test = expected_T_S(faces_on_dimension, narrow_3_C_S)
plot_magnet(expected_test, labels = ['Male', 'Ambiguous', 'Female'], outpath = 'narrow_3_magnet.png')

expected_vals.append(expected_test)
magnet_measures.append(measure_magnet(faces_on_dimension, expected_test))

'''
########
Wide 2
########
'''

category_priors = np.asarray([0.5, 0.5])
category_means = np.asarray([0, 10])
category_sd = np.asarray([2, 2])
category_var = category_sd**2
        
wide_2_C_S = log_posterior_C_S(faces_on_dimension)

plot_posterior_cat_id(wide_2_C_S, ['Male', 'Female'], faces_on_dimension, 'wide_2_id.png')
expected_test = expected_T_S(faces_on_dimension, wide_2_C_S)
plot_magnet(expected_test, labels = ['Male', 'Female'], outpath = 'wide_2_magnet.png')

expected_vals.append(expected_test)
magnet_measures.append(measure_magnet(faces_on_dimension, expected_test))

'''
########
Wide 3
########
'''

category_priors = np.asarray([0.4, 0.2, 0.4])
category_means = np.asarray([0, 5, 10])
category_sd = np.asarray([2, 2, 2])
category_var = category_sd**2
        
wide_3_C_S = log_posterior_C_S(faces_on_dimension)

plot_posterior_cat_id(wide_3_C_S, ['Male', 'Ambiguous', 'Female'], faces_on_dimension, 'wide_3_id.png')
expected_test = expected_T_S(faces_on_dimension, wide_3_C_S)
plot_magnet(expected_test, labels = ['Male', 'Ambiguous', 'Female'], outpath = 'wide_3_magnet.png')

expected_vals.append(expected_test)
magnet_measures.append(measure_magnet(faces_on_dimension, expected_test))

#plot magnet effect
bar_plot_df = pd.DataFrame()
bar_plot_df['abs(stim - E[T|S])'] = magnet_measures
bar_plot_df['type'] = ['narrow (SD = 0.5)','narrow (SD = 0.5)','wide (SD = 2)','wide (SD = 2)']
bar_plot_df['number_of_cats'] = ['2', '3', '2', '3']
plt.clf()
sb.barplot(x = 'type', y = 'abs(stim - E[T|S])', hue = 'number_of_cats', data = bar_plot_df)
plt.title('Magnet Effect')
plt.savefig('magnet_effect_graph.png', dpi = 400)

#plot distributions of magnet changes
#basically show how the extreme values decrease more in the narrow 2-->3
faces_minus_expected = faces_on_dimension - expected_vals
plt.clf()
names = ['narrow_2', 'narrow_3', 'wide_2', 'wide_3']
ylims = [0.5, 0.5, 7, 7]
xlims_min = [-4,-4,-0.4,-0.4]
xlims_max = [4, 4, 0.4, 0.4]
for it,cond in enumerate(faces_minus_expected):
    sb.distplot(cond, bins = 20)
    plt.title('%s distribution of stim - E[T|S]' % names[it])
    plt.xlabel('stim - E[T|S]')
    plt.ylim(0, ylims[it])
    plt.xlim(xlims_min[it], xlims_max[it])
    plt.savefig('%s_dist.png' % names[it], dpi = 400)
    plt.clf()
