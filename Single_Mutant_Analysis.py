#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 12:23:31 2021

@author: mahfuz
"""

import sys
import os
import math
import numpy as np
import pandas as pd
import h5py
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

from jeder import evaluation_table, vec_precision, vec_recall, fpr_convert
from jeder import parse_hitspec, eval_expression

import pickle as pickle
import random


data_dir = '/media/mahfuz/2A8B1AF404AA6CF6/CRISPR/MCMC_Jeder/Jeder_Manuscript/Analysis/WT_Minimal_analysis/'
input_files = ['lfc_WT_min_rep_3_seed_40.txt', 'lfc_WT_min_rep_5_seed_40.txt', 'lfc_WT_min_rep_7_seed_40.txt', 'lfc_WT_min_rep_10_seed_40.txt', 'lfc_WT_min_rep_15_seed_40.txt', 'lfc_WT_min_rep_21_seed_40.txt']
consensus_files = ['results.lfc.neg.1.0.FPR.0.005.0.009.FNR.0.05.0.09.3rep.hdf5', 'results.lfc.neg.1.0.FPR.0.005.0.009.FNR.0.07.0.11.5rep.hdf5', 'results.lfc.neg.1.0.FPR.0.005.0.009.FNR.0.08.0.12.7rep.hdf5', 'results.lfc.neg.1.0.FPR.0.005.0.009.FNR.0.08.0.12.10rep.hdf5', 'results.lfc.neg.1.0.FPR.0.005.0.009.FNR.0.08.0.12.15rep.hdf5', 'results.lfc.neg.1.0.FPR.0.005.0.009.FNR.0.08.0.12.21rep.hdf5']

number_of_reps = [3, 5, 7, 10, 15, 21]
lfc_cutoff = [-3, -2.5, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.5, 1, 2]



# =============================================================================
""" Question 1: How does a consensus perform against unseen replicates?
    - Generate consensus profiles for 5, and 7 screens
    - Take the 5/7 screens consensus and compare it against 3 other screens (that are not part of the 5/7 screens)
    - Calculate precision and recall at different lFC cutoffs for the 3 hold-out screens (compare against seen screens)
        - (-3.0, -2.5, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, 0.5, 1.0, 2.0)
        - Write a commentary about the comparison
"""
# =============================================================================
"""
For the 7 replicate consensus, choose separate 7 screens and compare them against the consensus.

# Make a precision-recall bar plot similar to the ones I did for ISMB poster
# For different consensus, use independent screens to calculate precision and recall at different cutoffs, plus a PR curve summarizing the two bar plots maybe
"""

# Calculate consensus / standard for the 7 replicate version
i = 2
input_file = data_dir + 'Input/' + input_files[i]
output_file = data_dir + 'Output/' + consensus_files[i]

hf = h5py.File(output_file, 'r')
vec_std = np.round(hf['vec_mean']).astype(np.bool) # consensus profile   

df = pd.read_table(input_file)
screens_in_consensus = set(df.repid.unique())


# Select 7 independent screens (randomly sampled from the rest)
df = pd.read_table(data_dir + 'Input/' + input_files[5])
input_df = df.pivot_table(index='repid', columns='expid', values='lfc', fill_value=False)
random.seed(40)
screen_names = random.sample(set.difference(set(input_df.index.to_list()), screens_in_consensus), 7)

# screen_names = screens_in_consensus


# Count TP, FP, TN, FN for individual screens across lfc cutoffs
rows = len(screen_names)
cols = len(lfc_cutoff)

TP_matrix = np.zeros((rows, cols))
FP_matrix = np.zeros((rows, cols))
FN_matrix = np.zeros((rows, cols))
TN_matrix = np.zeros((rows, cols))
num_ess = np.zeros((rows, cols))

# df_subset = df.loc[df['repid'].isin(screen_names)] # subset data
# df_subset.repid.value_counts()

for j in range(cols):
    print('LFC cutoff: {}'.format(lfc_cutoff[j]))
    df_tmp =  df.loc[df['repid'].isin(screen_names)]
    hits = np.ones(df_tmp.shape[0], dtype=np.bool)
    hits = hits & (df_tmp['lfc'] < lfc_cutoff[j])
    
    df_tmp['jeder_hits'] = hits
    tmp_df = df_tmp.pivot_table(index='repid', columns='expid', values='jeder_hits', fill_value=False)
    
    num_ess[:,j] = np.sum(tmp_df, axis = 1)
    y_truth = np.tile(vec_std, (tmp_df.shape[0],1))
    TP_matrix[:,j] = np.sum(y_truth & tmp_df, axis=1)
    TN_matrix[:,j] = np.sum((y_truth == False) & (tmp_df == False), axis=1)
    FP_matrix[:,j] = np.sum(tmp_df, axis=1) - TP_matrix[:,j]
    FN_matrix[:,j] = np.sum(y_truth, axis=1) - TP_matrix[:,j]

    
# Calculate Precision/Recall and plot barplot and PR curves
screen_TP = np.nansum(TP_matrix, axis = 0) 
screen_FP = np.nansum(FP_matrix, axis = 0)
screen_TN = np.nansum(TN_matrix, axis = 0)
screen_FN = np.nansum(FN_matrix, axis = 0)

screen_precision = np.round(screen_TP / (screen_TP + screen_FP), 4)
screen_recall = np.round(screen_TP / (screen_TP + screen_FN), 4)
screen_essentials = np.round(np.nanmean(num_ess, axis = 0))


# PR plot
lfc_cutoff_str = list(map(str, lfc_cutoff))
fig, ax = plt.subplots(1,3, figsize=(8, 4), sharey=True) # Share the same y axis
# fig.suptitle('LFC Consensus evaluation', fontsize=16)

# Precision
ax[0].plot(screen_recall, screen_precision)
ax[0].set(xlabel='Recall', ylabel='Precision')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
for i in range(len(screen_recall)-2):
    if(screen_precision[i] > 0.9):
        ax[0].text(screen_recall[i] + 0.01, screen_precision[i]+0.01, lfc_cutoff_str[i], fontsize=8,  rotation = 90)
    else:
        ax[0].text(screen_recall[i] + 0.01, screen_precision[i]+0.01, lfc_cutoff_str[i], fontsize=8)

for label in (ax[0].get_xticklabels() + ax[0].get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(8)

# Precision
barlist = ax[1].bar(lfc_cutoff_str, screen_precision)
ax[1].set_title('Precision')
ax[1].set(xlabel='log FC (LFC)', ylabel='')
ax[1].set_xticklabels(lfc_cutoff_str, rotation=90) 
barlist[6].set_color('r')
for label in (ax[1].get_xticklabels() + ax[1].get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(8)

# Recall
barlist = ax[2].bar(lfc_cutoff_str, screen_recall)
ax[2].set_title('Recall')
ax[2].set(xlabel='log FC (LFC)', ylabel='')
ax[2].set_xticklabels(lfc_cutoff_str, rotation=90)
barlist[6].set_color('r')
for label in (ax[2].get_xticklabels() + ax[2].get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(8)

# plt.show()
plt.savefig('LFC_consensus_evaluation_independent_screens.png', dpi=300, bbox_inches='tight')


# =============================================================================
""" Question 2: How many replicate screens is necessary for JEDER to run well?
    Generate consensus for 3,5,7,10,15, and 21 screens
    Caluclate a precision and recall vs number of screens heatmap at different lfc cutoffs (Same from 1 should be fine)
    Make a commentary about how many screens is necessary.
"""
# =============================================================================
rows = len(number_of_reps)
cols = len(lfc_cutoff)

# For individual replicates for the different sets (3,5,7,10,15,21), calculate stats against their consensus
screen_names = []
TP_matrix = np.zeros((sum(number_of_reps), cols))
FP_matrix = np.zeros((sum(number_of_reps), cols))
FN_matrix = np.zeros((sum(number_of_reps), cols))
TN_matrix = np.zeros((sum(number_of_reps), cols))
num_ess = np.zeros((sum(number_of_reps), cols))

indices = list(np.cumsum(number_of_reps))
start_ind = 0

for i in range(len(number_of_reps)):
    print('Numer of Reps: {}'.format(number_of_reps[i]))    
    input_file = data_dir + 'Input/' + input_files[i]
    output_file = data_dir + 'Output/' + consensus_files[i]
    
    valid_ind = range(start_ind, indices[i])
    
    hf = h5py.File(output_file, 'r')
    vec_std = np.round(hf['vec_mean']).astype(np.bool) # standard (17804 genes )
    
    # Remove frequent flyer genes from standard
    df = pd.read_table(input_file)
    input_df = df.pivot_table(index='repid', columns='expid', values='lfc', fill_value=False)
    screen_names.extend(input_df.index.to_list())

    for j in range(cols):    
        hits = np.ones(df.shape[0], dtype=np.bool)
        hits = hits & (df['lfc'] < lfc_cutoff[j])
        
        df['jeder_hits'] = hits
        tmp_df = df.pivot_table(index='repid', columns='expid', values='jeder_hits', fill_value=False)
        
        num_ess[valid_ind,j] = np.sum(tmp_df, axis = 1)
        y_truth = np.tile(vec_std, (tmp_df.shape[0],1))
        TP_matrix[valid_ind,j] = np.sum(y_truth & tmp_df, axis=1)
        TN_matrix[valid_ind,j] = np.sum((y_truth == False) & (tmp_df == False), axis=1)
        FP_matrix[valid_ind,j] = np.sum(tmp_df, axis=1) - TP_matrix[valid_ind,j]
        FN_matrix[valid_ind,j] = np.sum(y_truth, axis=1) - TP_matrix[valid_ind,j]
    
    start_ind = indices[i]
    
# ====================== Save data using pickle ======================
with open('TP_FP_neg_individual_screens.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([TP_matrix, FP_matrix, FN_matrix, TN_matrix, number_of_reps, num_ess, screen_names], f)
    
with open('TP_FP_neg_individual_screens.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    TP_matrix, FP_matrix, FN_matrix, TN_matrix, number_of_reps, num_ess, screen_names = pickle.load(f)
    
    
#  ====================== Plot Precision, recall, degree ======================
screen_TP = np.zeros((len(number_of_reps), cols)) # TP sum at a cutoff
screen_FP = np.zeros((len(number_of_reps), cols))
screen_TN = np.zeros((len(number_of_reps), cols))
screen_FN = np.zeros((len(number_of_reps), cols))

screen_precision = np.zeros((len(number_of_reps), cols))
screen_recall = np.zeros((len(number_of_reps), cols))
screen_essentials = np.zeros((len(number_of_reps), cols))

indices = list(np.cumsum(number_of_reps))
start_ind = 0

for i in range(len(number_of_reps)):    
    screen_id = range(start_ind, indices[i])
    screen_TP[i,:] = np.nansum(TP_matrix[screen_id,:], axis = 0) # TP sum at a cutoff
    screen_FP[i,:] = np.nansum(FP_matrix[screen_id,:], axis = 0)
    screen_TN[i,:] = np.nansum(TN_matrix[screen_id,:], axis = 0)
    screen_FN[i,:] = np.nansum(FN_matrix[screen_id,:], axis = 0)
    
    screen_precision[i,:] = np.round(screen_TP[i,:] / (screen_TP[i,:] + screen_FP[i,:]), 4)
    screen_recall[i,:] = np.round(screen_TP[i,:] / (screen_TP[i,:] + screen_FN[i,:]), 4)
    screen_essentials[i,:] = np.round(np.nanmean(num_ess[screen_id,:], axis = 0))
    
    start_ind = indices[i]


'''
# Plot a table
fig, ax = plt.subplots()
fig.patch.set_visible(False) # hide axes
ax.axis('off')
ax.axis('tight')

ax.table(cellText=global_precision, cellLoc = 'left', 
         colLabels=col_header, colLoc = 'left', 
         rowLabels=row_header, loc='center')
fig.tight_layout()
plt.show()
'''
 
# ============== plot a heatmap with annotation ==============
# https://seaborn.pydata.org/generated/seaborn.heatmap.html
row_header = list(map(str, number_of_reps)) # Number of replicates
# row_header[0] = '# of screens = ' + row_header[0]
col_header = list(map(str, lfc_cutoff)) # LFC Cutoff
# col_header[0] = 'LFC < ' + col_header[0]

df_prec = pd.DataFrame(screen_precision, columns = col_header, index = row_header)
df_recall = pd.DataFrame(screen_recall, columns = col_header, index = row_header)

# cmap = 'coolwarm', 'gist_yarg' (for grayscale)
# fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
fig, ax = plt.subplots(2,1, figsize=(8, 6), sharey=True)
fig.suptitle('Number of Replicates vs Performance', fontsize=16)

# Precision
sns.heatmap(df_prec.transpose(), annot=True, annot_kws={"size": 8}, linewidths=.5, cmap="gist_yarg", robust = True, ax=ax[0])
ax[0].set_title('Precision')
ax[0].set_xticks([])
ax[0].tick_params('y', labelrotation=0)
ax[0].set(xlabel='', ylabel='LFC cutoff (<)')
# plt.yticks(rotation=0)

# Degree
# sns.heatmap(df_ess, annot=True, annot_kws={"size": 8}, fmt=".0f", linewidths=.5, cmap="coolwarm", robust = True, ax=ax[0,1])
# ax[0,1].set_title('qGI degree')

# Recall
sns.heatmap(df_recall.transpose(), annot=True, annot_kws={"size": 8}, linewidths=.5, cmap="gist_yarg", robust = True, ax=ax[1])
ax[1].set_title('Recall')
# ax[-1, -1].axis('off') # Make last one blank
ax[1].tick_params('y', labelrotation=0)
ax[1].set(xlabel='# of replicates', ylabel='LFC cutoff (<)')

# plt.show()
plt.savefig('NUmber_of_replicates_vs_Performance_BW.png', dpi=300)



# =============================================================================
""" Question 3a: What is overlap between different Consensus profiles?
    - Calculate the essential gene set for different consensus
    - Create an upset plot to show their overlap
    - This is not the best way as the overlap in this figure may just be due to sampling
"""
# =============================================================================
# https://pypi.org/project/UpSetPlot/
# https://stackoverflow.com/questions/47407985/need-help-for-py-upset
# https://jokergoo.github.io/ComplexHeatmap-reference/book/upset-plot.html (in R)
# Example 1
from upsetplot import plot
from upsetplot import UpSet


# Get list of essential genes for each consensus
df = pd.read_table(data_dir + 'Input/' + input_files[0])
input_df = df.pivot_table(index='repid', columns='expid', values='lfc', fill_value=False)
gene_names = input_df.columns.to_numpy()

tmp = list(map(str, number_of_reps))
rep_str = ['reps: ' + data for data in tmp]
gene_ess_per_consensus = pd.DataFrame(columns=rep_str, index=gene_names)

for i in range(len(number_of_reps)):
    print('Numer of Reps: {}'.format(number_of_reps[i]))    
    output_file = data_dir + 'Output/' + consensus_files[i]
        
    hf = h5py.File(output_file, 'r')
    vec_std = np.round(hf['vec_mean']).astype(np.bool) # standard (17804 genes )
    
    # gene_ess_per_consensus.iloc[:,i] = 
    gene_ess_per_consensus[rep_str[i]] = vec_std

"""
         reps: 3  reps: 5  reps: 7  reps: 10  reps: 15  reps: 21
A1BG       False    False    False     False     False     False
A1CF       False    False    False     False     False     False
"""

gene_ess_per_consensus = gene_ess_per_consensus.assign(num_ess=gene_ess_per_consensus.sum(axis = 1))

# Only genes essential in at least one consensus
subset_ess_per_consensus = gene_ess_per_consensus.loc[gene_ess_per_consensus.num_ess > 0, :]
upset_df = subset_ess_per_consensus.set_index(rep_str)

# Sort by cardinality, but keep the categories as provided
# https://upsetplot.readthedocs.io/en/stable/api.html#plotting
upset = UpSet(upset_df, subset_size='count', sort_by='cardinality', show_counts=True, sort_categories_by=None) # show_percentages = True, facecolor='#756bb1'
upset.plot()
current_figure = plt.gcf()
current_figure.savefig("Upset_plot_between_consensus.png", dpi=300)


# =========== Do a barplot with number of times a gene is essential ===========#
# This is probably not the right way to show this as many of the consensus has duplicate genes in there
ess_number_of_times = subset_ess_per_consensus.num_ess.value_counts()

# libraries
import numpy as np
import matplotlib.pyplot as plt
 
# create dataset
data_sorted = ess_number_of_times.sort_index()
height = data_sorted.to_list()
bars = list(map(str, data_sorted.index.to_list()))
x_pos = np.arange(len(bars))
 
plt.bar(x_pos, height, color = (0.2, 0.2, 0.2, 0.8)) # Create bars and choose color
plt.title('Essential across consensus')
plt.xlabel('Number of Consensus')
plt.ylabel('Number of essentials')
plt.xticks(x_pos, bars) # Create names on the x axis
plt.show()

# ============ Maybe do a version on the total number of WT screens ===============
# * TODO: Make it cumulative (Only add a screen if it adds new essentials) *
df = pd.read_table(data_dir + 'Input/' + input_files[-1])
input_df = df.pivot_table(index='expid', columns='repid', values='lfc', fill_value=False)
input_df = input_df < -1.0

data_sorted = input_df.loc[input_df.sum(axis = 1) > 0, ].sum(axis = 1).value_counts().sort_index()
height = data_sorted.to_list()
bars = list(map(str, data_sorted.index.to_list()))
x_pos = np.arange(len(bars))
 
plt.bar(x_pos, height, color = (0.2, 0.2, 0.2, 0.8)) # Create bars and choose color
plt.title('Essential across WT screens')
plt.xlabel('Number of Screens')
plt.ylabel('Number of essential genes')
plt.xticks(x_pos, bars) # Create names on the x axis



# =============================================================================
""" Question 3b: What is overlap between consensus and independent essential set
    - Calculate the essential gene set for different consensus
    - Read in other essential sets (Traver's core & DepMap 60)
    - Calculate the overlap
"""
# =============================================================================




