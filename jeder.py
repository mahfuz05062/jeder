#!/usr/bin/env python3
doc = """
jeder (Jointly Estimate Data and Error Rates)

Usage:
   jeder.py [options] run <hit_spec> <input> <output>
   jeder.py [options] view <output>

Options:
   -b --burn=<int>         [default: 100] how many initial iterations
   -i --iterations=<int>   [default: 1000] how many subsequent iterations
   -e --expid=<colname>    [default: expid] which col contains the experiment id
   -r --repid=<colname>    [default: repid] which col contains the replicate id
   -p --fpr=<pspec>        [default: 0,0.05]  the prior distrubution for the FPR
   -n --fnr=<pspec>        [default: 0.2,0.8] the prior distrubution for the FNR
   -s --standard=<runfile> take "truth" vector from previous run instead of estimating it
   -t --trace              [default: False] save the profile trace (see NOTE) 
   -c --clobber            overwrite output file
   -q --quiet              do not print out messages, and disable the progress bar
   -h --help               show this help

hit_spec: A "hit" is defined using a combination of threholds on one or more
         columns. Use the column names from the input file as variables.
         
         Currently supported syntax
         BASIC AND "(score < -0.08) & (pvalue < 0.05)"
         RELATIVE  "(col_A > col_B) & (col_C > 0)"

         Future work: 

         NO SPACES "(score<-0.08)&(pvalue<0.05)"
         ORs       "(score < -0.08) | (score  > 0.08)"

         Nested expressions are not allowed, and evaluated left to right.
         e.g. ( A | B & C) is evaluated as ((A | B) & C)
         If you need more complex logic, you will have to preprocess
         your data, and include the result in a column 

defining priors: Currently, only uniform priors for fpr and fnr are implemented.
         They are defined using the upper and lower bound, comma-delimited,
         with NO WHITESPACE. e.g. use "-n 0.2,0.8" to search between 20% and 80%
         for the false negative rate.

NOTE: Saving the profile trace with the --trace option can potentially use a
lot of disk space depending on the size of your experiment. You may want to run
some jobs with a small number of iterations. The output file size should scale
approximately linearly with the number of iterations.

"""

import sys
import os
import time
import math
from scipy.stats import uniform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from docopt import docopt
import h5py
import itertools

# CORE MCMC METHODS ###########################################################


def mcmc_fit(matrix, burn=1000, iters=11000, fpr=(0,0.05), fnr=(0.2,0.8), quiet=False, standard=None):
   """
   mcmc_fit(matrix, iters=11000, burn=1000, fpr=(0,0.05), fnr=(0.2,0.8), quiet=False):
   matrix is a dataframe, values will be cast to bool
   each ROW is a REPLICATE
   each COL is a VARIABLE
   iters does not include burn, total rounds will be sum of the two
   """
   t = time.time()
   replicates = matrix.shape[0]
   col_sums = np.sum(matrix, axis=0)
   bin_matrix = matrix.values.astype(np.bool)

   fpr_prior = uniform(scale=fpr[1]-fpr[0], loc=fpr[0])
   fnr_prior = uniform(scale=fnr[1]-fnr[0], loc=fnr[0])

   # intial values
   fpr_current = fpr_prior.rvs()
   fnr_current = fnr_prior.rvs()
   if standard is None:
      vec_current = np.round(np.mean(bin_matrix, axis=0))
   else:
      vec_current = standard

   # hold the traces
   fpr_trace = np.zeros((iters,1))
   fnr_trace = np.zeros((iters,1))
   vec_trace = np.zeros((iters, bin_matrix.shape[1]))


   den_current = np.sum(bin_matrix.ravel()) / len(bin_matrix.ravel())
   #  den_current = 0.045

   for G in tqdm(range(burn + iters), disable=quiet):
      # step one, update the fpr estimate ----------------------------
      # this requires an evaluation of each datapoint
      l_current = l_obs_vec(col_sums, replicates, 
            vec_current, fpr_current, fnr_current)

      fpr_proposed = fpr_prior.rvs()
      l_proposed = l_obs_vec(col_sums, replicates, 
            vec_current, fpr_proposed, fnr_current)
      
      # add log probabilities to weight them
      p_current = l_current + np.log(fpr_prior.pdf(fpr_current))
      p_proposed = l_proposed + np.log(fpr_prior.pdf(fpr_proposed))
      p_accept = np.exp(p_proposed - p_current) # back to probability
      accept = np.random.rand() < p_accept
      
      if accept:
         fpr_current = fpr_proposed

      # step two, like step one, but for fnr ---------------------------
      l_current = l_obs_vec(col_sums, replicates, 
            vec_current, fpr_current, fnr_current)
      
      fnr_proposed = fnr_prior.rvs()
      l_proposed = l_obs_vec(col_sums, replicates, 
            vec_current, fpr_current, fnr_proposed)
      
      # add log probabilities to weight them
      p_current = l_current + np.log(fnr_prior.pdf(fnr_current))
      p_proposed = l_proposed + np.log(fnr_prior.pdf(fnr_proposed))
      p_accept = np.exp(p_proposed - p_current)
      accept = np.random.rand() < p_accept
      
      if accept:
         fnr_current = fnr_proposed

      # step three: iterate through the data vector -------------------
      # and give a chance for our estimate of EACH VARIABLE to change
      # but hold fdr fnr constant

      # skip step 3 if we are using an external standard
      if standard is None:
         for i in range(bin_matrix.shape[1]):
            val_current = vec_current[i]
            l_current = l_obs(col_sums[i], replicates, val_current, fpr_current, fnr_current)
            val_inverted = (vec_current[i] + 1) %2
            l_inverted = l_obs(col_sums[i], replicates, val_inverted, fpr_current, fnr_current)


            # priors of interaction and non-interaction are density-based
            if val_current:
               prior_val = den_current
               prior_inv = 1-den_current
            else:
               prior_val = 1-den_current
               prior_inv = den_current

            # weight each likelihood by the prior of an observed interaction (density)
            p_current = l_current * prior_val
            p_inverted = l_inverted * prior_inv
            p_accept = p_inverted / p_current
            accept = np.random.rand() < p_accept

            if accept:
               vec_current[i] = val_inverted

      # update the vector trace
      vec_trace[G-burn,:] = vec_current
      fpr_trace[G-burn] = fpr_current
      fnr_trace[G-burn] = fnr_current

   print('%d total iterations completed in %s' % (burn+iters, p_time(time.time() - t)))

   return fpr_trace, fnr_trace, vec_trace


# MCMC Plotting and analysis #################################################

def view_traces(hfile=None, save_pdf=None):
   """view the traces for on or more run files
      if no files are passed, process all files in current dir
      save_pdf takes a filename, and will save any/all plots there
   """

   if hfile is None:
      files = os.listdir()
      files = [f for f in files if f.endswith('.hdf5')]

   elif type(hfile) is str:
      files = [hfile]

   elif type(hfile) is not list:
      print('please pass a filename, list of filenames, or None for hfile')
      return

   if save_pdf is not None:
      pp = PdfPages(save_pdf)

   files.sort()
   for f in files:

      hf = h5py.File(f, 'r')

      # extract the data
      fpr_trace = hf['fpr_trace'].value
      fnr_trace = hf['fnr_trace'].value

      iters = len(fpr_trace)
      plt.figure(figsize=(8.5, 11))
      gs = gridspec.GridSpec(5,2,height_ratios=[1,8,8,8,8])

      # print the table
      plt.subplot(gs[0,:])
      table = evaluation_table(hf)
      thndle = plt.table(cellText=table, loc='top')
      tw_align(thndle)
      plt.gca().axis('off')

      plt.subplot(gs[1,:])
      plt.plot(fpr_trace, zorder=1)
      plt.ylabel('FPR trace')

      plt.subplot(gs[2,:])
      plt.plot(fnr_trace, zorder=1)
      plt.ylabel('FNR trace')

      plt.subplot(gs[3,0])
      plt.hist(fpr_trace, bins=40)
      plt.ylabel('FPR posterior')
      y2 = plt.ylim()[1]
      plt.vlines(np.nanmean(fpr_trace), 0, y2, color='k', zorder=2)

      plt.subplot(gs[3,1])
      plt.hist(fnr_trace, bins=40)
      plt.ylabel('FNR posterior')
      y2 = plt.ylim()[1]
      plt.vlines(np.nanmean(fnr_trace), 0, y2, color='k', zorder=2)

      plt.subplot(gs[4,:])
      if 'vec_trace' in hf:
         den_trace = np.sum(hf['vec_trace'].value, axis=1) / hf['vec_trace'].shape[1]
         plt.plot(den_trace, zorder=1)
         plt.ylabel('density trace')

      plt.xlabel('fpr: %.3f (%.2e), fnr: %.3f (%.2e)' % 
            (np.nanmean(fpr_trace), np.nanstd(fpr_trace),
             np.nanmean(fnr_trace), np.nanstd(fnr_trace)))
      gs.update(wspace=0.3)

      plt.suptitle(os.path.basename(f[:-5]))
      plt.show()
      hf.close()

      # leave all the plots open if we're not saving them
      if save_pdf is not None:
         pp.savefig()
         plt.close()

   if save_pdf is not None:
      pp.close()


# Generative model for testing ###############################################
def generate_model(vlength, replicates, density, fpr, fnr):
   """
   generate_model(vlength, replicates, density, fpr, fnr)
   returns truth, observations
   """

   truth = np.random.rand(vlength) < density
   true_flip = int(fnr * sum(truth))
   false_flip = int(fpr * sum(~truth))

   false_ix = np.where(~truth)[0]
   true_ix = np.where(truth)[0]

   observation = np.zeros((replicates, vlength))
   for i in range(replicates):
      # start with real data
      observation[i,:] = truth

      # add in false positives
      flip_ix = false_ix[np.random.choice(len(false_ix), false_flip, replace=False)]
      observation[i, flip_ix] = True

      # add in false negatives
      flip_ix = true_ix[np.random.choice(len(true_ix), true_flip, replace=False)]
      observation[i, flip_ix] = False


   return truth, observation


# UTILS ######################################################################

def l_obs(hits, replicates, truth, fpr, fnr):
   """
   l_obs(hits, replicates, fpr, fnr)
   given a fpr and a fnr, determine the likelihood of the data 
   given a real hit or miss
   """
   if truth:
      return math.pow(1-fnr, hits) * math.pow(fnr, replicates-hits) 
   else:
      return math.pow(fpr, hits) * math.pow(1-fpr, replicates-hits) 


def l_obs_vec(observed_hit_counts, replicates, model_vec, model_fpr, model_fnr):
   """
   likelihood of data given an entire vector, fpr, fnr
   """
   l_vec = np.zeros((len(observed_hit_counts)))
   for i in range(len(observed_hit_counts)):
      l_vec[i] = np.log(l_obs(observed_hit_counts[i], replicates, 
         model_vec[i], model_fpr, model_fnr))
   return l_vec.sum()


def fpr_convert(fpr, fnr, n, N):
   """
   convert(fpr, fnr, n, N)
   calculate TP FP... from fpr and fnr, for use in calculating other metrics
   returns TP, FP, TN, FN
   """
   alpha = fpr
   beta = fnr
   m = N-n

   # TP = (1-beta) * (alpha*m - n + alpha*n) / (alpha + beta - 1)
   N1 = 1-beta
   N2 = alpha * m - n + alpha * n
   D1 = alpha + beta - 1
   TP = np.round(N1 * N2 / D1)

   FP = n-TP

   #  TN = (1-alpha)(n-TP) / alpha 
   N3 = 1-alpha
   N4 = n-TP
   TN = np.round(N3 * N4 / alpha)

   FN = m-TN

   return TP, FP, TN, FN


def vec_precision(y_truth, y_primes):
   """
   evaluate profiles against some truth.
   Assumes that truth is a row vector and evals each row in primes
   """
   # replicate truth
   y_truth = np.tile(y_truth, (y_primes.shape[0],1))
   
   return np.sum(y_truth & y_primes, axis=1) / np.sum(y_primes, axis=1)

def vec_recall(y_truth, y_primes):
   """
   evaluate profiles against some truth.
   Assumes that truth is a row vector and evals each row in primes
   """
   # replicate truth
   y_truth = np.tile(y_truth, (y_primes.shape[0],1))
   
   return np.sum(y_truth & y_primes, axis=1) / np.sum(y_truth, axis=1)


def p_time(t):
   """p_time(t):
      convert a large number of seconds (int)
      to a human readable string for printing"""

   t = int(t)
   seconds = t % 60
   t = t-seconds
   minutes = int((t % 3600) / 60)
   t = t - 60 * minutes
   hours = int(t / 3600)

   result = ''
   if hours > 0:
      result = result + str(hours)+':h '
   if minutes > 0:
      result = result + str(minutes)+':m '
   result = result + str(seconds)+':s '
   return(result)


def eval_expression(df, expression):
   (arg1, oprtr, arg2) = expression #unpack

   eval_str = ''
   
   if type(arg1) is str:
      eval_str += ('df[\'%s\'] ' % arg1)
   elif type(arg1) is float:
      eval_str += ('%f ' % arg1)
   else:
      sys.exit('uknown argtype')

   assert(type(oprtr) is str)

   eval_str += oprtr
      
   if type(arg2) is str:
      eval_str += (' df[\'%s\']' % arg2)
   elif type(arg2) is float:
      eval_str += (' %f' % arg2)
   else:
      sys.exit('uknown argtype')

   try:
      ix = eval(eval_str)
   except:
      sys.exit('problem with expression:\n%s' % eval_str)


   return ix



def tw_align(handle):
   handle.auto_set_font_size(False)
   handle.set_fontsize(8)
   cells = handle.properties()['celld']
   for r in range(4):
      for c in [0,2]:
         cells[r,c]._loc = 'right'

      for c in [1,4]:
         cells[r,c]._loc = 'left'



def evaluation_table(hf):
   # extract a text summary of an hdf file
   fpr_mean = np.nanmean(hf['fpr_trace'][:])
   fpr_std = np.nanstd(hf['fpr_trace'][:])
   fnr_mean = np.nanmean(hf['fnr_trace'][:])
   fnr_std = np.nanstd(hf['fnr_trace'][:])
   vec_std = np.round(hf['vec_mean']).astype(np.bool)

   # we need the input data to calculate precision and recall
   input_df = reparse_data(hf)
   interactions = np.round(np.mean(np.sum(input_df, axis=1)))
   int_N = input_df.shape[1]

   if 'fpr' in hf.attrs:
      fpr_prior = hf.attrs['fpr']
      fnr_prior = hf.attrs['fnr']
   else:
      fpr_prior = 'x'
      fnr_prior = 'x'

   density = 100 * interactions / int_N

   # calculate FDR
   TP, FP, TN, FN = fpr_convert(fpr_mean, fnr_mean, 
         interactions, int_N)
   precision = TP / (TP + FP)
   recall = TP / (TP + FN)
   fdr = 1-precision

   table = []
   row = []
   
   row.append('Hits:')
   row.append(int(interactions))
   row.append('FPR mean:')
   row.append('%.3f' % fpr_mean)
   row.append('FNR mean:') 
   row.append('%.3f' % fnr_mean)
   table.append(row); row=[]

   row.append('N:')
   row.append(str(int_N))
   row.append('FPR std: ')
   row.append('%.3e' % fpr_std)
   row.append('FNR std: ')
   row.append('%.3e' % fnr_std)
   table.append(row); row=[]

   row.append('Density')
   row.append('%.1f%%' % density)
   row.append('FPR prior:')
   row.append(fpr_prior)
   row.append('FNR prior:')
   row.append(fnr_prior)
   table.append(row); row=[]

   row.append('Precision:')
   row.append('%.3f' % precision)
   row.append('Recall:')
   row.append('%.3f' % recall)
   row.append('FDR:')
   row.append('%.3f' % fdr)
   table.append(row)

   # legacy calclulation corner
   interactions_global = sum(vec_std)
   TPg, FPg, TNg, FNg = fpr_convert(fpr_mean, fnr_mean, 
         interactions_global, int_N)
   precisiong = TPg / (TPg + FPg)
   recallg = TPg / (TPg + FNg)
   fdrg = 1-precisiong

   row=[]
   row.append('PrecisionG:')
   row.append('%.3f' % precisiong)
   row.append('RecallG:')
   row.append('%.3f' % recallg)
   row.append('FDRg:')
   row.append('%.3f' % fdrg)
   table.append(row)



   # direct precision/recall calculations
   d_precision = vec_precision(vec_std, input_df)
   d_recall = vec_recall(vec_std, input_df)
   row = []
   row.append('PrecisionD:')
   row.append('%.3f' % np.mean(d_precision))
   row.append('RecallD:')
   row.append('%.3f' % np.mean(d_recall))
   row.append('FDRd:')
   row.append('%.3f' % (1 - np.mean(d_precision)))
   table.append(row)




   return table


# MAIN SUBROUTINES ###########################################

def validate_run(args):
   """ 
   docopt will ensure that command is well formed, but we still need to 
   sanity-check values """

   # does intput file exist
   if not os.path.exists(args['<input>']):
      sys.exit('unable to locate input file: ' + args['<input>'])


   # outputfile should have an hdf5 extension
   if not args['<output>'].endswith('.hdf5'):
      args['<output>'] += '.hdf5'


   # does output directory exist, and can we write to it
   outputdir = os.path.dirname(args['<output>'])
   if outputdir is '':
      outputdir = '.'

   if not os.path.exists(outputdir):
      sys.exit('no such directory for output: ' + outputdir)

   # file should not already exist
   if not args['--clobber']:
      if os.path.exists(args['<output>']):
         sys.exit('proposed outputfile already exists: %s' % args['<output>'])


   # are fpr/fnr priors well-formed
   try:
      fpr_low,fpr_high = args['--fpr'].split(',')
      fpr_low = float(fpr_low)
      fpr_high = float(fpr_high)
      assert(0 <= fpr_low < fpr_high <= 1)
      args['fpr'] = (fpr_low, fpr_high)
   except:
      sys.exit('malformed FPR specification')

   try:
      fnr_low,fnr_high = args['--fnr'].split(',')
      fnr_low = float(fnr_low)
      fnr_high = float(fnr_high)
      assert(0 <= fnr_low < fnr_high <= 1)
      args['fnr'] = (fnr_low, fnr_high)
   except:
      sys.exit('malformed FNR specification')

   # check run file exists
   if args['--standard'] and not os.path.exists(args['--standard']):
      sys.exit('cannot read standard from previous run: ' + args['--standard'])


   args['parsed_exps'], input_cols = parse_hitspec(args['<hit_spec>'], args['<input>'])

   # check for the replicate id and experiment id columns
   if args['--repid'] not in input_cols:
      sys.exit('cannot find replicate id column: %s' % args['--repid'])

   if args['--expid'] not in input_cols:
      sys.exit('cannot find experiment id column: %s' % args['--expid'])


def parse_hitspec(hitspec, inputfile):

   # reading the entire input file may be quite slow
   # peek at the top few rows so we can validate the input spec
   input_cols = pd.read_table(inputfile, nrows=2).columns


   exprs = hitspec.split(' & ')
   parsed_exps = []
   for ex in exprs:
      if not (ex.startswith('(') and ex.endswith(')')):
         sys.exit('malformed hit spec, surround each expression with ()')
      else:
         ex = ex[1:-1]

      arg1, oprtr, arg2 = ex.split(' ')

      # check for a valid operator
      if oprtr not in ['<', '<=', '>', '>=', '==', '=']:
         sys.exit('unsupported operator ( %s ) ' % oprtr)

      # rewrite assignment as isequalto
      if oprtr == '=':
         oprtr = '=='


      # which arguments for this expression are numeric?
      # failure here means arg1/arg2 still a str, no need for real except
      try:
         arg1 = float(arg1)
      except:
         pass

      try:
         arg2 = float(arg2)
      except:
         pass

      if (type(arg1) is float) and (type(arg2) is float):
         sys.exit('expression contained no column names: ' % ex)

      if (type(arg1) is str) and (arg1 not in input_cols):
         print('detected columns:')
         print(input_cols)
         sys.exit('cannot find column %s in data' % arg1)

      if (type(arg2) is str) and (arg2 not in input_cols):
         sys.exit('cannot find column %s in data' % arg2)

      # save the parsed tuple
      parsed_exps.append((arg1, oprtr, arg2))


   # if everything checks out, return the parsed expressions
   return parsed_exps, input_cols


def parse_data(args):
   """ df = parse_data(args)
   load the input data, 
      apply the hit spec, and 
      return a rectangular DataFrame

      Each row contains a replicate profile
      Each col contains an observed variable

   """
   
   df = pd.read_table(args['<input>'])

   hits = np.ones(df.shape[0], dtype=np.bool)
   for expression in args['parsed_exps']:
      hits = hits & eval_expression(df, expression)

   df['jeder_hits'] = hits

   df_wide = df.pivot_table(index=args['--repid'], columns=args['--expid'],
         values='jeder_hits', fill_value=False)

   return df_wide


def reparse_data(hf):
   """ reconstruct args from hfile and use parse_data to load the DataFrame """

   parsed_exps, input_cols = parse_hitspec(hf.attrs['hit_spec'], hf.attrs['input'])
   args = {'<input>':hf.attrs['input'], 
           '--repid':hf.attrs['repid'], '--expid':hf.attrs['expid'],
           'parsed_exps':parsed_exps}
   return parse_data(args)


def save_results(args, df, results):
   """
   model_data preps the data, calls mcmc_fit, and saves the result
   results is (fpr_trace, fnr_trace, vec_trace)
   """

   # save the results to disk
   hf = h5py.File(args['<output>'], 'w')

   # save the parameters, as strings, in main group attributes
   hf.attrs['full_cmd']   = ' '.join(sys.argv)
   hf.attrs['hit_spec']   = args['<hit_spec>']
   hf.attrs['input']      = args['<input>']
   hf.attrs['output']     = args['<output>']
   hf.attrs['burn']       = args['--burn']
   hf.attrs['iterations'] = args['--iterations']
   hf.attrs['expid']      = args['--expid']
   hf.attrs['repid']      = args['--repid']
   hf.attrs['fpr']        = args['--fpr']
   hf.attrs['fnr']        = args['--fnr']
   hf.attrs['trace']      = args['--trace']
   hf.attrs['quiet']      = args['--quiet']
   hf.attrs['replicates'] = df.shape[0]
   hf.attrs['variables']  = df.shape[1]
   if args['--standard']:
      hf.attrs['standard']= args['--standard']

   # save the results as datasets
   hf.create_dataset('fpr_trace', data=results[0])
   hf.create_dataset('fnr_trace', data=results[1])
   hf.create_dataset('vec_mean', data=np.mean(results[2], axis=0))
   if args['--trace']:
      hf.create_dataset('vec_trace', data=results[2])

   # close the file
   hf.close()

   # append save the dataset itself?
   # no. DataFrame.to_hdf needs pytables to work, which wont build
   # we are saving the input file, and the spec string, should be
   # enough to recreate the dataset exactly

   return


# MAIN #######################################################################
if __name__ == '__main__':
   args = docopt(doc)
   #  print(args)

   if args['run']:

      validate_run(args)
      df = parse_data(args)

      standard = None
      if args['--standard']:
         hf = h5py.File(args['--standard'], 'r')
         standard = np.round(hf['vec_mean'])

      results = mcmc_fit(df, 
         burn=int(args['--burn']), iters=int(args['--iterations']),
         fpr=args['fpr'], fnr=args['fnr'],
         quiet=args['--quiet'], standard=standard)

      save_results(args, df, results)

   elif args['view']:
      view_traces(hfile=args['<output>'], save_pdf=None)
   else:
      print(doc)



   
