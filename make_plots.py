import pandas as pd
import numpy as np
import seaborn as sbn
import time
import os
import matplotlib.pyplot as plt
import seaborn as sbn

start_time = time.time()

## ### # ### ##
# Import Data #
## ### # ### ##

print('Loading files... (' + str(round(time.time() - start_time, 2)) + 's elapsed)')

csvs = []

for root, dir, files in os.walk(os.getcwd() + '/csv_outputs'):
   for file in files:
      if file.endswith('unambiguous_words.csv') or file.startswith('morphemes.csv'):
         csvs.append(file)

# ### ## ### #
# Make Plots #
# ### ## ### #

print('Making plots... (' + str(round(time.time() - start_time, 2)) + 's elapsed)')

def make_plot(plot, x, y, title, log = False):
   plot.set(xlabel = x, ylabel = y)
   plot.set_title(title)

   if log:
      plot.set_yscale('log')

   return plot

bins, maxes, labs = {}, {}, {}

for csv in csvs:
   print('  Making plot for ' + csv + '... (' + str(round(time.time() - start_time, 2)) + 's elapsed)')

   df = pd.read_csv(os.path.join(root, csv))

   sbn.set_theme(style = 'whitegrid')
   fig, axs = plt.subplots(nrows = 3, ncols = 2)
   fig.set_size_inches(20, 15)

   x = 'gloss' if 'morpheme' in csv else 'gr.pos'

   make_plot(sbn.countplot(data = df, x = x, palette = 'cool', ax = axs[0][0]), '', 'count', 'Frequency of Items by POS')
   make_plot(sbn.countplot(data = df, x = x, palette = 'cool', ax = axs[0][1]), '', 'count', 'Frequency of Items by POS [log]', True)
   make_plot(sbn.barplot(data = df, x = x, y = 'freq', palette = 'cool', ci = None, estimator = sum, ax = axs[1][0]), '', 'freq', 'Total Frequency of Items by POS')
   make_plot(sbn.barplot(data = df, x = x, y = 'freq', palette = 'cool', ci = None, estimator = sum, ax = axs[1][1]), '', 'freq', 'Total Frequency of Items by POS [log]', True)
   make_plot(sbn.barplot(data = df, x = x, y = 'freq', palette = 'cool', ci = None, ax = axs[2][0]), '', 'count', 'Average Frequency of Items by POS (Uniqueness)')
   make_plot(sbn.barplot(data = df, x = x, y = 'freq', palette = 'cool', ci = None, ax = axs[2][1]), '', 'count', 'Average Frequency of Items by POS (Uniqueness) [log]', True)
   
   fig.savefig(os.getcwd() + '/img_outputs/' + csv[:-4] + '.png')

   # Get sampling data
   total_choices = 1000

   bins[csv] = np.log([h.get_height() for h in axs[0][1].patches])
   bins[csv] = np.round(np.divide(bins[csv], sum(bins[csv]) / total_choices)).astype(int)

   bins[csv] = [b + 1 if b == 0 else b for b in bins[csv]]

   # Make sure the total is the exact amount desired
   rounderror = sum(bins[csv]) - total_choices
   if rounderror > 0:
      for i in range(rounderror):
         is_one = True
         while is_one:
            idx = np.random.choice(np.arange(len(bins[csv])))
            if bins[csv][idx] != 1:
               bins[csv][idx] -= 1
               is_one = False
   if rounderror < 0:
      for i in range(rounderror * -1):
         bins[csv][np.random.choice(np.arange(len(bins[csv])))] += 1

   maxes[csv] = [h.get_height() for h in axs[0][0].patches]

   # Remove bin numbers that go over the actual number and redistribute
   extra = 0
   full = []
   for i in range(len(bins[csv])):
      if bins[csv][i] >= maxes[csv][i]:
         extra += bins[csv][i] - maxes[csv][i]
         bins[csv][i] = maxes[csv][i]
         full.append(i)

   while extra != 0 and len(full) != len(bins[csv]):
      for i in range(len(bins[csv])):
         if i not in full:
            bins[csv][i] += 1
            extra -= 1

            if extra == 0:
               break

            if bins[csv][i] == maxes[csv][i]:
               full.append(i)

   labs[csv] = [lab.get_text() for lab in axs[1][1].get_xticklabels()]

## ### ## ### ##
# Sample Items #
## ### ## ### ##

print('Sampling items... (' + str(round(time.time() - start_time, 2)) + 's elapsed)')

for csv in csvs:
   print('  Sample items for ' + csv + '... (' + str(round(time.time() - start_time, 2)) + 's elapsed)')

   df = pd.read_csv(os.path.join(root, csv))

   dfs = []
   x = 'gloss' if 'morpheme' in csv else 'gr.pos'

   for bin, lab in zip(bins[csv], labs[csv]):
      curr_pos = df[df[x] == lab]
      idxs = np.random.choice(curr_pos.index, bin, p = curr_pos['freq'] / sum(curr_pos['freq']), replace = False)

      dfs.append(df.iloc[idxs])

   pd.concat(dfs).to_csv(os.getcwd() + '/samples/sampled_' + csv, index = False) # Export to csv

print('Finished. (' + str(round(time.time() - start_time, 2)) + 's elapsed)')