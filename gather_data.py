import pandas as pd
import numpy as np
import json
import os
import time
import ast

start_time = time.time()

## ### # ### ##
# Gather Data #
## ### # ### ##

print('Loading files... (' + str(round(time.time() - start_time, 2)) + 's elapsed)')

jsons = []

for root, dir, files in os.walk(os.getcwd() + '/corpus'): # Remove /parallel for whole corpus
   for file in files:
      if file.endswith('.json'):
         jsons.append(os.path.join(root, file))

## ### ## ### ##
# Process Data #
## ### ## ### ##

def get_words(json):
   keys = ['lex', 'gr.pos', 'parts', 'gloss', 'trans_ru']
   
   # Create dicts for each type of data
   data = {'word': [], 'lex': [], 'gr.pos': [], 'parts': [], 'gloss': [], 'trans_ru': []}
   ambig_data = {'word': [], 'lex': [], 'gr.pos': [], 'parts': [], 'gloss': [], 'trans_ru': []}
   miss_data = {'word': [], 'lex': [], 'gr.pos': [], 'parts': [], 'gloss': [], 'trans_ru': []}

   # Iterate over all the data, extract the desired pieces, and store them in the appropriate dict
   for sentence in json['sentences']:
      for word in sentence['words']:
         if 'ana' in word:
            for lex in word['ana']:
               if all(e in lex and type(lex[e]) != list and lex[e] != '???' for e in keys): # If no data is missing and the extracted data is all unambiguous (has exactly 1 option)
                  data['word'].append(word['wf'])
                  for e in keys:
                     data[e].append(lex[e])
               elif all(e in lex and lex['gloss'] != '???' for e in keys): # If no data is missing, but at least 1 of the extracted data is ambiguous (has more than 1 option)
                  ambig_data['word'].append(word['wf'])
                  for e in keys:
                     ambig_data[e].append(lex[e])
               else: # Missing pieces of data we are looking for
                  miss_data['word'].append(word['wf'])
                  for e in keys:
                     miss_data[e].append(lex[e]) if e in lex else miss_data[e].append('')
                  
   return pd.DataFrame(data), pd.DataFrame(ambig_data), pd.DataFrame(miss_data) # return as DataFrames

print('Creating DataFrames from the files... (' + str(round(time.time() - start_time, 2)) + 's elapsed)')

# Create the Dataframes from data across all the json files
words = pd.DataFrame({'word': [], 'lex': [], 'gr.pos': [], 'parts': [], 'gloss': [], 'trans_ru': []})
ambig_words = pd.DataFrame({'word': [], 'lex': [], 'gr.pos': [], 'parts': [], 'gloss': [], 'trans_ru': []})
miss_words = pd.DataFrame({'word': [], 'lex': [], 'gr.pos': [], 'parts': [], 'gloss': [], 'trans_ru': []})

for jsn in jsons:
   jsn = json.load(open(jsn))
   words_tmp, ambig_words_tmp, miss_words_tmp = get_words(jsn)

   words_tmp = words_tmp.astype(str)
   words_tmp.loc[words_tmp['lex'].str.contains(r'\[.*\]'), 'lex'] = words_tmp.loc[words_tmp['lex'].str.contains(r'\[.*\]'), 'lex'].str[1:-1] # Remove brackets if present
   words_tmp['word'] = words_tmp['word'].str.lower() # Normalize case
   words = pd.concat([words, words_tmp[~words_tmp['word'].str.contains(r'\d')]]) # Concat non-numeric rows

   ambig_words_tmp = ambig_words_tmp.astype(str)
   ambig_words_tmp.loc[ambig_words_tmp['lex'].str.contains(r'\[.*\]'), 'lex'] = ambig_words_tmp.loc[ambig_words_tmp['lex'].str.contains(r'\[.*\]'), 'lex'].str[1:-1]
   ambig_words_tmp['word'] = ambig_words_tmp['word'].str.lower()   
   ambig_words = pd.concat([ambig_words, ambig_words_tmp[~ambig_words_tmp['word'].str.contains(r'\d')]])
   
   miss_words_tmp = miss_words_tmp.astype(str)
   miss_words_tmp.loc[miss_words_tmp['lex'].str.contains(r'\[.*\]'), 'lex'] = miss_words_tmp.loc[miss_words_tmp['lex'].str.contains(r'\[.*\]'), 'lex'].str[1:-1]
   miss_words_tmp['word'] = miss_words_tmp['word'].str.lower()   
   miss_words = pd.concat([miss_words, miss_words_tmp[~miss_words_tmp['word'].str.contains(r'\d')]])

# Reset Indices and standardize case
words = words.reset_index(drop = True)
ambig_words = ambig_words.reset_index(drop = True)
# miss_words = miss_words.reset_index(drop = True)

print('Creating word frequency csvs... (' + str(round(time.time() - start_time, 2)) + 's elapsed)')

# Generate affix information
def affixate(lst):
   passed_stem = False
   affixes = []
   for e in lst:
      if e == 'STEM':
         affixes.append('root')
         passed_stem = True
      elif passed_stem: 
         affixes.append('suffix')
      else:
         affixes.append('prefix')
   return affixes

# Show frequency counts for the words; plot and generate csvs
for name, df in {'unambiguous': words, 'ambiguous': ambig_words, 'missing': miss_words}.items():
   df_freq = df.astype(str)
   df_freq = pd.DataFrame({'freq': df_freq.groupby(['word', 'lex', 'gr.pos', 'parts', 'gloss', 'trans_ru']).size()}).reset_index().sort_values('freq', ascending = False)
   
   df_freq['parts'] = df_freq['parts'].str.split('-') # Split segmented data into lists
   df_freq['gloss'] = df_freq['gloss'].str.split('-')
   df_freq['affix'] = df_freq['gloss'].apply(affixate)
   
   # Convert tags to Universal Tag Set
   df_freq = df_freq.replace(['PRO', 'NPRO'], 'PRON')
   df_freq = df_freq.replace('N', 'NOUN')
   df_freq = df_freq.replace(['A', 'APRO'], 'ADJ')
   df_freq = df_freq.replace('V', 'VERB')
   df_freq = df_freq.replace('PTCL', 'PART')
   df_freq = df_freq.replace('POST', 'ADP')
   df_freq = df_freq.replace('ANUM', 'NUM')

   # Correct incorrect POS tagging
   df_freq.loc[np.logical_and(np.logical_or(df_freq['gr.pos'] == 'NOUN', df_freq['gr.pos'] == 'PRON'), df_freq['gloss'].apply(lambda x: x[-1]) == np.repeat('ADV', len(df_freq.index))), 'gr.pos'] = 'ADV'
   df_freq.loc[np.logical_and(np.logical_or(df_freq['gr.pos'] == 'NOUN', df_freq['gr.pos'] == 'PRON'), df_freq['gloss'].apply(lambda x: x[-1]) == np.repeat('PRED', len(df_freq.index))), 'gr.pos'] = 'VERB'

   df_freq.to_csv(os.getcwd() + '/csv_outputs/' + name + '_words.csv', index = False) # Export to csv

print('Creating morpheme frequency csvs... (' + str(round(time.time() - start_time, 2)) + 's elapsed)')

def unzip(iter):
   return list(zip(*iter))

words = pd.concat([words, ambig_words, miss_words]).reset_index(drop = True)
lemmata = words.replace(r'^\s*$', np.nan, regex = True).dropna()
miss_lemmata = words[~words.index.isin(lemmata.index)]

# Remove from memory once unneeded
words = None
ambig_words = None
miss_words = None

# Show frequency counts for the morphemes
for name, df in {'': lemmata, 'missing_': miss_lemmata}.items():
   df_freq = pd.DataFrame({'freq': df.groupby(['word', 'lex', 'gr.pos', 'parts', 'gloss', 'trans_ru']).size()}).reset_index()

   df_freq['parts'] = df_freq['parts'].str.split('-') # Split segmented data into lists
   df_freq['gloss'] = df_freq['gloss'].str.split('-')
   df_freq['affix'] = df_freq['gloss'].apply(affixate)

   df_freq['segmented'] = list(zip(df_freq['parts'], df_freq['gloss'], df_freq['affix'])) # Rework DataFrame to be for lexes
   df_freq['segmented'] = df_freq['segmented'].apply(unzip).apply(list)
   df_freq = df_freq.explode('segmented')
   df_freq['parts'], df_freq['gloss'], df_freq['affix'] = unzip(df_freq['segmented'])

   df_freq_bad_gloss = df_freq[df_freq['gloss'].str.contains(r'а|б|в|г|ъ|д|ж|з|е|ё|ж|з|и|й|к|л|м|н|о|п|р|с|т|ӏ|ф|х|ц|ч|ш|щ|ы|ь|э|ю|я|у')].copy()

   df_freq = df_freq.drop(columns = ['segmented', 'word', 'lex', 'gr.pos', 'trans_ru']).rename(columns = {'parts': 'morpheme'})
   df_freq_bad_gloss = df_freq_bad_gloss.drop(columns = ['segmented', 'word', 'lex', 'gr.pos']).rename(columns = {'parts': 'morpheme', 'trans_ru': 'word_trans_ru'})

   df_freq = df_freq.groupby(['morpheme', 'gloss', 'affix']).sum().reset_index().sort_values('freq', ascending = False) # Re-group by frequency
   df_freq_bad_gloss = df_freq_bad_gloss.groupby(['morpheme', 'gloss', 'affix']).sum().reset_index().sort_values('freq', ascending = False) # Re-group by frequency

   df_freq.to_csv(os.getcwd() + '/csv_outputs/' + name + 'morphemes.csv', index = False) # Export to csv
   df_freq_bad_gloss.to_csv(os.getcwd() + '/csv_outputs/' + name + 'poorly_glossed_morphemes.csv', index = False) # Export to csv

# Remove from memory once unneeded
lemmata = None
miss_lemmata = None

def get_sentences(json):
   keys = ['gr.pos', 'gloss', 'trans_ru']
   
   # Create dicts for each type of data
   data = {'sentence': [], 'gr.pos': [], 'gloss': [], 'trans_ru': []}
   ambig_data = {'sentence': [], 'gr.pos': [], 'gloss': [], 'trans_ru': []}

   # Iterate over all the data, extract the desired pieces, and store them in the appropriate dict
   for sentence in json['sentences']:
      contains_ambig = False
      skip = False
      sen = {'sentence': [], 'gr.pos': [], 'gloss': [], 'trans_ru': []}

      for word in sentence['words']:
         if skip: 
            break

         wrd = {'word': '', 'gr.pos': [], 'gloss':[], 'trans_ru': []}

         if 'ana' in word:
            wrd['word'] = word['wf']

            for lex in word['ana']:
               if all(e in lex and type(lex[e]) != list and lex[e] != '???' and lex[e] != '' for e in keys): # If no data is missing and the extracted data is all unambiguous (has exactly 1 option)
                  if (lex['gr.pos'] == 'N' or lex['gr.pos'] == 'PRO' or lex['gr.pos'] == 'NPRO') and lex['gloss'].split('-')[-1] == 'ADV': # Correct POS
                     lex['gr.pos'] = 'ADV'
                  if (lex['gr.pos'] == 'N' or lex['gr.pos'] == 'PRO' or lex['gr.pos'] == 'NPRO') and lex['gloss'].split('-')[-1] == 'PRED':
                     lex['gr.pos'] = 'V'
                  
                  for e in keys:
                     wrd[e].append(lex[e])
               elif all(e in lex and lex[e] != '???' and lex[e] != '' for e in keys): # If no data is missing, but at least 1 of the extracted data is ambiguous (has more than 1 option)
                  contains_ambig = True
                  for e in keys:
                     wrd[e].append(lex[e]) 
               else:
                  skip = True
                  break
         
            sen['sentence'].append(wrd['word'])
            sen['gr.pos'].append(wrd['gr.pos'] if len(wrd['gr.pos']) != 1 else wrd['gr.pos'][0])
            sen['gloss'].append(wrd['gloss'] if len(wrd['gr.pos']) != 1 else wrd['gloss'][0])
            sen['trans_ru'].append(wrd['trans_ru'] if len(wrd['trans_ru']) != 1 else wrd['trans_ru'][0])
         elif word['wtype'] == 'punct':
            sen['sentence'].append(word['wf'])
            sen['gr.pos'].append('PUNCT')
            sen['gloss'].append('PUNCT')
            sen['trans_ru'].append(word['wf'])
         else:
            skip = True
      
      if skip:
         None
      elif contains_ambig:
         ambig_data['sentence'].append(sen['sentence'])
         ambig_data['gr.pos'].append(sen['gr.pos'])
         ambig_data['gloss'].append(sen['gloss'])
         ambig_data['trans_ru'].append(sen['trans_ru'])
      else:
         data['sentence'].append(sen['sentence'])
         data['gr.pos'].append(sen['gr.pos'])
         data['gloss'].append(sen['gloss'])
         data['trans_ru'].append(sen['trans_ru'])
                  
   return pd.DataFrame(data), pd.DataFrame(ambig_data) # return as DataFrames

print('Creating sentence csvs... (' + str(round(time.time() - start_time, 2)) + 's elapsed)')

# Create the Dataframes from data across all the json files
sentences = pd.DataFrame({'sentence': [], 'gr.pos': [], 'gloss': [], 'trans_ru': []})
ambig_sentences = pd.DataFrame({'sentence': [], 'gr.pos': [], 'gloss': [], 'trans_ru': []})

for jsn in jsons:
   jsn = json.load(open(jsn))
   sentences_tmp, ambig_sentences_tmp = get_sentences(jsn)

   if len(sentences_tmp.columns) != 0:
      sentences_tmp = sentences_tmp.astype(str)
      sentences = pd.concat([sentences, sentences_tmp[~sentences_tmp['sentence'].str.contains(r'\d|(?:\\\\n)')]]) # Concat non-numeric rows that do not have a newline character

   if len(ambig_sentences_tmp.columns) != 0:
      ambig_sentences_tmp = ambig_sentences_tmp.astype(str)  
      ambig_sentences = pd.concat([ambig_sentences, ambig_sentences_tmp[~ambig_sentences_tmp['sentence'].str.contains(r'\d')]])

# Reset Indices
sentences = sentences.reset_index(drop = True)
ambig_sentences = ambig_sentences.reset_index(drop = True)

# Show frequency counts for the sentences
for name, df in {'unambiguous': sentences, 'ambiguous': ambig_sentences}.items():
   df = df.replace(['\'PRO\'', '\'NPRO\''], '\'PRON\'', regex = True)
   df = df.replace('\'N\'', '\'NOUN\'', regex = True)
   df = df.replace(['\'A\'', '\'APRO\''], '\'ADJ\'', regex = True)
   df = df.replace('\'V\'', '\'VERB\'', regex = True)
   df = df.replace('\'PTCL\'', '\'PART\'', regex = True)
   df = df.replace('\'POST\'', '\'ADP\'', regex = True)
   df = df.replace('\'ANUM\'', '\'NUM\'', regex = True)

   df_freq = pd.DataFrame({'freq': df.astype('str').groupby(['sentence', 'gr.pos', 'gloss', 'trans_ru']).size()}).reset_index().sort_values('freq', ascending = False)

   df_freq.to_csv(os.getcwd() + '/csv_outputs/' + name + '_sentences.csv', index = False) # Export to csv

print('Finished. (' + str(round(time.time() - start_time, 2)) + 's elapsed)')