import pandas as pd
import numpy as np
import ast

csv = pd.read_csv('csv_outputs/unambiguous_sentences.csv')

### ## ### ## ### ## ###
# Clean Part of Speech #
### ## ### ## ### ## ###

# Evaluate part of speech strings to 2d-lists
csv['gr.pos'] = csv['gr.pos'].apply(ast.literal_eval)

# Remove unnecessary ambiguities (this way, the part of speech options won't have multiple repeats of the same options)
csv['gr.pos'] = csv['gr.pos'].apply(lambda lst: [list(np.unique(sublst)) for sublst in lst])

# Unpack lists with 1 element
csv['gr.pos'] = csv['gr.pos'].apply(lambda lst: [sublst[0] if len(sublst) == 1 else sublst for sublst in lst])

## ### ## ## ## ### ## ## ##
# Filter Out Unneeded Data #
## ### ## ## ## ### ## ## ##

# Get only Sentence-POS data
csv = csv[['sentence', 'gr.pos']]

# Remove all unambiguous data
csv = csv[csv['gr.pos'].apply(lambda lst: np.all([type(e) != list for e in lst]))]

# Export to csv
csv.to_csv('cleaned_sentences/' + 'unambiguous_parts_of_speech.csv', index = False)

# Export sample to csv
sampled_csv = csv.iloc[np.random.choice(len(csv.index), 200)]
sampled_csv.to_csv('cleaned_sentences/' + 'sampled_unambiguous_parts_of_speech.csv', index = False)