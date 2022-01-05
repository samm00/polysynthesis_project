import pandas as pd
import numpy as np
import xmltodict

with open('bible_parallel.xml') as xml_file:
   data = xmltodict.parse(xml_file.read())['html']['body']['para']

data = [(d['@id'], d['se'][0]['#text'], d['se'][1]['#text']) for d in data if '#text' in d['se'][1].keys()]

data = pd.DataFrame(data, columns = ['id', 'Adyghe', 'Russian'])

data[['Ady_id', 'Adyghe']] = data['Adyghe'].str.extract(r'\[([\s\S]+\d)\] ([\s\S]+)')
data[['Rus_id', 'Russian']] = data['Russian'].str.extract(r'\[([\s\S]+\d)\] ([\s\S]+)')

# Ensure verse ids line up
ady_nums = data['Ady_id'].str.extract(r'(\d+:\d+)', expand = False)
rus_nums = data['Rus_id'].str.extract(r'(\d+:\d+)', expand = False)
print('Ids match') if ady_nums.equals(rus_nums) else print('Error: id mismatch')

# Convert verse ids to standard ids
conv = {'1Ин._': 'joshua??', '1Кор._': 'b.1CO.', '2Кор._': 'b.2CO.', '1Пет._': 'b.1PE.', '2Пет._': 'b.2PE.', '1Тим._': 'b.1TI.', '2Тим._': 'b.2TI.', '1Фес._': 'b.1TH.', '2Фес._': 'b.2TH.', '1Цар._': 'b.1SA.', '2Цар._': 'b.2SA.', '3Цар._': 'b.1KI.', '4Цар._': 'b.2KI.', 'Быт._': 'b.GEN.', 'Деян._': 'b.ACT.', 'Евр._': 'b.HEB.', 'Есф.': 'b.EST.', 'Иак.': 'b.JAM.', 'Ин._': 'b.JHN', 'Иуд.': 'b.JDE.', 'ККнига_пророка_Ионы_': 'b.JNH.', 'Лк._': 'b.LUK.', 'Мк._': 'b.MRK.', 'Мф._': 'b.MAT.', 'Откр._': 'b.REV.', 'Послание_к_Галатам_': 'b.GAL', 'Послание_к_Ефесянам_': 'b.EPH.', 'Послание_к_Колоссянам_': 'b.COL.', 'Послание_к_Титу_': 'b.TTS.', 'Послание_к_Филлипийцам_': 'b.PHP.', 'Рим._': 'b.ROM', 'Руфь_': 'b.RUT.', 'Филим._': 'b.PHM'}
data['verse'] = data['Rus_id']
for rus, eng in conv.items():
   data['verse'] = data['verse'].str.replace(rus, eng, regex = True)
data['verse'] = data['verse'].str.replace(':', '.', regex = True)

# Separate data for training
adyghe = data[['verse', 'Adyghe']]
russian = data[['verse', 'Russian']]

data = data.set_index('id') # set ids to index (others won't use index in output)

data.to_csv('csv_outputs/bible_parallel.csv')
adyghe.to_csv('csv_outputs/ady_bible_parallel.csv', index = None)
russian.to_csv('csv_outputs/rus_bible_parallel.csv', index = None)