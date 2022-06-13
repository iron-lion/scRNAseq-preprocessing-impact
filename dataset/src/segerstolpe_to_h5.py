import sys
import glob
import pandas as pd
import numpy as np
import h5py

file_list = sys.argv[2]
print(file_list)

LABEL = 'Characteristics[cell_type]'
total_data = pd.DataFrame()
for ff in [file_list]:
    data = pd.read_csv(ff, sep='\t', index_col=0)
    total_data = total_data.append(data)
    print(total_data.shape)

labels = pd.read_csv(sys.argv[3], sep='\t', index_col=0, header=0)
labels = labels[LABEL]
print(set(labels))

total_data = total_data.iloc[:,1:]
total_data = total_data.append(labels.transpose())

## ERCC cleaning
genes = total_data.index
ercc_list = []
for g in genes:
    if 'ERCC' in g:
        ercc_list.append(g)
print(len(ercc_list))
print(total_data.shape)
total_data = total_data.drop(ercc_list, axis=0)
print(total_data.shape)

total_data=total_data.transpose()

test = total_data.loc[total_data[LABEL] != 'not_applicable',]
print(test.shape)
test = test.loc[total_data[LABEL] != 'unclassified_cell',]
print(test.shape)
test = test.loc[total_data[LABEL] != 'unclassified_endocrine_cell',]
print(test.shape)
total_data = test

labels = total_data[LABEL]
labels = labels.values.tolist()
total_data.pop(LABEL)
total_data.pop('eGFP')
total_data = total_data.astype('float32')
print(set(labels))

data_columns = total_data.columns.astype('S').tolist()
data_index = total_data.index.astype('S').tolist()

hf = h5py.File(sys.argv[1], 'w')
hf.create_dataset('data', data=total_data)
hf.create_dataset('column', data=data_columns)
hf.create_dataset('index', data=data_index)
hf.create_dataset('label', data=pd.DataFrame(labels).values.astype('S'))
hf.close()
