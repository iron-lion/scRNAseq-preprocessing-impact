import sys
import glob
import pandas as pd
import numpy as np
import h5py

file_list = sys.argv[2:]
print(file_list)

total_data = pd.DataFrame()
for ff in file_list:
    data = pd.read_csv(ff, sep='\t', index_col=0)
    total_data = total_data.append(data)
    print(total_data.shape)

new_index = []
tmp_index = total_data.index
for ll in tmp_index:
    new_index.append(ll.split("__")[0])
total_data.index = new_index


labels = pd.read_csv('../dataset/muraro/RAW/cell_type_annotation_Cels2016.csv', sep='\t', index_col=0)
labels.columns = ['labels']
label_colname = labels.columns[0]
labels.index = [l.replace(".","-") for l in labels.index]
labels = labels.transpose().filter(items=total_data.columns)
#labels = labels[0]

total_data = total_data.append(labels)
total_data = total_data.transpose()
#print(total_data)
total_data[label_colname] = total_data[label_colname].replace(np.nan, 'nolabel')
label_set = set(labels)
print(len(labels))

us = total_data[label_colname] != 'nolabel'
total_data = total_data.loc[us,]
labels = total_data[label_colname]
total_data.pop(label_colname)
total_data = total_data.astype('float32')
#print(total_data)

labels = labels.values.tolist()

data_columns = total_data.columns.astype('S').tolist()
data_index = total_data.index.astype('S').tolist()

hf = h5py.File(sys.argv[1], 'w')
hf.create_dataset('data', data=total_data)
hf.create_dataset('column', data=data_columns)
hf.create_dataset('index', data=data_index)
hf.create_dataset('label', data=pd.DataFrame(labels).values.astype('S'))
hf.close()
