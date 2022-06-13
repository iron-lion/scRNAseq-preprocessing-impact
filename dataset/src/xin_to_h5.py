import sys
import glob
import pandas as pd
import numpy as np
import h5py

file_list = sys.argv[2:]
print(file_list)

LABEL = 'cell.type'

total_data = pd.DataFrame()
for ff in file_list:
    data = pd.read_csv(ff, sep='\t', index_col=0)
    total_data = total_data.append(data)
    print(total_data.shape)

gene_anno = pd.read_csv('../dataset/xin/RAW/human_gene_annotation.csv', sep=",", index_col=0)

total_data = pd.concat([gene_anno, total_data], axis=1)
total_data.set_index('symbol', inplace=True)


labels = pd.read_csv('../dataset/xin/RAW/human_islet_cell_identity.txt', sep='\t', index_col=0)
labels = labels[LABEL]

#total_data = total_data.iloc[:,1:]
#total_data.columns = [k.split(".")[1] for k in total_data.columns]

total_data = total_data.append(labels)
total_data=total_data.transpose()
#print(total_data)

labels = total_data[LABEL].values.tolist()
total_data.pop(LABEL)
total_data = total_data.astype('float32')
#print(total_data)
print(set(labels))

data_columns = total_data.columns.astype('S').tolist()
data_index = total_data.index.astype('S').tolist()

hf = h5py.File(sys.argv[1], 'w')
hf.create_dataset('data', data=total_data)
hf.create_dataset('column', data=data_columns)
hf.create_dataset('index', data=data_index)
hf.create_dataset('label', data=pd.DataFrame(labels).values.astype('S'))
hf.close()
