import sys
import glob
import pandas as pd
import numpy as np
import h5py

file_list = sys.argv[2:]
print(file_list)


LABEL = 'Sample_characteristics_ch1'

total_data = pd.DataFrame()
for ff in file_list:
    data = pd.read_csv(ff, sep='\t', index_col=6)
    total_data = total_data.append(data)
    print(total_data.shape)

labels = pd.read_csv('../dataset/wang/RAW/label.txt', sep='\t', index_col=0)
labels2 = pd.read_csv('../dataset/wang/RAW/label2.txt', sep='\t', index_col=0)
labels = pd.concat([labels,labels2],axis=1)

labels.columns = [k.split("_")[1] for k in labels.columns]

#labels = labels[LABEL]
total_data = total_data.iloc[:,6:]
total_data.columns = [k.split(".")[1] for k in total_data.columns]

total_data = total_data.append(labels)
total_data=total_data.transpose()

total_data = total_data.loc[total_data[LABEL] != 'dropped',]
print(total_data.shape)

labels = total_data[LABEL]
labels = labels.values.tolist()
total_data.pop(LABEL)

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
