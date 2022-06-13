import sys
import glob
import pandas as pd
import numpy as np
import h5py

file_list = sys.argv[2:]
print(file_list)

total_data = pd.DataFrame()
for ff in file_list:
    data = pd.read_csv(ff, sep=',', index_col=0, header=0)
    total_data = total_data.append(data)
    print(total_data.shape)

labels = total_data['assigned_cluster'].values.tolist()
label_set = set(labels)
print(label_set)

total_data = total_data.iloc[:,2:]





hf = h5py.File(sys.argv[1], 'w')
hf.create_dataset('data', data=total_data)
hf.create_dataset('label', data=pd.DataFrame(labels).values.astype('S'))
hf.close()
