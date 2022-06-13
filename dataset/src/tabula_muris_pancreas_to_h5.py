import sys
import glob
import pandas as pd
import numpy as np
import h5py

file_list = glob.glob(sys.argv[2])

total_data = pd.DataFrame()
for ff in file_list:
    data = pd.read_csv(ff, sep=',', index_col=0, header=0)
    data = data.transpose()
    c = list(range(data.shape[0]))
    total_data = pd.concat([total_data, data.iloc[c,]], axis=0)


print(total_data)
annot_label = pd.read_csv(sys.argv[3], sep=',', index_col=2, header=0)
com = annot_label['tissue'].index.intersection(total_data.index)
m = annot_label.filter(com, axis=0)
print(m)
total_data = total_data.filter(com, axis=0)
total_data = pd.concat([total_data, m['cell_ontology_class']],axis =1)

labels = total_data['cell_ontology_class']
labels = labels.values.tolist()

total_data.pop('cell_ontology_class')
total_data.pop('zsGreen_transgene')
print(set(labels))





hf = h5py.File(sys.argv[1], 'w')
hf.create_dataset('data', data=total_data)
hf.create_dataset('label', data=pd.DataFrame(labels).values.astype('S'))
hf.close()
