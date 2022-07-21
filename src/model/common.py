import numpy as np
import pandas as pd
import csv
import glob
import random


# TODO
GPWD = "./data/genesort_string_hit.txt"


def hp_symbol_set_load(gene_filter=None, string_gene_list_pwd=GPWD, delimiter=" "):
    string_set = set()
    with open(string_gene_list_pwd) as open_fd:
        data = csv.reader(open_fd, delimiter=delimiter)
        for x in data:
            string_set.add(x[0])
    return string_set


## imputation util code end
def sample_test_split(geo, num_of_class_test, num_of_example, num_of_testing, label_dic=False, pp=False):
    class_folders = geo.keys()
    class_folders = random.sample(class_folders, num_of_class_test)
    if label_dic:
        labels_to_text = label_dic
        labels_converter = {value:key for key, value in label_dic.items()}
    else:
        labels_converter = np.array(range(len(class_folders)))
        labels_converter = dict(zip(class_folders, labels_converter))
        labels_to_text = {value:key for key, value in labels_converter.items()}

    example_set = pd.DataFrame()
    test_set = pd.DataFrame()
    example_label = []
    test_label = []

    # balance sampler
    for subtype in labels_converter.keys():
        this_exp = geo[subtype]
        #this_exp.reset_index(drop=True, inplace=True) 
        total_colno = (this_exp.shape)[1]
        col_nu = list(range(total_colno) )
        random.shuffle(col_nu)
        assert(len(col_nu) >= num_of_example+num_of_testing), [total_colno, num_of_example+num_of_testing, subtype]
        example_ids = col_nu[0 : num_of_example]
        ex = this_exp.iloc[:,example_ids]
        test_ids = col_nu[num_of_example : num_of_example + num_of_testing]
        te = this_exp.iloc[:,test_ids]
        
        example_set = pd.concat([example_set,ex],axis=1)
        test_set = pd.concat([test_set, te],axis=1)
        example_label += [labels_converter[subtype]] * num_of_example
        test_label += [labels_converter[subtype]] * num_of_testing
    
    test_set = test_set.transpose()
    test_set['label'] = test_label
    test_set = test_set.sample(frac=1)
    test_label = test_set['label']
    test_set = test_set.drop(columns='label')
    test_set = test_set.transpose()   

    out_ex = example_set
    out_te = test_set
    
    if pp == True:
        print(out_ex.index)
        print(out_te.index)

    out_ex = out_ex.values
    out_te = out_te.values
    return out_ex, example_label, out_te, test_label, labels_to_text


def geo_data_loader(root_dir, pref=None, string_set=None, data_transformation=None):
    # merged
    # single_cell pancreas data scan/read
    pd_list = dict()

    file_list = glob.glob(root_dir + "*.csv.pd")

    #print(file_list)
    for type_data in file_list:
        this_pd = pd.read_pickle(type_data)
        class_name = type_data[len(root_dir):]
        class_name = class_name.split('.csv')[0]
        if pref is not None:
            class_name = class_name + "_" + str(pref)
        
        if (string_set is not None):
            if (len(this_pd.index.intersection(string_set)) == 0):
                this_pd = this_pd.transpose()
                print('transposed', type_data)
                assert(len(this_pd.index.intersection(string_set)) != 0), "exp array has not symbol"
            
            #print('ori', this_pd.shape, this_pd.index)
            this_pd = this_pd[~this_pd.index.duplicated(keep='first')]
            this_pd = this_pd.transpose()
            this_pd = this_pd.filter(items=string_set)
            this_pd = this_pd[~this_pd.index.duplicated(keep='first')]
            this_pd = this_pd.transpose()

            out_pd = pd.DataFrame(index=string_set)
            out_pd = pd.concat([out_pd, this_pd],axis=1)
            out_pd = out_pd.replace(np.nan,0)
            
            if data_transformation is not None:
                out_pd = out_pd.transpose()
                out_pd = data_transformation(out_pd)
                out_pd = out_pd.transpose()
            
            pd_list[class_name] = out_pd
        else:
            assert(True, 'ToDo @ geo_data_loader')
            this_pd = this_pd[~this_pd.index.duplicated(keep='first')]
            this_pd = this_pd.transpose()
            this_pd = this_pd[~this_pd.index.duplicated(keep='first')]
            out_pd = this_pd.transpose()
            out_pd = out_pd.replace(np.nan,0)
            
            pd_list[class_name] = out_pd
        #print(this_pd.shape)
    return pd_list


