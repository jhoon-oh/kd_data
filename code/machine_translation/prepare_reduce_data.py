import os 
import numpy as np
import shutil


if __name__ =='__main__':

    path = './examples/translation/wmt14_ende_distill'
    file_name = ['train.en-de.en','train.en-de.de']

    with open(os.path.join(path,file_name[0]), 'r') as f:
        english_lines = f.readlines()

    with open(os.path.join(path,file_name[1]), 'r') as f:
        deutch_lines = f.readlines()

    p_values = np.linspace(0,1,11)
    p_values = [round(p,2) for p in p_values]
    for p in p_values:
        if p == 0.0 or p == 1.0 : continue 

        target_path = path + '_' + str(p)

        ## English File
        end_idx = int(len(english_lines) * p)
        temp_lines = english_lines[:end_idx]
        target_path = path + '_' + str(p)
        if not os.path.exists(target_path) : os.mkdir(target_path)
        with open(os.path.join(target_path,file_name[0]), 'w') as f: f.writelines(line for line in temp_lines)

        ## German Lines
        de_idx = int(len(deutch_lines) * p)
        temp_lines = deutch_lines[:de_idx]
        target_path = path + '_' + str(p)
        if not os.path.exists(target_path) : os.mkdir(target_path)
        with open(os.path.join(target_path,file_name[1]), 'w') as f: f.writelines(line for line in temp_lines)

        print("Reducing Data : ", p)
