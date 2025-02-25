from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import os
import numpy as np
import sys
import joblib
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn

from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler

# for DNN 
from Net import Net
from calculate_descriptors import calculate_descriptors_for_single_smiles, calculate_descriptors_for_multiple_smiles
from eval_utils import predict_single_query, predict_query_by_all_models

def main():
    parser = ArgumentParser(description="Forward prediction by BBB model")
    parser.add_argument('query_smiles_filepath', type=Path, help="Path to dataset")
    parser.add_argument('model_dir', type=Path, help="Path to model directory")
    parser.add_argument('output_dir', type=Path, help="Path to output storage dir")
    args = parser.parse_args()
    
    # load smiles
    #data = pd.read_excel(args.query_smiles_filepath, header=None)
    data = pd.read_excel(args.query_smiles_filepath, sheet_name=0)
    data.columns = ['SMILES','ACTIVITY']
    print(f'input_data : \n{data}')
    # only keep columns that are present in descr_names table
    int_features_main = calculate_descriptors_for_multiple_smiles(data.loc[:,['SMILES']])
    print(f'Descriptor calculation done! Total descriptors = {int_features_main.shape[1]}')
    print(f'int_features_main : \n{int_features_main.head()}')
    
    # lets remove smiles for which missing descriptors got calculates 
    loc_for_missing_descr = int_features_main.index[int_features_main.isnull().any(axis=1)]
    print(f'loc_for_missing_descr : \n{loc_for_missing_descr}')
    smiles_with_missing_value_descriptors = pd.DataFrame(data.iloc[loc_for_missing_descr], columns=['SMILES','ACTIVITY'])
    print(f'\nTotal {len(smiles_with_missing_value_descriptors)} smiles had missing descriptor values \nsmiles:\n{smiles_with_missing_value_descriptors}\n')
    
    # lets drop those smiles from main input and r
    
    data = data[~data.index.isin(int_features_main.index[int_features_main.isnull().any(axis=1)])].reset_index(drop=True)
    print(f'\nRevised number of smiles = {len(data)}\nvalues:{data.head()}\n')
    int_features_main = int_features_main.dropna(how='any').reset_index(drop=True).rename(index=data.SMILES)
    print(f'\nRevised Descriptor shape {int_features_main.shape} \nvalues = {int_features_main.iloc[0:2,0:3]}')
    
    # get model directory 
    model_directory = args.model_dir
    print(f'\nFor given {len(data)} compunds, predictions are in progress using {model_directory}\n')
    
    # run predictions 
    output_classes, output_values  = predict_query_by_all_models(data.SMILES, int_features_main, 2, 10, model_directory)
    #output_batch_classes, output_batch_values, output_batch_original_values  = predict_query_by_all_models(int_features_main, 2, 10, model_directory)
    # get directory and cd
    path_dir = args.output_dir
    os.chdir(path_dir)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_classes.to_csv('Pred_classes_'+timestr+'.csv',header=True, index=True, index_label=['Query_SMILES'], sep=',')
    output_values.to_csv('Pred_values_'+timestr+'.csv',header=True, index=True,sep=',')
    smiles_with_missing_value_descriptors.to_csv('SMILES_that_cannot_be_screened_'+timestr+'.csv',header=True, index=True, index_label=['Entry'], sep=',')
    print('Job finished!')
    
if __name__ == '__main__':
    main()