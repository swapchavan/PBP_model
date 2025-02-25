from argparse import ArgumentParser
from pathlib import Path

#from batch_dataset import batch_dataset
from Net import Net
from train_model import train_model
from remove_highly_corr_descr import remove_highly_corr_descr

import pandas as pd
import numpy as np
import os
import numpy as np
import random
import shutil
import joblib

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score, roc_auc_score, r2_score,explained_variance_score

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, BatchNorm1d
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler, DataLoader, TensorDataset
from torch.utils.data.dataset import ConcatDataset
import torch.optim as optim
from torch.backends import cudnn

from rdkit import Chem
from rdkit import RDLogger

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

RDLogger.DisableLog('rdApp.*')

def main():
    parser = ArgumentParser(description="Train a regression-classification multi-task RNN model")

    parser.add_argument('training_dataset_path', type=Path, help="Path to descriptor file")
    parser.add_argument('training_dataset_tox_classes_path', type=Path, help="Path to tox-classes")
    parser.add_argument('training_dataset_tox_values_path', type=Path, help="Path to tox-values")
    parser.add_argument('n_CV', type=int, help="Number of folds for cross-validation")
    parser.add_argument('n_rep', type=int, help="Number of repeated CV")
    parser.add_argument('--output_dir', type=Path, help="Root directory to store results to", default=Path('models'))
    parser.add_argument('--num-workers', help="Number of parallel workers for the dataloaders", type=int, default=0)
    parser.add_argument('--device', help='which device to use', default='cpu')
    args = parser.parse_args()

    path_dir = args.output_dir
    os.chdir(path_dir)

    n_fold_CV = int(args.n_CV)
    number_of_repeats = int(args.n_rep)

    device = args.device
    print(f"\nDevice is set to {device}")

    import warnings
    warnings.filterwarnings("ignore")

    # load descriptor file 
    mc_set = pd.read_csv(args.training_dataset_path, header = 0, index_col = 0)
    print(f'Imported data size : {mc_set.shape}')
    # remove nan values
    #mc_set = mc_set.replace([np.inf, -np.inf], np.nan).replace(["Infinity","-Infinity"],  np.nan).dropna(axis=1, how="any") # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    mc_set.replace([np.inf, -np.inf], np.nan, inplace=True)
    mc_set.dropna(axis=1, how="any", inplace=True)
    print(f'Imported data size after dropping missing value columns : {mc_set.shape}')
    
    # import classes & values 
    mc_set_classes = pd.read_csv(args.training_dataset_tox_classes_path, header = None, index_col = None)
    mc_set_classes.columns = ['CLASSES']
    mc_set_values = pd.read_csv(args.training_dataset_tox_values_path, header = None, index_col = None, keep_default_na=False)  # ,  keep_default_na=False, na_values=[np.nan]
    mc_set_values.columns = ['VALUES']
    # since there are missing mother/featus ratio for few compounds, lets replace them by numpy.nan 
    #print(f'values before : {mc_set_values}')
    mc_set_values['VALUES'] = mc_set_values['VALUES'].replace('NA', np.nan)
    mc_set_values['VALUES'] = mc_set_values['VALUES'].astype('float64')
    #print(f'values after : {mc_set_values}')
    # lets convert Y in range of 0 and 1
    Y_scaler = MinMaxScaler(feature_range=(0, 1)).fit(mc_set_values.values)
    mc_set_values = pd.DataFrame(Y_scaler.transform(mc_set_values.values), columns=mc_set_values.columns, index=None)
    #mc_set_values = pd.DataFrame(Y_scaler.inverse_transform(mc_set_values.values), columns=mc_set_values.columns, index=None) # to get original values back
    joblib.dump(Y_scaler,'Y_scaler.joblib')

    # devide in pre-train & test set
    pre_train_index, test_index = train_test_split(mc_set_classes.index, test_size=0.1, stratify=mc_set_classes.CLASSES, random_state=1)
    # slice pre-train set 
    X_pre_train, Y_class_pre_train, Y_value_pre_train = mc_set.iloc[pre_train_index], mc_set_classes['CLASSES'].iloc[pre_train_index], mc_set_values['VALUES'].iloc[pre_train_index]
    print(f'X_pre_train : {X_pre_train.shape} | Y_class_pre_train : {Y_class_pre_train.shape} | Y_value_pre_train : {Y_value_pre_train.shape}')
    # slice TEST SET
    X_TEST, Y_class_test, Y_value_test = mc_set.iloc[test_index], mc_set_classes['CLASSES'].iloc[test_index], mc_set_values['VALUES'].iloc[test_index]
    print(f'X_TEST : {X_TEST.shape} | Y_class_test : {Y_class_test.shape} | Y_value_test : {Y_value_test.shape}')
    # save pre-train and test set 
    pd.DataFrame(X_pre_train.index).to_csv('X_pre_train.csv', header=['SMILES'], index=False, sep=',')
    Y_class_pre_train.to_csv('Y_class_pre_train.csv', header=True, index=False, sep=',')
    Y_value_pre_train.to_csv('Y_value_pre_train.csv', header=True, index=False, sep=',')
    pd.DataFrame(X_TEST.index).to_csv('X_TEST.csv', header=['SMILES'], index=False, sep=',')
    Y_class_test.to_csv('Y_TEST_class.csv', header=True, index=False, sep=',')
    Y_value_test.to_csv('Y_TEST_value.csv', header=True, index=False, sep=',')
    
    # To save final model outcome-create a empty dataframe
    col_names =  ['CV_fold', 'train_ROC', 'train_R2', 'val_ROC', 'val_R2', 'test_ROC', 'test_R2']
    final_output = []
    total_rows = int(n_fold_CV)*int(number_of_repeats)
    final_output = pd.DataFrame(index=range(1,int(total_rows)), columns = col_names)

    #skf = StratifiedKFold(n_splits = int(n_fold_CV), shuffle=True, random_state=1234)
    rskf = RepeatedStratifiedKFold(n_splits=n_fold_CV, n_repeats=number_of_repeats, random_state=1)
    i = 0
    # run 5 fold CV. 
    # "rskf.split" will take smiles and binary classes as input and will provide serial number (train_indices, val_indices) as output for train & val set 
    for train_indices, val_indices in rskf.split(X_pre_train, Y_class_pre_train):
        
        i += 1
        if (number_of_repeats==5) and (n_fold_CV==5):
            if i < 6:
                r = 1
            elif (i > 5) and (i < 11):
                r = 2
            elif (i > 10) and (i < 16):
                r = 3
            elif (i > 15) and (i < 21):
                r = 4
            else:
                r = 5
        elif (number_of_repeats==2) and (n_fold_CV==10):
            if i < 11:
                r = 1
            else:
                r = 2
        elif (number_of_repeats==3) and (n_fold_CV==10):
            if i < 11:
                r = 1
            elif (i > 10) and (i < 21):
                r = 2
            else:
                r = 3
        else:
            print('Given repeated cross validation is not allowed! Please check if its 5 repeats for 5-fold CV or 3 repeats for 10-fold CV or 2 repeats for 10-fold CV')
            break;

        print(f'\n------------------------- Repeat {r} | Fold {i}-------------------------------\n')
        #print('\n______________ fold : %s _______________' %i)
        
        random.seed(i)
        np.random.seed(i)
        torch.manual_seed(i)
        cudnn.deterministic = True
        cudnn.benchmark = False
        cudnn.enabled = False
        
        if (number_of_repeats==5) and (n_fold_CV==5):
            if r==1:
                folder_nm = 'Repeat_' + str(r) + '_CV_'+ str(i)
            elif r==2:
                folder_nm = 'Repeat_' + str(r) + '_CV_'+ str(i-5)
            elif r==3:
                folder_nm = 'Repeat_' + str(r) + '_CV_'+ str(i-10)
            elif r==4:
                folder_nm = 'Repeat_' + str(r) + '_CV_'+ str(i-15)
            else:
                folder_nm = 'Repeat_' + str(r) + '_CV_'+ str(i-20)
        elif (number_of_repeats==2) and (n_fold_CV==10):
            if r==1:
                folder_nm = 'Repeat_' + str(r) + '_CV_'+ str(i)
            else:
                folder_nm = 'Repeat_' + str(r) + '_CV_'+ str(i-10)
        elif (number_of_repeats==3) and (n_fold_CV==10):
            if r==1:
                folder_nm = 'Repeat_' + str(r) + '_CV_'+ str(i)
            elif r==2:
                folder_nm = 'Repeat_' + str(r) + '_CV_'+ str(i-10)
            else:
                folder_nm = 'Repeat_' + str(r) + '_CV_'+ str(i-20)
        else:
            print('Given repeated cross validation is not allowed! Please check if its 5 repeats for 5-fold CV or 3 repeats for 10-fold CV')
            break;
        
        
        CV_dir_subfolder_0 = os.path.join(path_dir, folder_nm)     
        if not os.path.exists(CV_dir_subfolder_0):
            os.makedirs(CV_dir_subfolder_0)
        elif os.path.exists(CV_dir_subfolder_0):
            shutil.rmtree(CV_dir_subfolder_0)
            os.makedirs(CV_dir_subfolder_0)
        os.chdir(CV_dir_subfolder_0)

        # split pre-train to Train & Val set 
        X_TRAIN, Y_CLASS_TRAIN, Y_VALUE_TRAIN = X_pre_train.iloc[train_indices], Y_class_pre_train.iloc[train_indices],Y_value_pre_train.iloc[train_indices]
        X_VAL, Y_CLASS_VAL, Y_VALUE_VAL = X_pre_train.iloc[val_indices], Y_class_pre_train.iloc[val_indices],Y_value_pre_train.iloc[val_indices]
        print(f'Training set shapes :: X:{X_TRAIN.shape} Y-class:{Y_CLASS_TRAIN.shape} Y-value:{Y_VALUE_TRAIN.shape} \nValidation set shapes :: X:{X_VAL.shape} Y-class:{Y_CLASS_VAL.shape} Y-value:{Y_VALUE_VAL.shape}')
        
        # save file 
        pd.DataFrame(X_TRAIN.index).to_csv('X_TRAIN.csv', header=['SMILES'], index=False, sep=',')
        Y_CLASS_TRAIN.to_csv('Y_CLASS_TRAIN.csv', header=True, index=False, sep=',')
        Y_VALUE_TRAIN.to_csv('Y_VALUE_TRAIN.csv', header=True, index=False, sep=',')
        pd.DataFrame(X_VAL.index).to_csv('X_VAL.csv', header=['SMILES'], index=False, sep=',')
        Y_CLASS_VAL.to_csv('Y_CLASS_VAL.csv', header=True, index=False, sep=',')
        Y_VALUE_VAL.to_csv('Y_VALUE_VAL.csv', header=True, index=False, sep=',')
        
        # remove constant variance descriptors
        print(f'\nX_TRAIN shape before removing low variance descriptors : {X_TRAIN.shape}')
        var_thr = VarianceThreshold(threshold = 0.0)
        var_thr.fit(X_TRAIN.values)
        columns_to_drop = [column for column in X_TRAIN.columns if column not in X_TRAIN.columns[var_thr.get_support()]]
        print(f'Low_variance_columns : {len(columns_to_drop)} \n{columns_to_drop}')
        X_TRAIN = X_TRAIN.drop(columns_to_drop,axis=1)
        print(f'X_TRAIN shape after removing low variance descriptors : {X_TRAIN.shape}')
        # remove highly correlated descr
        corr_threshold = 0.90
        X_TRAIN = remove_highly_corr_descr(X_TRAIN,corr_threshold)
        print(f'\nX_TRAIN shape after removing highly correlated columns : {X_TRAIN.shape}')

        # slice columns that present in X_TRAIN
        X_VAL = X_VAL[X_TRAIN.columns.tolist()]
        X_test = X_TEST[X_TRAIN.columns.tolist()] # remembered we have main test set as "X_TEST"

        # standardize
        #scaler = StandardScaler().fit(X_TRAIN.values)
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_TRAIN.values)
        X_TRAIN = pd.DataFrame(scaler.transform(X_TRAIN.values), columns=X_TRAIN.columns, index=None)
        X_VAL = pd.DataFrame(scaler.transform(X_VAL.values), columns=X_VAL.columns, index=None)
        X_test = pd.DataFrame(scaler.transform(X_test.values), columns=X_test.columns, index=None)
        # save scaler 
        CV_dir_subfolder_0_B = os.path.join(CV_dir_subfolder_0, 'scaler')
        os.makedirs(CV_dir_subfolder_0_B)
        os.chdir(CV_dir_subfolder_0_B)
        joblib.dump(scaler,'scaler.joblib')
        pd.DataFrame(X_TRAIN.columns).to_csv('selected_descriptor_names.csv', header=['Descriptors'], index=False, sep=',')
        X_mean_df = pd.DataFrame(X_TRAIN.mean().to_dict(),index=[X_TRAIN.index.values[-1]]).reset_index(drop=True)
        #print(f'\nX_mean_df : \n{X_mean_df}\n')
        X_mean_df.to_csv('X_TRAIN_mean.csv', header=X_TRAIN.columns.tolist(), index=False, sep=',')
        
        #print(f'X_TRAIN :: min = {X_TRAIN.to_numpy().min()} | max = {X_TRAIN.to_numpy().max()}')
        #print(f'X_VAL :: min = {X_VAL.to_numpy().min()} | max = {X_VAL.to_numpy().max()}')
        #print(f'X_test :: min = {X_test.to_numpy().min()} | max = {X_test.to_numpy().max()}\n')
        
        # create balanced dataset
        # Derive weights and construct a sampler for TRAINING SET
        targets = torch.tensor(Y_CLASS_TRAIN.values)
        class_sample_count = torch.tensor([(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in targets])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        #TRAIN_dataset = batch_dataset(X_TRAIN, Y_CLASS_TRAIN, Y_VALUE_TRAIN) 
        TRAIN_dataset = TensorDataset(torch.Tensor(X_TRAIN.values).float(), torch.Tensor(Y_CLASS_TRAIN.values).float(), torch.Tensor(Y_VALUE_TRAIN.values).float())
        VAL_dataset = TensorDataset(torch.Tensor(X_VAL.values).float(), torch.Tensor(Y_CLASS_VAL.values).float(), torch.Tensor(Y_VALUE_VAL.values).float())
        print(f'\nTrain set size: {len(TRAIN_dataset)}')

        # make data-loader
        BATCH_SIZE = 16
        train_loader = DataLoader(TRAIN_dataset, batch_size = BATCH_SIZE, sampler=sampler, pin_memory=True, drop_last=True, num_workers=args.num_workers)
        val_loader = DataLoader(VAL_dataset, batch_size = BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)
        
        # Training model
        print('\n Model training has started ...\n\n')
        CV_dir_subfolder_2 = os.path.join(CV_dir_subfolder_0, 'Final_model')
        if not os.path.exists(CV_dir_subfolder_2):
            os.makedirs(CV_dir_subfolder_2)
        elif os.path.exists(CV_dir_subfolder_2):
            shutil.rmtree(CV_dir_subfolder_2)
            os.makedirs(CV_dir_subfolder_2)
        os.chdir(CV_dir_subfolder_2)

        # Model definition:
        n_descriptors = int(X_TRAIN.shape[1])
        dropout_rate = 0.0
        my_model = Net(n_descriptors, 64, 1, dropout_rate).to(device) # nInput, nHidden=64, nOutput=1, dropout=0.0

        print(f'\noptimized_model configuration :\n{my_model}')

        # model functions
        n_EPO = 500
        loss_func_regress = nn.MSELoss()
        loss_func_classify = nn.BCELoss()
        #optimizer = optim.Adam(my_model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.000001, amsgrad=False)
        optimizer = optim.AdamW(my_model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.000001, amsgrad=False)
        #optimizer = optim.SGD(my_model.parameters(), lr=0.01, weight_decay=0.0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
        loss_adj_factor = 5.0
        # train
        #train_ROC, train_R2, val_ROC, val_R2, test_ROC, test_R2 = train_model(n_EPO, my_model, train_loader, val_loader, test_loader, loss_func_regress, loss_func_classify, optimizer, scheduler, device, CV_dir_subfolder_2, loss_adj_factor)
        #if i < 2:
        train_ROC, train_R2, val_ROC, val_R2, test_ROC, test_R2 = train_model(n_EPO, my_model, train_loader, val_loader, X_test, Y_class_test, Y_value_test, loss_func_regress, loss_func_classify, optimizer, scheduler, device, CV_dir_subfolder_2, i, loss_adj_factor)
        #else:
        #    break;
        # Save output for each fold's final model 
        final_output.loc[i,['CV_fold']] = i
        final_output.loc[i,['train_ROC']] = train_ROC
        final_output.loc[i,['train_R2']] = train_R2
        final_output.loc[i,['val_ROC']] = val_ROC
        final_output.loc[i,['val_R2']] = val_R2
        final_output.loc[i,['test_ROC']] = test_ROC
        final_output.loc[i,['test_R2']]= test_R2
        os.chdir(path_dir)
        final_output.to_csv('CV_results.csv',header=True, sep=',')
        
        #if i > 1:
        #    break;
    print('\nJob done!')
	
if __name__ == '__main__':
    main()
	