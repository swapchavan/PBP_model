import pandas as pd
import numpy as np
import os
import joblib
import pickle
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from Net import Net

def predict_single_query(my_model, test_df, thre):
    # testing
    te_pred_class = np.empty((0), int)
    te_pred_value = np.empty((0), int)

    my_model.eval()
    # turn off gradients for validation
    with torch.no_grad():
        # convert input to torch tensor
        batch_x = torch.tensor(test_df.values).to(dtype=torch.float32)
        # predictions
        output_value = my_model.forward(batch_x)
        output_class_proba = my_model.pred_proba(output_value)
        # descretized using best threshold 
        output_class = np.where(output_class_proba >= thre, 1, 0)
        output_class = np.squeeze(output_class, axis=1)
        # save output
        te_pred_class = np.append(te_pred_class, output_class, axis=0)
        te_pred_value = np.append(te_pred_value, output_value.squeeze(1).cpu().data.numpy(), axis=0)
    return te_pred_class, te_pred_value
    
def predict_query_by_all_models(X_smiles, X_test, n_repeats, n_CV, model_folder_path):
    # declare empty result table
    Result_pred_class = pd.DataFrame(index=range(len(X_test)))
    #Result_pred_BB_ratio_0_1_scale = pd.DataFrame( index=range(len(X_test)))
    Result_pred_values = pd.DataFrame( index=range(len(X_test)))
    # Lets run query for each fold in each repeated experiements
    fold = 0
    for repeat in range(n_repeats):
        #print('_______________')
        if fold == 0:
            seed = 0
        elif fold > 10:
            seed = 9
        # lets run 2-repeat-10 fold CV evaluation for the given X_test 
        for fold in range(n_CV):
            seed += 1
            print(f'\n------------ repeat {repeat+1} | fold {fold+1} | seed {seed} -----------------')
            # set seed 
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            cudnn.deterministic = True
            cudnn.benchmark = False
            cudnn.enabled = False

            # run predictions 
            temp_model_nm = 'Repeat_' + str(repeat+1) + '_CV_'+ str(fold+1)
            # Lets import model & other important files
            # 1. model 
            folder_nm = 'Repeat_' + str(repeat+1) + '_CV_'+ str(fold+1) + '/Final_model/entire_model.pt' # created path of model within each CV fold of each repeat  
            model_dir = os.path.join(model_folder_path, folder_nm)
            model = torch.load(model_dir, map_location='cpu')
            # 2. Descr scaler
            X_scaler_path = 'Repeat_' + str(repeat+1) + '_CV_'+ str(fold+1) + '/scaler/scaler.joblib'
            X_scaler_path_full = os.path.join(model_folder_path, X_scaler_path)
            X_scaler = joblib.load(X_scaler_path_full)
            # 3. Descr names
            X_names = 'Repeat_' + str(repeat+1) + '_CV_'+ str(fold+1) + '/scaler/selected_descriptor_names.csv'
            X_names_full = os.path.join(model_folder_path, X_names)
            descr_names = pd.read_csv(X_names_full,header=0,index_col = None)
            descr_names.columns = ['descriptors'] # renamed column 
            # 4. X Train mean df
            X_mean = 'Repeat_' + str(repeat+1) + '_CV_'+ str(fold+1) + '/scaler/X_TRAIN_mean.csv'
            X_mean_full = os.path.join(model_folder_path, X_mean)
            descr_mean = pd.read_csv(X_mean_full,header=0,index_col = None)
            # 5. probability threshold
            best_thr_path = 'Repeat_' + str(repeat+1) + '_CV_'+ str(fold+1) + '/Final_model/best_threshold.pickle'
            best_thr_path_full = os.path.join(model_folder_path, best_thr_path)
            best_threshold = pd.read_pickle(best_thr_path_full)
            print('model & associated files loaded successfully!')            
            # descr preprocessing
            print(f'Descriptors before pre-processing : {X_test.shape} ')
            int_features = X_test[descr_names.descriptors]
            print(f'Descriptors after slicing model-descriptors : {int_features.shape}')
            # fill missing values if any nan exists else do nothing
            if int_features.isnull().values.any():
                int_features = int_features.fillna(descr_mean)
                print(f'Descriptor shape after filling missing values wrt training set : {int_features.shape}')
            # standardization
            X_TEST_std = pd.DataFrame(X_scaler.transform(int_features.values), columns=int_features.columns, index=None)
            print(f'Descriptor shape after standardization : {X_TEST_std.shape}')
            # prediction
            Query_classes, Query_values = predict_single_query(model, X_TEST_std, best_threshold)
            # Values were squeeze between 0 and 1, lets get original values
            #Query_values_2d_format = np.expand_dims(Query_values, axis=1)
            #Query_values_original_single_dim = np.squeeze(Query_values_original, axis=1)
            
            # save output to result table 
            Result_pred_class.loc[:,[temp_model_nm]] = np.expand_dims(Query_classes, axis=1) #Query_classes
            #Result_pred_BB_ratio_0_1_scale.loc[:,[temp_model_nm]] = np.expand_dims(Query_values, axis=1) #Query_values
            Result_pred_values.loc[:,[temp_model_nm]] = np.expand_dims(Query_values, axis=1)
            
            print(f'Evaluation finished for : {temp_model_nm}\n')
            #print(f'Result_pred_class : {Result_pred_class}\n')
            #print(f'Result_pred_values : {Result_pred_values}\n')

    # calculate uncertainty in classsification
    # pred class 
    Result_pred_class_uncerainty = pd.DataFrame(Result_pred_class.sum(axis=1), columns=['Total_models_pred_postive'])
    Result_pred_class_uncerainty.loc[:,['Predicted_class']] = Result_pred_class_uncerainty['Total_models_pred_postive'].apply(lambda x: 1 if x>10 else 0)
    Result_pred_class_uncerainty.loc[:,['Confidence (%)']] = Result_pred_class_uncerainty['Total_models_pred_postive'].apply(lambda x: (x/20)*100 if x>10 else (1-(x/20))*100).round(0)
    Result_pred_class_uncerainty.loc[:,['Remark']] = Result_pred_class_uncerainty['Confidence (%)'].apply(lambda x: 'Uncertain' if x==50 else ('Low confidence' if x<75 else 'High confidence'))
    Result_pred_class_uncerainty.drop(columns=['Total_models_pred_postive'], inplace=True)
    Result_pred_class_uncerainty = Result_pred_class_uncerainty.rename(index=X_smiles)

    # non-log BB 
    Result_pred_values_uncerainty = pd.DataFrame(Result_pred_values.mean(axis=1), columns=['Predicted_values']).round(4)
    Result_pred_values_uncerainty['Uncertainty'] =  Result_pred_values.std(axis=1).round(4)
    Result_pred_values_uncerainty = Result_pred_values_uncerainty.rename(index=X_smiles)

    return Result_pred_class_uncerainty, Result_pred_values_uncerainty