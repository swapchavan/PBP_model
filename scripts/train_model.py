import pandas as pd
import numpy as np
import os
import math
import random
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score, roc_auc_score, roc_curve, r2_score, explained_variance_score

from predict_external_query import predict_query

def find_threshold(true_class, scores):
    """Find the threshold which optimizes the G-mean of specificity vs. sensitivity using Youden's J statistic"""
    fpr, tpr, thresholds = roc_curve(true_class, scores)
    # get the best threshold
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    return best_thresh

def train_model(n_epochs, optimized_model, train_loader, val_loader, X_test, Y_class_test, Y_value_test, loss_function_regress, loss_function_classify, opt, scheduler, device, path_dir, seed, loss_adj_factor=1.0):

    col_names =  ['Epoch','Train_ROC','Train_BA','Train_R2','Val_ROC','Val_BA','Val_R2','Test_ROC','Test_BA','Test_R2']
    my_output = []
    my_output = pd.DataFrame(index=range(0, n_epochs), columns = col_names)

    #min_val_ROC = float(0.0)
    #min_R2 = float(-3000.58950865093647e+20)
    #min_loss = float(3000.58950865093647e+20)
    #min_val_score = float(-3000.58950865093647e+20)
    min_val_R2 = float(-3000.58950865093647e+20)

    epochs_no_improve = 0
    final_total_loss = 0.0
    final_classify_loss = 0.0
    final_regress_loss = 0.0
    for epoch in range(n_epochs):
        my_output.loc[epoch,['Epoch']] = epoch
        #print(f"\nEpoch {epoch} (LR : {opt.param_groups[0]['lr']}) ____________")
        # train
        optimized_model.train()
        for i, (batch_x, batch_y_class, batch_y_values) in enumerate(train_loader):
            batch_x, batch_y_class, batch_y_values = batch_x.to(device, dtype=torch.float32), batch_y_class.to(device, dtype=torch.float32), batch_y_values.to(device, dtype=torch.float32)
            #print(f'\n\nSHAPE => batch_x : {batch_x.shape} | batch_y_class : {batch_y_class.shape} | batch_y_values : {batch_y_values.shape}\n\n')
            #print(f'\nx : \n{batch_x} \n\nbatch_y_class : \n{batch_y_class} \n\nbatch_y_values : {batch_y_values}\n')
            # train model 
            opt.zero_grad()
            output = optimized_model.forward(batch_x)
            # For classification
            output_for_classify = optimized_model.pred_proba(output)
            #print(f'output_1 : {output_1}')
            classify_loss_orig = loss_function_classify(output_for_classify.squeeze(1), batch_y_class)
            # For regression 
            #print(f'output shape {output.shape}')
            #output_for_regre = output.squeeze(0)
            output_for_regre = output.squeeze(1)
            
            #print(f'\noutput : \n{output}')
            #print(f'\nbatch_y_class : \n{batch_y_class}')
            #print(f'\noutput_for_classify : \n{output_for_classify}')
            #print(f'\nbatch_y_values : \n{batch_y_values}')
            #print(f'\noutput_for_regre : \n{output_for_regre}')
            
            #print(f'output shape after removing all dimensions : {output_for_regre.shape}')
            # lets drop output where logBB values were missing in original Y
            output_for_regre = output_for_regre[~batch_y_values.isnan()]
            #print(f'output shape after removing nan locations : {output_for_regre.shape}')
            batch_y_values = batch_y_values[~batch_y_values.isnan()]
            #batch_y_values = batch_y_values[~batch_y_values.isnan()]
            
            # make sure after removing elements that are nan, final tensor must not be empty 
            if output_for_regre.numel() != 0:
                regress_loss_orig = loss_function_regress(output_for_regre, batch_y_values)
                total_loss_orig = classify_loss_orig + regress_loss_orig*loss_adj_factor
                #print(f'No empty loss :: Total Loss : {total_loss_orig} | Classify : {classify_loss_orig} | Regression : {regress_loss_orig*20}')
            else:
                # if actual logBB value tensor is empty then --> regress_loss_orig = 0 & total loss == classification loss 
                regress_loss_orig = torch.zeros(1).item()
                total_loss_orig = classify_loss_orig 
                #print(f'While backward Total Loss : {total_loss_orig} | Classify : {classify_loss_orig}')
                #print(f'When empty loss :: Total Loss : {total_loss_orig} | Classify : {classify_loss_orig} | Regression : {regress_loss_orig}')
            #print(f'batch_y_values : {batch_y_values} output_for_regre {output_for_regre} | regress_loss {regress_loss}')
            #print(f'output_1 {output_1.shape} | batch_y_class {batch_y_class.shape}')
            #print(f'batch_y_class : {batch_y_class}') #print(f'output_1 {output_1.shape} | batch_y_class {batch_y_class.shape}')
            # adjust losses to get total loss 
            #factor_to_multiply = regress_loss/classify_loss
            
            #total_loss = classify_loss + regress_loss
            total_loss_orig.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(optimized_model.parameters(), 1)
            opt.step()
            #final_total_loss += total_loss
            #final_classify_loss += classify_loss*0.05
            #final_regress_loss += regress_loss
            
        # eval
        
        # define losses 
        train_loss = 0.0
        val_loss = 0.0
        regress_train_loss = 0.0
        classify_train_loss = 0.0
        regress_val_loss = 0.0
        classify_val_loss = 0.0
        
        # define empty array to save results
        tr_actual_class = np.empty((0), int)
        tr_actual_value = np.empty((0), int)
        tr_pred_class_prob = np.empty((0), int)
        tr_pred_value_prob = np.empty((0), int)
        vl_actual_class = np.empty((0), int)
        vl_actual_value = np.empty((0), int)
        vl_pred_class_prob = np.empty((0), int)
        vl_pred_value_prob = np.empty((0), int)
        
        optimized_model.eval()
        with torch.no_grad():
            # Train            
            for i, (batch_x, batch_y_class, batch_y_values) in enumerate(train_loader):
                #print(f'SHAPE => batch_x : {batch_x.shape} | batch_y_class : {batch_y_class.shape} | batch_y_values : {batch_y_values.shape}')
                batch_x, batch_y_class, batch_y_values = batch_x.to(device, dtype=torch.float32), batch_y_class.to(device, dtype=torch.float32), batch_y_values.to(device, dtype=torch.float32)
                output = optimized_model.forward(batch_x)
                # For classification 
                output_for_classify = optimized_model.pred_proba(output)
                classify_loss = loss_function_classify(output_for_classify.squeeze(1), batch_y_class)
                # For regression 
                output_for_regre = output.squeeze(1)
                output_for_regre = output_for_regre[~batch_y_values.isnan()]
                batch_y_values = batch_y_values[~batch_y_values.isnan()]
                #print(f'\noutput_for_regre - Train : {output_for_regre}')
                #print(f'batch_y_values - Train : {batch_y_values}\n')
                #if len(output_for_regre.size()) != 0:
                if output_for_regre.numel() != 0:
                    regress_loss = loss_function_regress(output_for_regre, batch_y_values)
                    train_epoch_loss = classify_loss + regress_loss*loss_adj_factor
                else:
                    regress_loss_orig = torch.tensor([0])
                    train_epoch_loss = classify_loss
                
                #print(f'regress_loss : {regress_loss}') 
                #print(f'Train total loss : {train_epoch_loss}')
                train_loss += train_epoch_loss
                regress_train_loss += regress_loss
                classify_train_loss += classify_loss
                
                # save output 
                tr_actual_class = np.append(tr_actual_class, batch_y_class.cpu().data.numpy(), axis=0)
                tr_actual_value = np.append(tr_actual_value, batch_y_values.cpu().data.numpy(), axis=0)
                tr_pred_value_prob = np.append(tr_pred_value_prob, output_for_regre.cpu().data.numpy(), axis=0)
                tr_pred_class_prob = np.append(tr_pred_class_prob, output_for_classify.squeeze(1).cpu().data.numpy(), axis=0)
                
            # Val
            for i, (batch_x, batch_y_class, batch_y_values) in enumerate(val_loader):
                #print(f'SHAPE => batch_x : {batch_x.shape} | batch_y_class : {batch_y_class.shape} | batch_y_values : {batch_y_values.shape}')
                batch_x, batch_y_class, batch_y_values = batch_x.to(device, dtype=torch.float32), batch_y_class.to(device, dtype=torch.float32), batch_y_values.to(device, dtype=torch.float32)
                output = optimized_model.forward(batch_x)
                # For classification 
                output_for_classify = optimized_model.pred_proba(output)
                classify_loss = loss_function_classify(output_for_classify.squeeze(1), batch_y_class)
                # For regression 
                output_for_regre = output.squeeze(1)
                output_for_regre = output_for_regre[~batch_y_values.isnan()]
                batch_y_values = batch_y_values[~batch_y_values.isnan()]
                #if len(output_for_regre.size()) != 0:
                if output_for_regre.numel() != 0:
                    regress_loss = loss_function_regress(output_for_regre, batch_y_values)
                    val_epoch_loss = classify_loss + regress_loss*loss_adj_factor
                else:
                    regress_loss_orig = torch.tensor([0])
                    val_epoch_loss = classify_loss
                
                val_loss += val_epoch_loss
                regress_val_loss += regress_loss
                classify_val_loss += classify_loss
                # save output 
                vl_actual_class = np.append(vl_actual_class, batch_y_class.cpu().data.numpy(), axis=0)
                vl_actual_value = np.append(vl_actual_value, batch_y_values.cpu().data.numpy(), axis=0)
                vl_pred_value_prob = np.append(vl_pred_value_prob, output_for_regre.cpu().data.numpy(), axis=0)
                vl_pred_class_prob = np.append(vl_pred_class_prob, output_for_classify.squeeze(1).cpu().data.numpy(), axis=0)
        
        train_R2 = float(format(r2_score(tr_actual_value, tr_pred_value_prob)))
        val_R2 = float(format(r2_score(vl_actual_value, vl_pred_value_prob)))
        try:
            train_roc = float(format(roc_auc_score(tr_actual_class, tr_pred_class_prob)))
        except:
            train_roc = np.nan
        try:
            val_roc = float(format(roc_auc_score(vl_actual_class, vl_pred_class_prob)))
        except:
            val_roc = np.nan
        
        # find best threshold for class-descretization
        best_thre = find_threshold(vl_actual_class, vl_pred_class_prob)
        # descretized using best threshold 
        tr_pred_class = np.where(tr_pred_class_prob >= best_thre, 1, 0)
        vl_pred_class = np.where(vl_pred_class_prob >= best_thre, 1, 0)
        
        # calc BA 
        train_BA = float(format(balanced_accuracy_score(tr_actual_class, tr_pred_class)))
        val_BA = float(format(balanced_accuracy_score(vl_actual_class, vl_pred_class)))
        
        my_output.loc[epoch,['Train_ROC']] = train_roc
        my_output.loc[epoch,['Train_BA']] = train_BA
        my_output.loc[epoch,['Train_R2']] = train_R2
        my_output.loc[epoch,['Val_ROC']] = val_roc
        my_output.loc[epoch,['Val_BA']] = val_BA
        my_output.loc[epoch,['Val_R2']] = val_R2

        #val_score = val_BA+val_R2
        #scheduler.step(val_score)
        scheduler.step(val_R2)
        
        #print(f'\nEpoch:{epoch} train_roc : {train_roc:.4f} | val_roc : {val_roc:.4f} | test_roc : {test_roc:.4f} || train_R2 : {train_R2:.4f} | val_R2 : {val_R2:.4f} | test_R2 : {test_R2:.4f}')
        #print(f'train loss : {train_loss:.4f} | Val loss : {val_loss:.4f} || regress total loss : {regress_train_val_loss:.4f} | classify total loss : {classify_train_val_loss:.4f}')
        #print(f'final_total_loss : {final_total_loss:.4f} | final_classify_loss : {final_classify_loss:.4f} | final_regress_loss : {final_regress_loss:.4f} ')
        #print(f'\nEpoch:{epoch} train_BA : {train_BA:.4f} | val_BA : {val_BA:.4f}  || train_R2 : {train_R2:.4f} | val_R2 : {val_R2:.4f} ')
        #print(f'LOSS :: TRAIN:{epoch} total : {train_loss:.8f} | regre : {regress_train_loss:.8f} | classify : {classify_train_loss:.8f} ')
        #print(f'LOSS :: VAL:{epoch} total : {val_loss:.8f} | regre : {regress_val_loss:.8f} | classify : {classify_val_loss:.8f} ')
        
        # Stopping crterion
        if val_R2 > min_val_R2:
        #if train_R2 > min_R2:
        #if regress_train_val_loss < min_loss:
        #if train_loss < min_loss:
        #if val_score > min_val_score:
            # update variables
            epochs_no_improve = 0
            #print(f'R2 improved {min_R2} ---> {train_R2}')
            #min_R2 = train_R2
            #print(f'Train loss has decreased ({min_loss:.8f} ---> {train_loss:.8f})')
            #print(f'val_score has decreased {min_val_score} ---> {val_score}')
            #min_val_score = val_score
            #min_loss = train_loss
            print(f'val_R2 has increased {min_val_R2} ---> {val_R2}')
            min_val_R2 = val_R2
            # save R2 & ROC at given checkpoint
            best_model_train_roc = train_roc
            best_model_val_roc = val_roc
            best_model_train_R2 = train_R2
            best_model_val_R2 = val_R2
            
            # Lets test best model on test set 
            Query_classes_proba, Query_classes, Query_values = predict_query(optimized_model, X_test, best_thre, seed, device)
            # remove missing logBB value rows
            Y_value_test_with_header = pd.DataFrame(Y_value_test.reset_index(drop=True))
            #Y_value_test_with_header.columns[0] = 'VALUES'
            Y_value_test_with_header.rename(columns={0 :'VALUES'}, inplace=True)
            
            non_nan_value_loc = Y_value_test_with_header.index[Y_value_test_with_header['VALUES'].notna()]
            Y_value_test_actual = Y_value_test_with_header.iloc[non_nan_value_loc,:]
            Query_values_pred = Query_values[non_nan_value_loc]
            #print(f'Y_value_test_actual : {Y_value_test_actual}')
            #print(f'Query_values_pred : {Query_values_pred}')
            best_model_test_roc = float(format(roc_auc_score(Y_class_test, Query_classes_proba)))
            best_model_test_BA = float(format(roc_auc_score(Y_class_test, Query_classes)))
            best_model_test_R2 = float(format(r2_score(Y_value_test_actual, Query_values_pred)))
            print(f'\nBEST model at epoch {epoch}')
            print(f'best_model_train_roc : {best_model_train_roc:.4f} | best_model_val_roc : {best_model_val_roc:.4f} | best_model_test_roc : {best_model_test_roc:.4f} || best_model_train_R2 : {best_model_train_R2:.4f} | best_model_val_R2 : {best_model_val_R2:.4f} | best_model_test_R2 : {best_model_test_R2:.4f} || best_model_test_BA : {best_model_test_BA:.4f}')

            # save model's state at given check point
            os.chdir(path_dir)
            torch.save(optimized_model.state_dict(), 'checkpoint.pt')
            # save output            
            # train 
            np.savetxt("tr_actual_class.csv", tr_actual_class, delimiter=",", fmt='%i')
            np.savetxt("tr_actual_value.csv", tr_actual_value, delimiter=",", fmt='%f')
            np.savetxt("tr_pred_class_prob.csv", tr_pred_class_prob, delimiter=",", fmt='%f')
            np.savetxt("tr_pred_class_tuned.csv", tr_pred_class, delimiter=",", fmt='%i')
            np.savetxt("tr_pred_value.csv", tr_pred_value_prob, delimiter=",", fmt='%f')
            # val
            np.savetxt("vl_actual_class.csv", vl_actual_class, delimiter=",", fmt='%i')
            np.savetxt("vl_actual_value.csv", vl_actual_value, delimiter=",", fmt='%f')
            np.savetxt("vl_pred_class_prob.csv", vl_pred_class_prob, delimiter=",", fmt='%f')
            np.savetxt("vl_pred_class_tuned.csv", vl_pred_class, delimiter=",", fmt='%i')
            np.savetxt("vl_pred_value.csv", vl_pred_value_prob, delimiter=",", fmt='%f')
            #test
            np.savetxt("te_actual_class.csv", Y_class_test, delimiter=",", fmt='%i')
            np.savetxt("te_actual_value.csv", Y_value_test_actual, delimiter=",", fmt='%f')
            np.savetxt("te_pred_class.csv", Query_classes, delimiter=",", fmt='%f')
            np.savetxt("te_pred_value.csv", Query_values_pred, delimiter=",", fmt='%f')
            
            # save best threshold
            #np.savetxt("best_threshold.csv", best_thre, delimiter=",", fmt='%f')
            file_path = 'best_threshold.pickle'
            # Open the file in binary mode
            with open(file_path, 'wb') as file:
                pickle.dump(best_thre, file)
            
        else:
            epochs_no_improve += 1
            print(f'EarlyStopping counter: {epochs_no_improve} out of {20}')

        if epochs_no_improve == 20:
            early_stop = True
            #print(f"Early stopping at Epoch {e}")
            break;
        else:
            continue;
    # save model
    os.chdir(path_dir)
    my_output.dropna(axis = 0, how = 'all').to_csv('BBB_multitask_model_stats.csv',header=True, sep=',')
    #print(f'\ntrain roc : {best_model_train_roc:.4f} || val ROC : {best_model_val_roc:.4f}  || test ROC : {best_model_test_roc:.4f}')
    #print(f'\nROC :: train {best_model_train_roc:.4f} | val {best_model_val_roc:.4f} |  test {best_model_test_roc} || R2 :: train {best_model_train_R2:.4f} | val {best_model_val_R2:.4f} | test {best_model_test_R2}')
    # save final model
    optimized_model.load_state_dict(torch.load('checkpoint.pt'))
    # Specify a path
    PATH = "entire_model.pt"
    # Save
    torch.save(optimized_model, PATH)
    return best_model_train_roc, best_model_train_R2, best_model_val_roc, best_model_val_R2, best_model_test_roc, best_model_test_R2