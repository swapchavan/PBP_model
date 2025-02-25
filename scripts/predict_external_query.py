import numpy as np
import random 
import torch
from torch.backends import cudnn

def predict_query(my_model, test_df, thre, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    cudnn.enabled = False
    # Lets test best model on test set 
    # 1. declare empty array to staore outcome
    te_pred_proba = np.empty((0), int)
    te_pred_class = np.empty((0), int)
    te_pred_value = np.empty((0), int)
    # 2. perform evaluations
    my_model.eval()
    with torch.no_grad():
        batch_x = torch.tensor(test_df.values).to(device, dtype=torch.float32)
        # predictions
        output_value = my_model.forward(batch_x)
        output_class_proba = my_model.pred_proba(output_value)
        output_class_proba = output_class_proba.cpu()
        #print(f'\noutput_class_proba shape : {output_class_proba.shape} ')
        # descretized using best threshold 
        output_class = np.where(output_class_proba >= thre, 1, 0)
        output_class = np.squeeze(output_class, axis=1)
        # save output
        te_pred_proba = np.append(te_pred_proba, output_class_proba.squeeze(1).cpu(), axis=0)
        te_pred_class = np.append(te_pred_class, output_class, axis=0)
        te_pred_value = np.append(te_pred_value, output_value.squeeze(1).cpu(), axis=0)
    return te_pred_proba, te_pred_class, te_pred_value