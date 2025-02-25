#!/usr/bin/env python

# conda execute
# env:
#  - python <=3
#  - numpy


#################################################################################################################################
#
# Author: Swapnil Chavan
# Subject: "2D numerical descriptors calculation using RdKit."
# Date: 2019-03-07

#################################################################################################################################


# Load important libraries

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd

#from rdkit import RDLogger
#RDLogger.DisableLog('rdApp.*')
#from rdkit import RDLogger
#from rdkit.rdBase import DisableLog
#for level in RDLogger._levels:
#    DisableLog(level)

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

import warnings 
warnings.filterwarnings("ignore")

def calculate_descriptors_for_single_smiles(x_i):
    # declare variable
    FP_all = []
    m_i = Chem.MolFromSmiles(x_i)
    nms = [x[0] for x in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
    FP_all_as_list = list(calc.CalcDescriptors(m_i))
    FP_all.append(FP_all_as_list)
    output_final = pd.DataFrame(FP_all)
    output_final.columns = nms
    return output_final
    
def calculate_descriptors_for_multiple_smiles(input_smi):
    # count total chemicals
    n_all = input_smi.shape[0]
    # declare variable
    FP_all = []
    # calc for each smiles
    for i in range(0,n_all):
        x_i = input_smi.iloc[i,0]
        m_i = Chem.MolFromSmiles(x_i)
        nms = [x[0] for x in Descriptors._descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
        FP_all_as_list = list(calc.CalcDescriptors(m_i))
        FP_all.append(FP_all_as_list)
    output_final = pd.DataFrame(FP_all)
    output_final.columns = nms
    return output_final