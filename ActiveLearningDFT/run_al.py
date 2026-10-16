#%%
# IMPORT DEPENDENCIES--------------------------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu January 26 15:48:51 2023
Edited on Tues May 02 12:13:40 2023
@author: kangming
@edit: daniel
@edit: rafael 
"""

import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import random

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import  GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import math
import time
import pathlib

from distill import get_data, return_model

from uncertaintyAL import grow_random
from uncertaintyAL import grow_QBC



from uncertaintyAL import  grow2QBC

from uncertaintyAL import grow_EA_QBC_2Obj_twoTargets_from3Outcomes

from uncertaintyAL import grow_EA_QBC_2Obj_twoTargets_plusDiversity_from3Outcomes

from uncertaintyAL import grow_EA_QBC_1Obj_withDiversity_from3Outcomes

from uncertaintyAL import grow_disagreementHV_topM_kmedoids_OneTarget
from uncertaintyAL import grow_TS_QBC_no_diversity_OneTarget
from uncertaintyAL import grow_DPP_TS_QBC_OneTarget 

from uncertaintyAL import grow_DPP_TS_QBC_TwoTargets


from uncertaintyAL import grow_disagreementHV_topM_kmedoids_TwoTargets

from uncertaintyAL import grow_TS_QBC_no_diversity_TwoTargets

from uncertaintyAL import grow_disagreementHV_no_diversity_TwoTargets

print('Dependencies imported successfully!')

#%%
# REQUIRE USER INPUT---------------------------------------------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser(
    description='Grow datasets with different growing methods.'
    )

parser.add_argument('--growingCriteria', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--target', type=str, required=True)
parser.add_argument('--outputDir', type=str, required=True)
parser.add_argument('--stepFrac', type=float, required=False, default=0.01)
parser.add_argument('--trainFracStop', type=float, required=False, default=1.0)
parser.add_argument('--randomSeed', type=int, required=False, default=0)


growingCriteria = parser.parse_args().growingCriteria
dataset = parser.parse_args().dataset
target = parser.parse_args().target
outputDir = parser.parse_args().outputDir
fltStepFrac = parser.parse_args().stepFrac
fltTrainFracStop = parser.parse_args().trainFracStop
intRandomSeed = parser.parse_args().randomSeed


# %%
# IMPORT MODELS--------------------------------------------------------------------------------------------------------
# set random state
random_state = 0

# make a dictionary to hold the models
model = {}

# loop through the models and add them to the dictionary
for modelname in ['xgb','rf']:
    model[modelname] = return_model(modelname, random_state)
    # print(modelname, ' imported successfully: ', model[modelname])

print('Models imported successfully!')
print("GrowingCriteria is",growingCriteria)

print(intRandomSeed)

# Training two outcomes training cases
if growingCriteria in (
    'grow2QBC',
    'grow_2TargetsOnlyQBC',
    'grow_DPP_TS_QBC_TwoTargets',
    'grow_disagreementHV_topM_kmedoids_TwoTargets',
    'grow_TS_QBC_no_diversity_TwoTargets',
    'grow_disagreementHV_no_diversity_TwoTargets'):
    if target=='eform_bulk_modulus':
        target1='e_form'
        target2='bulk_modulus'

    if target=='bandgap_eform':
        target1='bandgap'
        target2='e_form'
        
        
    if target=='bandgap_bulkmodulus':
        target1='bandgap'
        target2='bulk_modulus'



#%%
# GROW DATASETS WITH BASED ON THE ARGUMENTS PASSED---------------------------------------------------------------------


#---------------------------------------IMPORTANT TO REMEMBER. IN ALL THE CODE 
#                                       y1=e_form 
#                                       y2=bulk_modulus
#                                       y3=bandgap



def drop_most_correlated_features(
    X: pd.DataFrame,
    threshold: float = 0.85,
    method: str = "pearson",
):
    """
    Iteratively drops the feature with the largest number of
    pairwise correlations above the threshold.

    This only uses X, not the targets Y.
    """

    X_work = X.copy()
    dropped_features = []
    history = []

    while True:
        corr = X_work.corr(method=method).abs()

        # Ignore self-correlation
        corr = corr.mask(np.eye(len(corr), dtype=bool), 0)

        # Count how many correlations each feature has above threshold
        high_corr_counts = (corr > threshold).sum(axis=0)

        max_count = high_corr_counts.max()

        # Stop when no pairwise correlation is above threshold
        if max_count == 0:
            break

        feature_to_drop = high_corr_counts.idxmax()

        history.append({
            "dropped_feature": feature_to_drop,
            "number_of_correlations_above_threshold": int(max_count),
            "correlated_with": corr.index[
                corr[feature_to_drop] > threshold
            ].tolist(),
        })

        X_work = X_work.drop(columns=[feature_to_drop])
        dropped_features.append(feature_to_drop)

    return X_work, dropped_features, history





alldata , _, _,_,_, xTrainVal, xTest, y1TrainVal,y2TrainVal,y3TrainVal,y1Test,y2Test,y3Test = get_data(dataset = dataset, random_state = 0)

alldata['compound possible'] = alldata['compound possible'].astype(float)
alldata_colFilter=alldata.copy()
xTrainVal['compound possible'] = xTrainVal['compound possible'].astype(float)
xTest['compound possible'] = xTest['compound possible'].astype(float)


alldata_colFilter, dropped_features, _ =drop_most_correlated_features(alldata_colFilter)


xTrainVal=xTrainVal.drop(columns=dropped_features)
xTest=xTest.drop(columns=dropped_features)

try:
    alldata_colFilter.shape[1]==xTest.shape[1]
    xTest.shape[1]==xTrainVal.shape[1]

except:
    # Code executed if an error occurs
    print("Number of columns of all data, the pool and the test dataset do not coincide")


print("Number of columns kept:",xTest.shape[1] )

print("-------------------Saving all data-------------------------- ")
alldata_colFilter.to_csv("alldata_ColFilter"+dataset+".csv")

alldata.to_csv("alldata_NoColFilter"+dataset+".csv")
xTrainVal.to_csv("xTrainVal_"+dataset+".csv")

xTest.to_csv("xTest_"+dataset+".csv")
y1TrainVal.to_csv("y1TrainVal_"+dataset+".csv")
y2TrainVal.to_csv("y2TrainVal_"+dataset+".csv")
y3TrainVal.to_csv("y3TrainVal_"+dataset+".csv")
y1Test.to_csv("y1Test_"+dataset+".csv")
y2Test.to_csv("y2Test_"+dataset+".csv")
y3Test.to_csv("y3Test_"+dataset+".csv")

print("-------------------Finished Saving all data---------------------------- ")
# Concatenate columns (preserving df1's index)
#df_combined = pd.concat([xTrainVal, y1TrainVal, y2TrainVal, y3TrainVal], axis=1)

# Save to CSV (with index)
#df_combined.to_csv("X_train_Val.csv", index=True)

#print('CREATED DATA SET')
#import sys
#sys.exit()

print('\n--------------------------------------------------------------------------------------------')
print ('Dataset: ', dataset, '\nTarget: ', target)
# print the size of the training sets
print('xTrainVal Length: ', len(xTrainVal))
tStart = time.time()

# if the growing criteria is random
if growingCriteria == 'random':
    # grow by random for the dataset and target combination
    grow_random(model['rf'], model['xgb'],
                           xTrainVal, y1TrainVal,
                           y2TrainVal,y3TrainVal,
                           xTest, y1Test,
                           y2Test,y3Test,
                           strSaveDir = outputDir,
                           strSaveName = dataset + '_' + target + '_random',
                           strModel1Name = 'rf',
                           strModel2Name = 'xgb',
                           fltStepFrac = fltStepFrac,
                           intRandomSeed = intRandomSeed)

elif growingCriteria=='growing_EA_2obj_qbc_only':
    grow_EA_QBC_2Obj_twoTargets_from3Outcomes(
    model['rf'], model['xgb'],
    xTrainVal, y1TrainVal, y2TrainVal, y3TrainVal,
    xTest, y1Test, y2Test, y3Test,
    target1,
    target2,
    outcome_names=("e_form", "bulk_modulus", "bandgap"),
    strSaveDir=outputDir,
    strSaveName= dataset + '_' + target +"growing_EA_2obj_qbc_only",
    strModel1Name="rf",
    strModel2Name="xgb",
    fltStepFrac=fltStepFrac,
    fltTrainFracStop=1.0,
    intRandomSeed=intRandomSeed,
    # EA params
    pop_size=120,
    n_gen=60,
    p_cx=0.9,
    p_mut=0.25,
    # rounding for logs
    n_round=6,
    n_train_ratio=6,
    )
    
elif growingCriteria=='grow_EA_QBC_2Obj_twoTargets_plusDiversity':
    grow_EA_QBC_2Obj_twoTargets_plusDiversity_from3Outcomes(
    model['rf'], model['xgb'],
    xTrainVal, y1TrainVal, y2TrainVal, y3TrainVal,
    xTest, y1Test, y2Test, y3Test,
    target1,
    target2,
    outcome_names=("e_form", "bulk_modulus", "bandgap"),
    strSaveDir=outputDir,
    strSaveName= dataset + '_' + target +"growing_EA_2obj_qbc_Diversity",
    strModel1Name="rf",
    strModel2Name="xgb",
    fltStepFrac=fltStepFrac,
    fltTrainFracStop=1.0,
    intRandomSeed=intRandomSeed,
    # EA params
    pop_size=120,
    n_gen=60,
    p_cx=0.9,
    p_mut=0.25,
    # rounding for logs
    n_round=6,
    n_train_ratio=6,
    )



elif growingCriteria == 'QBC':
    # grow by QBC for the dataset and target combination
    grow_QBC(model['rf'], model['xgb'],
             xTrainVal, y1TrainVal,
             y2TrainVal,y3TrainVal,
             xTest, y1Test,
             y2Test,y3Test,
             target,
             strSaveDir = outputDir,
             strSaveName = dataset + '_' + target + '_QBC',
             strModel1Name = 'rf',
             strModel2Name = 'xgb',
             fltStepFrac = fltStepFrac,
             intRandomSeed = intRandomSeed)


elif  growingCriteria=='grow_EA_QBC_1Obj_withDiversity':
    grow_EA_QBC_1Obj_withDiversity_from3Outcomes(
    model['rf'], model['xgb'],
    xTrainVal, y1TrainVal, y2TrainVal, y3TrainVal,
    xTest, y1Test, y2Test, y3Test,
    target,
    strSaveDir=outputDir,
    strSaveName=dataset + '_' + target + 'grow_EA_QBC_1Obj_withDiversity_from3Outcomes',
    strModel1Name="rf",
    strModel2Name="xgb",
    # loop control
    fltStepFrac=fltStepFrac,
    fltTrainFracStop=fltTrainFracStop,
    intRandomSeed=intRandomSeed)


elif growingCriteria=="grow_disagreementHV_topM_kmedoids_OneTarget":
    grow_disagreementHV_topM_kmedoids_OneTarget(model['rf'], model['xgb'],
        xTrainVal, y1TrainVal, y2TrainVal, y3TrainVal,
        xTest, y1Test, y2Test, y3Test,
        target,
        strSaveDir=outputDir,
        strSaveName=dataset + '_' + target + 'grow_disagreementHV_topM_kmedoids_OneTarget',
        strModel1Name="rf",
        strModel2Name="xgb",
        fltStepFrac=fltStepFrac,
        intRandomSeed=intRandomSeed)


elif growingCriteria=="grow_TS_QBC_no_diversity_OneTarget":
    grow_TS_QBC_no_diversity_OneTarget(model['rf'], model['xgb'],
        xTrainVal, y1TrainVal, y2TrainVal, y3TrainVal,
        xTest, y1Test, y2Test, y3Test,
        target,
        strSaveDir=outputDir,
        strSaveName=dataset + '_' + target +'growing_TS_QBC_no_diversity_1target',
        strModel1Name='rf',
        strModel2Name='xgb',
        fltStepFrac=fltStepFrac,
        intRandomSeed=intRandomSeed)

elif growingCriteria=="grow_DPP_TS_QBC_OneTarget":
    grow_DPP_TS_QBC_OneTarget(model['rf'], model['xgb'],
        xTrainVal, y1TrainVal, y2TrainVal, y3TrainVal,
        xTest, y1Test, y2Test, y3Test,
        target,
        strSaveDir=outputDir,
        strSaveName=dataset + '_' + target +'grow_DPP_TS_QBC_OneTarget',
        strModel1Name='rf',
        strModel2Name='xgb',
        fltStepFrac=fltStepFrac,
        intRandomSeed=intRandomSeed)

elif growingCriteria=='grow2QBC':
    grow2QBC(model['rf'], model['xgb'],
                           xTrainVal, y1TrainVal,
                           y2TrainVal,y3TrainVal,
                           xTest, y1Test,
                           y2Test,y3Test,
                           target1,
                           target2,
                           strSaveDir = outputDir,
                           strSaveName =dataset + '_'+target+ 'grow2QBC',
                           strModel1Name = 'rf',
                           strModel2Name = 'xgb',
                           fltStepFrac = fltStepFrac,
                           fltTrainFracStop = fltTrainFracStop,
                           intRandomSeed = intRandomSeed)




elif growingCriteria== 'grow_DPP_TS_QBC_TwoTargets':
    grow_DPP_TS_QBC_TwoTargets(model['rf'], model['xgb'],
                           xTrainVal, y1TrainVal,
                           y2TrainVal,y3TrainVal,
                           xTest, y1Test,
                           y2Test,y3Test,
                           target1,
                           target2,
                           strSaveDir = outputDir,
                           strSaveName =dataset + '_'+target+ 'grow_DPP_TS_QBC_2targets',
                           strModel1Name = 'rf',
                           strModel2Name = 'xgb',
                           fltStepFrac = fltStepFrac,
                           intRandomSeed = intRandomSeed)

elif growingCriteria== 'grow_disagreementHV_topM_kmedoids_TwoTargets':
    grow_disagreementHV_topM_kmedoids_TwoTargets(
    model['rf'], model['xgb'],
                           xTrainVal, y1TrainVal,
                           y2TrainVal,y3TrainVal,
                           xTest, y1Test,
                           y2Test,y3Test,
                           target1,
                           target2,
                           strSaveDir = outputDir,
                           strSaveName =dataset + '_'+target+ 'grow_disagreementHV_topM_kmedoids_TwoTargets',
                           strModel1Name = 'rf',
                           strModel2Name = 'xgb',
                           fltStepFrac = fltStepFrac,
                           intRandomSeed = intRandomSeed)


elif growingCriteria== 'grow_disagreementHV_no_diversity_TwoTargets':
    grow_disagreementHV_no_diversity_TwoTargets(model['rf'], model['xgb'],
                               xTrainVal, y1TrainVal,
                               y2TrainVal,y3TrainVal,
                               xTest, y1Test,
                               y2Test,y3Test,
                               target1,
                               target2,
                               strSaveDir = outputDir,
                               strSaveName =dataset + '_'+target+ 'growing_HV_no_diversity_2targets',
                               strModel1Name = 'rf',
                               strModel2Name = 'xgb',
                               fltStepFrac = fltStepFrac,
                               intRandomSeed = intRandomSeed)



elif growingCriteria== 'grow_TS_QBC_no_diversity_TwoTargets':
    grow_TS_QBC_no_diversity_TwoTargets(model['rf'], model['xgb'],
                               xTrainVal, y1TrainVal,
                               y2TrainVal,y3TrainVal,
                               xTest, y1Test,
                               y2Test,y3Test,
                               target1,
                               target2,
                               strSaveDir = outputDir,
                               strSaveName =dataset + '_'+target+ 'grow_TS_QBC_no_diversity_TwoTargets',
                               strModel1Name = 'rf',
                               strModel2Name = 'xgb',
                               fltStepFrac = fltStepFrac,
                               intRandomSeed = intRandomSeed)

else:
    print('Invalid growing criteria!')
    print('Please enter one of the following: random, QBC, rfMaxUncertainty, xgbMaxUncertainty_ibug')
    print('Exiting...')
    exit()   

tEnd = time.time()
print('Time elapsed: ', round(tEnd - tStart, 2), ' seconds')
print ('Done growing by ', growingCriteria, ' on ', dataset, '-', '!')
