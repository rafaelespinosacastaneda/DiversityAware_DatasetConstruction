import numpy as np
import scipy.linalg
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, normalize as sk_normalize

import numpy as np
from sklearn.metrics import pairwise_distances

from myfunc import get_scores 
# Number of decimals when saving the data
n_round=6
n_train_ratio=2
def minMaxStandarize(array_to_standarize):
    min_=np.min(array_to_standarize)
    max_=np.max(array_to_standarize)
    standarized_array=(array_to_standarize-min_)/(max_-min_)
    return standarized_array


def splitTrainVal (xTrainVal, yTrainVal, lstTrainIndices):
    '''
    Split the training+validation set (non-holdout) into training and validation sets

    Parameters
    ----------
    xTrainVal : pandas.DataFrame
        Features of the training+validation set.

    yTrainVal : pandas.DataFrame
        Target of the training+validation set.

    lstTrainIndices : list
        List of indices to use for the training set.

    Returns
    -------
    xTrain : pandas.DataFrame
        Features of the training set.

    xVal : pandas.DataFrame
        Features of the validation set.

    yTrain : pandas.DataFrame
        Target of the training set.

    yVal : pandas.DataFrame
        Target of the validation set.
    '''

    # import dependencies
    import pandas as pd

    # subsample the TrainVal set down to the training set based on the indicies passed

    y1TrainVal=yTrainVal["y1TrainVal"]
    y2TrainVal=yTrainVal["y2TrainVal"]
    y3TrainVal=yTrainVal["y3TrainVal"]

    xTrain = xTrainVal.loc[lstTrainIndices]
    y1Train = y1TrainVal.loc[lstTrainIndices]
    y2Train = y2TrainVal.loc[lstTrainIndices]
    y3Train = y3TrainVal.loc[lstTrainIndices]



    # get the validation set 
    xVal = xTrainVal.drop(lstTrainIndices)
    y1Val = y1TrainVal.drop(lstTrainIndices)
    y2Val = y2TrainVal.drop(lstTrainIndices)
    y3Val = y3TrainVal.drop(lstTrainIndices)

    yTrain=pd.DataFrame({'y1Train':y1Train,'y2Train':y2Train,'y3Train':y3Train})
    yVal=pd.DataFrame({'y1Val':y1Val,'y2Val':y2Val,'y3Val':y3Val})

    return xTrain, xVal, yTrain, yVal



   
def grow_random (model1, 
                 model2,xTrainVal, y1TrainVal,
                y2TrainVal,y3TrainVal,
                xTest, y1Test,
                y2Test,y3Test, 
                strSaveDir = '',
                strSaveName = 'growing_random',
                strModel1Name = 'rf',
                strModel2Name = 'xgb',
                fltStepFrac = 0.01,
                intRandomSeed = 0):
    '''
    Grow the training set from a randomly selected stepfrac number of points up to the full training+validation 
    set by iterativly adding a stepfrac number of points, which are selected from the validation set at random

    Parameters
    ----------
    model1 : regressor
        The first model to be evaluated

    model2 : regressor
        The second model to be evaluated

    xTrainVal : pandas.DataFrame
        Features of the training+validation set
        shape = (n_samples, n_features)

    yTrainVal : pandas.Series
        Target of the training+validation set
        shape = (n_samples,)

    xTest : pandas.DataFrame
        Features of the test set
        shape = (n_samples, n_features)

    yTest : pandas.Series
        Target of the test set
        shape = (n_samples,)

    strSaveDir : str, optional
        The path to save the results to. The default is '' which means the results will be saved in the same
        directory as the script.

    strSaveName : str, optional
        The name of the file to save the results to. The default is 'growing_xgbMaxUncertainty_ibug'.

    strModel1Name : str, optional
        The name of the first model. The default is 'model1'.

    strModel2Name : str, optional
        The name of the second model. The default is 'model2'.

    fltStepFrac : float, optional
        The fraction of the training+validation set to add to the training set at each step. The default is 0.01
        which means that the training set will grow by 1% at each step.

    fltTrainFracStop : float, optional
        The fraction of the training+validation set to stop growing the training set at. The default is 1.0,
        which means that the training set will grow to the full training+validation set.
    
    intRandomSeed : int, optional
        The random state to use for the random selection of points from the validation set. The default is 0.

    Returns
    -------
    dfGrowing_random_model1: pandas.DataFrame
        Dataframe containing the results of the growing training set, evaluated using model1 on the test set

    dfGrowing_random_model2: pandas.DataFrame
        Dataframe containing the results of the growing training set, evaluated using model2 on the test set

    '''

    # import dependencies
    import random
    import numpy as np
    import pandas as pd
    from myfunc import get_scores                                               # CHANGE DEPENDING ON KANGMING'S FILE NAME 
    import time

    # check that the path given ends with a '/'
    if strSaveDir != '':
        if strSaveDir[-1] != '/':
            strSaveDir += '/'
    print('RANDOM SEED IN THE GROW RANDOM FUNCTION')
    print(intRandomSeed)
    # initialize dataframe to hold results
    # will have 'train_ratio', 'maes', 'rmse','r2', 'train_size' columns
    dfGrowing_random_model1 = pd.DataFrame()
    dfGrowing_random_model2 = pd.DataFrame()

    # set the random state
    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    # initialize the size of the training set
    intTempTrainSize = int(len(xTrainVal)*fltStepFrac)
    # initialize the step size
    intStepSize = int(len(xTrainVal)*fltStepFrac)

    # initialize a list with the indices of the first randomly selected training set
    lstTrainIndices = np.random.choice(xTrainVal.index, intTempTrainSize, replace=False).tolist()


    while intTempTrainSize < (len(xTrainVal)+1):
        # start time
        timeStart = time.time()


        # --- SUBSAMPLE THE TRAININGVAL SET ---
        yTrainVal=pd.DataFrame({'y1TrainVal':y1TrainVal,'y2TrainVal':y2TrainVal,'y3TrainVal':y3TrainVal})

        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(xTrainVal, yTrainVal, lstTrainIndices)
        y1Train_temp=yTrain_temp["y1Train"]
        y2Train_temp=yTrain_temp["y2Train"]
        y3Train_temp=yTrain_temp["y3Train"]
        
       
        assert len(set(xTrainVal.index).intersection(xTest.index)) == 0, "Train/Test sets have overlapping indices!"
        assert len(set(xTrain_temp.index).intersection(xVal_temp.index)) == 0, "Train/Val sets have overlapping indices!"
        assert len(set(xTrain_temp.index).intersection(xTest.index)) == 0, "Train/Val sets have overlapping indices!"



        # --- CALCULATE, PRINT AND STORE RESULTS ---
        print('\n--------------------------------------------------------------------------------------------')
        print('Training set size: ', len(xTrain_temp))
        print('Training set percentage: ', '{:.1%}'.format(len(xTrain_temp)/len(xTrainVal)), '\n')

        print("------------------MODEL1 METRICS FOR EFORM------------------ ")
        _,_1maes_1, _1rmse_1, _1r2_1 = get_scores(model1,
                                          xTrain_temp,y1Train_temp,
                                          xTest,y1Test,xVal_temp)
        print("------------------MODEL1 METRICS FOR BULK MODULUS ------------------")

        _,_2maes_1, _2rmse_1, _2r2_1 = get_scores(model1,
                                          xTrain_temp,y2Train_temp,
                                          xTest,y2Test,xVal_temp)
        print("------------------MODEL1 METRICS FOR BAND GAP------------------ ")

        _,_3maes_1, _3rmse_1, _3r2_1 = get_scores(model1,
                                          xTrain_temp,y3Train_temp,
                                          xTest,y3Test,xVal_temp)
        

        print("------------------MODEL2 METRICS FOR EFORM------------------ ")
        _,_1maes_2, _1rmse_2, _1r2_2 = get_scores(model2,
                                          xTrain_temp,y1Train_temp,
                                          xTest,y1Test,xVal_temp)
        print("------------------MODEL2 METRICS FOR BULK MODULUS ------------------")

        _,_2maes_2, _2rmse_2, _2r2_2 = get_scores(model2,
                                          xTrain_temp,y2Train_temp,
                                          xTest,y2Test,xVal_temp)
        print("------------------MODEL2 METRICS FOR BAND GAP------------------ ")

        _,_3maes_2, _3rmse_2, _3r2_2 = get_scores(model2,
                                          xTrain_temp,y3Train_temp,
                                          xTest,y3Test,xVal_temp)
        
        
        # store the results for the first model predictions

        dfGrowing_random_model1 = pd.concat([dfGrowing_random_model1,
                                                       pd.DataFrame({'train_ratio': [round(len(xTrain_temp)/len(xTrainVal), 2)],
                                                                     'maes_eform': [round(_1maes_1, n_round)],
                                                                     'rmse_eform': [round(_1rmse_1, n_round)],
                                                                     'r2_eform': [round(_1r2_1, n_round)],
                                                                     'maes_bulk_modulus':[round(_2maes_1, n_round)],
                                                                     'rmse_bulk_modulus':[round(_2rmse_1, n_round)],
                                                                     'r2_bulk_modulus':[round(_2r2_1, n_round)],
                                                                     'maes_bandgap':[round(_3maes_1, n_round)] ,
                                                                     'rmse_bandgap':[round(_3rmse_1, n_round)] ,
                                                                     'r2_bandgap':[round(_3r2_1, n_round)] ,
                                                                     'train_size': [len(xTrain_temp)]})], 
                                                      ignore_index=True)
        


        dfGrowing_random_model2 = pd.concat([dfGrowing_random_model2,
                                                       pd.DataFrame({'train_ratio': [round(len(xTrain_temp)/len(xTrainVal), 2)],
                                                                     'maes_eform': [round(_1maes_2, n_round)],
                                                                     'rmse_eform': [round(_1rmse_2, n_round)],
                                                                     'r2_eform': [round(_1r2_2, n_round)],
                                                                     'maes_bulk_modulus':[round(_2maes_2, n_round)],
                                                                     'rmse_bulk_modulus':[round(_2rmse_2, n_round)],
                                                                     'r2_bulk_modulus':[round(_2r2_2, n_round)],
                                                                     'maes_bandgap':[round(_3maes_2, n_round)] ,
                                                                     'rmse_bandgap':[round(_3rmse_2, n_round)] ,
                                                                     'r2_bandgap':[round(_3r2_2, n_round)] ,
                                                                     'train_size': [len(xTrain_temp)]})], 
                                                      ignore_index=True)
        # --- SAVE THE RESULTS ---
        # save the results to a csv file
        dfGrowing_random_model1.to_csv(strSaveDir + strSaveName + '_' + strModel1Name + str(intRandomSeed)+'.csv')
        dfGrowing_random_model2.to_csv(strSaveDir + strSaveName + '_' + strModel2Name +  str(intRandomSeed)+ '.csv')
        # make a dataframe with the indices of the training set
        dfTrainIndices = pd.DataFrame(lstTrainIndices, columns=['trainIndices'])
        # save the indices of the training set to a csv file
        dfTrainIndices.to_csv(strSaveDir + strSaveName + '_trainIndices.csv')


        # --- GROWING CRITERIA ---
        # check if the training set is the full training+validation set
        if intTempTrainSize == len(xTrainVal):
            # if so, break the loop
            intTempTrainSize+=10
            break

        if intTempTrainSize + intStepSize > len(xTrainVal):
            # if the next step would be larger than the full training+validation set, set the step size to the
            # difference between the current training set size and the full training+validation set size
            intStepSize = len(xTrainVal) - intTempTrainSize
    
        # randomly select the next stepfrac indices from xVal_temp
        lstTempIndices = np.random.choice(xVal_temp.index, intStepSize, replace=False).tolist()

        # --- UPDATE THE TRAINING SET SIZE ---
        # add the indices to the list of training indices
        lstTrainIndices = lstTrainIndices + lstTempIndices
        # update the training set size
        intTempTrainSize = len(lstTrainIndices)

        # end time
        timeEnd = time.time()
        # print the time it took to run the loop
        print('Time to run loop: ', round(timeEnd-timeStart, 2), ' seconds', flush=True)
        print('\n--------------------------------------------------------------------------------------------')
    # --- SAVE THE RESULTS ---
    # save the results to a csv file
    dfGrowing_random_model1.to_csv(strSaveDir + strSaveName + '_' + strModel1Name + str(intRandomSeed)+'.csv')
    dfGrowing_random_model2.to_csv(strSaveDir + strSaveName + '_' + strModel2Name +  str(intRandomSeed)+ '.csv')

    # make a dataframe with the indices of the training set
    dfTrainIndices = pd.DataFrame(lstTrainIndices, columns=['trainIndices'])
    # save the indices of the training set to a csv file
    dfTrainIndices.to_csv(strSaveDir + strSaveName + '_trainIndices.csv')

    return


def grow_QBC(model1, model2,
             xTrainVal, y1TrainVal,
             y2TrainVal,y3TrainVal,
             xTest, y1Test,
             y2Test,y3Test,
             target,
             strSaveDir = '',
             strSaveName ='growing_QBC',
             strModel1Name = 'rf',
             strModel2Name = 'xgb',
             fltStepFrac = 0.01,
             intRandomSeed =0):

    '''
    Grow the training set from a randomly selected stepfrac number of points up to the full training+validation 
    set by iterativly adding a stepfrac number of points, which are selected from the validation set by querying 
    the 'committee' of model1 and model2 for the points with the highest difference in the two model's predictions

    Parameters
    ----------
    model1 : regressor
        The first model to be evaluated

    model2 : regressor
        The second model to be evaluated

    xTrainVal : pandas.DataFrame
        Features of the training+validation set
        shape = (n_samples, n_features)

    yTrainVal : pandas.Series
        Target of the training+validation set
        shape = (n_samples,)

    xTest : pandas.DataFrame
        Features of the test set
        shape = (n_samples, n_features)

    yTest : pandas.Series
        Target of the test set
        shape = (n_samples,)

    strSaveDir : str, optional
        The path to save the results to. The default is '' which means the results will be saved in the same
        directory as the script.

    strSaveName : str, optional
        The name of the file to save the results to. The default is 'growing_xgbMaxUncertainty_ibug'.

    strModel1Name : str, optional
        The name of the first model. The default is 'model1'.

    strModel2Name : str, optional
        The name of the second model. The default is 'model2'.

    fltStepFrac : float, optional
        The fraction of the training+validation set to add to the training set at each step. The default is 0.01
        which means that the training set will grow by 1% at each step.

    fltTrainFracStop : float, optional
        The fraction of the training+validation set to stop growing the training set at. The default is 1.0,
        which means that the training set will grow to the full training+validation set.
    
    intRandomSeed : int, optional
        The random seed to use. The default is 0.

    Returns
    -------
    dfGrowing_QBC_model1 : pandas.DataFrame
        Dataframe containing the results of the growing training set, evaluated using the model1 on the test set

    dfGrowing_QBC_model2 : pandas.DataFrame
        Dataframe containing the results of the growing training set, evaluated using the model2 on the test set
    '''

    # import dependencies
    import random
    import numpy as np
    import pandas as pd
    from myfunc import get_scores                                               # CHANGE DEPENDING ON KANGMING'S FILE NAME
    import time 

    # check that the path given ends with a '/'
    if strSaveDir != '':
        if strSaveDir[-1] != '/':
            strSaveDir += '/'

    # initialize dataframe to hold results
    # will have 'train_ratio', 'maes', 'rmse','r2', 'train_size' columns
    dfGrowing_QBC_model1 = pd.DataFrame()
    dfGrowing_QBC_model2 = pd.DataFrame()

    # set the random state
    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    # intialize the size of the training set
    intTempTrainSize = int(len(xTrainVal)*fltStepFrac)
    intStepSize = int(len(xTrainVal)*fltStepFrac)

    # initialize a list with the indices of the first randomly selected training set
    lstTrainIndices = np.random.choice(xTrainVal.index, intTempTrainSize, replace=False).tolist()



    while intTempTrainSize < (len(xTrainVal)+1):
        # start time
        timeStart = time.time()
   

        yTrainVal=pd.DataFrame({'y1TrainVal':y1TrainVal,'y2TrainVal':y2TrainVal,'y3TrainVal':y3TrainVal})


        # --- SUBSAMPLE THE TRAININGVAL SET ---
        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(xTrainVal, yTrainVal, lstTrainIndices)

        # --- CALCULATE, PRINT AND STORE RESULTS ---
        print('\n--------------------------------------------------------------------------------------------')
        print('Training set size: ', len(xTrain_temp))
        print('Training set percentage: ', '{:.1%}'.format(len(xTrain_temp)/len(xTrainVal)), '\n')



        y1Train_temp=yTrain_temp["y1Train"]
        y2Train_temp=yTrain_temp["y2Train"]
        y3Train_temp=yTrain_temp["y3Train"]
        
    

        if target=='e_form':
            yTrain_temp=y1Train_temp

        if target=='bulk_modulus':
            yTrain_temp=y2Train_temp
        if target=='bandgap':
            yTrain_temp=y3Train_temp

        # get scores using function for the random forest model

        print("------------------MODEL1 METRICS FOR EFORM------------------ ")
        eform_pred_val1,_1maes_1, _1rmse_1, _1r2_1 = get_scores(model1,
                                          xTrain_temp,y1Train_temp,
                                          xTest,y1Test,X_val=xVal_temp)
        print("------------------MODEL1 METRICS FOR BULK MODULUS ------------------")

        bulk_pred_val1,_2maes_1, _2rmse_1, _2r2_1 = get_scores(model1,
                                          xTrain_temp,y2Train_temp,
                                          xTest,y2Test,X_val=xVal_temp)
        print("------------------MODEL1 METRICS FOR BAND GAP------------------ ")

        bandgap_pred_val1,_3maes_1, _3rmse_1, _3r2_1 = get_scores(model1,
                                          xTrain_temp,y3Train_temp,
                                          xTest,y3Test,X_val=xVal_temp)
        


        print("------------------MODEL2 METRICS FOR EFORM------------------ ")
        eform_pred_val2,_1maes_2, _1rmse_2, _1r2_2 = get_scores(model2,
                                          xTrain_temp,y1Train_temp,
                                          xTest,y1Test,X_val=xVal_temp)
        print("------------------MODEL2 METRICS FOR BULK MODULUS ------------------")

        bulk_pred_val2,_2maes_2, _2rmse_2, _2r2_2 = get_scores(model2,
                                          xTrain_temp,y2Train_temp,
                                          xTest,y2Test,X_val=xVal_temp)
        print("------------------MODEL2 METRICS FOR BAND GAP------------------ ")

        bandgap_pred_val2,_3maes_2, _3rmse_2, _3r2_2 = get_scores(model2,
                                          xTrain_temp,y3Train_temp,
                                          xTest,y3Test,X_val=xVal_temp)
        
    
        # store the results for the random forest model predictions
        dfGrowing_QBC_model1 = pd.concat([dfGrowing_QBC_model1,
                                                       pd.DataFrame({'train_ratio': [round(len(xTrain_temp)/len(xTrainVal), n_train_ratio)],
                                                                     'maes_eform': [round(_1maes_1, n_round)],
                                                                     'rmse_eform': [round(_1rmse_1, n_round)],
                                                                     'r2_eform': [round(_1r2_1, n_round)],
                                                                     'maes_bulk_modulus':[round(_2maes_1, n_round)],
                                                                     'rmse_bulk_modulus':[round(_2rmse_1, n_round)],
                                                                     'r2_bulk_modulus':[round(_2r2_1, n_round)],
                                                                     'maes_bandgap':[round(_3maes_1, n_round)] ,
                                                                     'rmse_bandgap':[round(_3rmse_1, n_round)] ,
                                                                     'r2_bandgap':[round(_3r2_1, n_round)] ,
                                                                     'train_size': [len(xTrain_temp)]})], 
                                                      ignore_index=True)
        # store the results for the extra model predictions
        dfGrowing_QBC_model2 = pd.concat([dfGrowing_QBC_model2,
                                                       pd.DataFrame({'train_ratio': [round(len(xTrain_temp)/len(xTrainVal), n_train_ratio)],
                                                                     'maes_eform': [round(_1maes_2, n_round)],
                                                                     'rmse_eform': [round(_1rmse_2, n_round)],
                                                                     'r2_eform': [round(_1r2_2, n_round)],
                                                                     'maes_bulk_modulus':[round(_2maes_2, n_round)],
                                                                     'rmse_bulk_modulus':[round(_2rmse_2, n_round)],
                                                                     'r2_bulk_modulus':[round(_2r2_2, n_round)],
                                                                     'maes_bandgap':[round(_3maes_2, n_round)] ,
                                                                     'rmse_bandgap':[round(_3rmse_2, n_round)] ,
                                                                     'r2_bandgap':[round(_3r2_2, n_round)] ,
                                                                     'train_size': [len(xTrain_temp)]})], 
                                                      ignore_index=True)
        
        # --- GROWING CRITERIA ---
        # check if the training set is the full training+validation set
        if intTempTrainSize == len(xTrainVal):
            # if so, break the loop
            intTempTrainSize+=10
            break

        # get the predictions of model 1 and 2 on the validation set      
        # 
        # The target controls which property determines the QBC disagreement selection  
        
        if target=='e_form':
            xVal_temp_pred1=eform_pred_val1
            xVal_temp_pred2=eform_pred_val2

        if target=='bulk_modulus':
            xVal_temp_pred1=bulk_pred_val1
            xVal_temp_pred2=bulk_pred_val2

        if target=='bandgap':
            xVal_temp_pred1=bandgap_pred_val1
            xVal_temp_pred2=bandgap_pred_val2


        # get the difference between the two models' predictions
        yVal_temp_diff = abs(xVal_temp_pred1 - xVal_temp_pred2)
        yVal_temp_diff = pd.Series(yVal_temp_diff, index = yVal_temp.index) 
        # get the indices of the intStepSize points with the highest difference
        lstMaxDiffIndices = yVal_temp_diff.nlargest(intStepSize).index.tolist()

        # --- UPDATE THE TRAINING SET ---
        # add the indices to the training set
        lstTrainIndices = lstTrainIndices + lstMaxDiffIndices
        # update the size of the training set
        intTempTrainSize = len(lstTrainIndices)

        # end time
        timeEnd = time.time()
        # print the time it took to run the loop
        print('Time to run loop: ', round(timeEnd-timeStart, 2), ' seconds', flush=True)
        print('\n--------------------------------------------------------------------------------------------')

    # --- SAVE THE RESULTS ---
    # save the results to a csv file
    dfGrowing_QBC_model1.to_csv(strSaveDir + strSaveName + '_' + strModel1Name + '.csv')
    dfGrowing_QBC_model2.to_csv(strSaveDir + strSaveName + '_' + strModel2Name + '.csv')
    # make a dataframe with the indices of the training set
    dfTrainIndices = pd.DataFrame(lstTrainIndices, columns=['trainIndices'])
    # save the indices of the training set to a csv file
    dfTrainIndices.to_csv(strSaveDir + strSaveName + '_trainIndices.csv')
    return



# ============================================================
# Utilities
# ============================================================
def minMaxStandarize(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax - xmin < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - xmin) / (xmax - xmin + eps)


# ============================================================
# NSGA-II components (works for M objectives)
# ============================================================
def _fast_non_dominated_sort(F):
    """
    F: (pop, M), higher is better for all objectives.
    Returns: fronts (list of lists), rank (array)
    """
    pop = F.shape[0]
    S = [[] for _ in range(pop)]
    n = np.zeros(pop, dtype=int)
    rank = np.zeros(pop, dtype=int)
    fronts = [[]]

    for p in range(pop):
        for q in range(pop):
            if p == q:
                continue
            if np.all(F[p] >= F[q]) and np.any(F[p] > F[q]):
                S[p].append(q)
            elif np.all(F[q] >= F[p]) and np.any(F[q] > F[p]):
                n[p] += 1

        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()
    return fronts, rank


def _crowding_distance(F, front):
    """
    F: (pop, M), front: list of indices
    Returns: dict idx->crowding (higher is better)
    """
    if len(front) == 0:
        return {}
    if len(front) <= 2:
        return {i: np.inf for i in front}

    crowd = {i: 0.0 for i in front}
    M = F.shape[1]

    for m in range(M):
        vals = np.array([F[i, m] for i in front], dtype=float)
        order = np.argsort(vals)
        sorted_idx = [front[j] for j in order]

        crowd[sorted_idx[0]] = np.inf
        crowd[sorted_idx[-1]] = np.inf

        vmin = vals[order[0]]
        vmax = vals[order[-1]]
        denom = (vmax - vmin) if (vmax > vmin) else 1.0

        for k in range(1, len(front) - 1):
            i_mid = sorted_idx[k]
            i_prev = sorted_idx[k - 1]
            i_next = sorted_idx[k + 1]
            crowd[i_mid] += (F[i_next, m] - F[i_prev, m]) / denom

    return crowd


def _tournament_select(pop_indices, rank, crowd, rng, k=2):
    """Binary tournament using (lower rank better) then (higher crowding better)."""
    best = None
    for _ in range(k):
        i = int(rng.choice(pop_indices))
        if best is None:
            best = i
            continue
        if rank[i] < rank[best]:
            best = i
        elif rank[i] == rank[best]:
            if crowd.get(i, 0.0) > crowd.get(best, 0.0):
                best = i
    return best


def _make_unique(chrom, pool_size, rng):
    """Repair duplicates by resampling."""
    chrom = np.array(chrom, dtype=int)
    used = set()
    for t in range(len(chrom)):
        if int(chrom[t]) in used:
            while True:
                r = int(rng.integers(0, pool_size))
                if r not in used:
                    chrom[t] = r
                    break
        used.add(int(chrom[t]))
    return chrom


def _ordered_crossover(a, b, rng):
    """Safe order crossover (OX) for set/list chromosomes."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    B = a.size
    if B < 2:
        return a.copy(), b.copy()

    i, j = sorted(rng.choice(np.arange(B), size=2, replace=False))
    if i == j:
        j = min(B, i + 1)

    child1 = np.full(B, -1, dtype=int)
    child2 = np.full(B, -1, dtype=int)

    child1[i:j] = a[i:j]
    child2[i:j] = b[i:j]

    def fill(child, parent):
        taken = set(child[child != -1].tolist())
        remaining = [g for g in parent.tolist() if g not in taken]
        empty_pos = np.where(child == -1)[0]
        m = min(len(empty_pos), len(remaining))
        child[empty_pos[:m]] = remaining[:m]
        if np.any(child == -1):
            left = np.where(child == -1)[0]
            child[left] = rng.choice(parent, size=left.size, replace=True)
        return child

    return fill(child1, b), fill(child2, a)


def _mutate_swap(chrom, pool_size, rng, p_mut=0.2):
    """Replace one gene with random pool index."""
    chrom = chrom.copy()
    if rng.random() < p_mut:
        t = int(rng.integers(0, len(chrom)))
        chrom[t] = int(rng.integers(0, pool_size))
    return chrom


# ============================================================
# 2-objective EA batch selection:
#   (quality1, quality2)  -- NO diversity
# ============================================================
def EA_select_batch_twoQuality_only(
    candidates: pd.DataFrame,
    diffs_df: pd.DataFrame,          # columns ['q1','q2'] aligned to candidates.index
    step_size: int,
    pop_size: int = 120,
    n_gen: int = 60,
    p_cx: float = 0.9,
    p_mut: float = 0.25,
    random_state: int = 0,
):
    """
    Objectives (maximize):
      1) mean q1 over batch
      2) mean q2 over batch

    Returns: list of candidates.index (length <= step_size)
    """
    rng = np.random.default_rng(random_state)

    n_pool = len(candidates)
    B = int(min(step_size, n_pool))
    if B <= 0:
        return []

    diffs_df = diffs_df.reindex(candidates.index)
    q1 = np.nan_to_num(diffs_df["q1"].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    q2 = np.nan_to_num(diffs_df["q2"].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def fitness(chrom):
        idx = np.asarray(chrom, dtype=int)
        return float(np.mean(q1[idx])), float(np.mean(q2[idx]))  # 2 objectives

    pop = [rng.choice(n_pool, size=B, replace=False).astype(int) for _ in range(pop_size)]

    for _ in range(n_gen):
        F = np.array([fitness(ind) for ind in pop], dtype=float)  # (pop,2)
        fronts, rank = _fast_non_dominated_sort(F)
        crowd = {}
        for fr in fronts:
            crowd.update(_crowding_distance(F, fr))

        offspring = []
        pop_indices = np.arange(len(pop))

        while len(offspring) < pop_size:
            p1 = pop[_tournament_select(pop_indices, rank, crowd, rng)]
            p2 = pop[_tournament_select(pop_indices, rank, crowd, rng)]

            c1, c2 = p1.copy(), p2.copy()
            if rng.random() < p_cx:
                c1, c2 = _ordered_crossover(c1, c2, rng)

            c1 = _make_unique(c1, n_pool, rng)
            c2 = _make_unique(c2, n_pool, rng)

            c1 = _mutate_swap(c1, n_pool, rng, p_mut=p_mut)
            c2 = _mutate_swap(c2, n_pool, rng, p_mut=p_mut)

            c1 = _make_unique(c1, n_pool, rng)
            c2 = _make_unique(c2, n_pool, rng)

            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        combined = pop + offspring
        F2 = np.array([fitness(ind) for ind in combined], dtype=float)
        fronts2, _ = _fast_non_dominated_sort(F2)

        new_pop = []
        for fr in fronts2:
            if len(new_pop) + len(fr) <= pop_size:
                new_pop.extend([combined[i] for i in fr])
            else:
                cd = _crowding_distance(F2, fr)
                fr_sorted = sorted(fr, key=lambda i: cd.get(i, 0.0), reverse=True)
                needed = pop_size - len(new_pop)
                new_pop.extend([combined[i] for i in fr_sorted[:needed]])
                break
        pop = new_pop

    # pick one solution from Pareto front
    F_final = np.array([fitness(ind) for ind in pop], dtype=float)
    fronts, _ = _fast_non_dominated_sort(F_final)
    front0 = fronts[0]
    f0 = F_final[front0]  # (k,2)

    fmin = f0.min(axis=0)
    fmax = f0.max(axis=0)
    denom = np.where((fmax - fmin) > 1e-12, (fmax - fmin), 1.0)
    f0n = (f0 - fmin) / denom
    score = f0n.mean(axis=1)
    best = front0[int(np.argmax(score))]

    best_local = pop[best]
    return candidates.index[np.asarray(best_local, dtype=int)].tolist()


# ============================================================
#   - NSGA-II selection on (q1,q2) -- NO diversity
# ============================================================
def grow_EA_QBC_2Obj_twoTargets_from3Outcomes(
    model1, model2,
    xTrainVal, y1TrainVal, y2TrainVal, y3TrainVal,
    xTest, y1Test, y2Test, y3Test,
    target1: str,
    target2: str,
    # names of the 3 outcomes in THIS dataset:
    outcome_names=("e_form", "bulk_modulus", "bandgap"),
    strSaveDir="",
    strSaveName="growing_EA_2obj_qbc_only",
    strModel1Name="model1",
    strModel2Name="model2",
    # loop control
    fltStepFrac=0.01,
    fltTrainFracStop=1.0,
    intRandomSeed=0,
    # EA params
    pop_size=120,
    n_gen=60,
    p_cx=0.9,
    p_mut=0.25,
    # rounding for logs
    n_round=6,
    n_train_ratio=6,
):
    """
    Keeps EXACTLY the EA logic of your first code, but with the "3 outcomes passed separately" interface.

    Requirements:
      - splitTrainVal_fn(XTrainVal, YTrainVal_df, lstTrainIndices) -> xTrain_temp, xVal_temp, yTrain_temp_df, yVal_temp_df
      - get_scores_fn(model, X_train, y_train, X_test, y_test, X_val=...) -> pred_val, mae, rmse, r2
        (positional X_val also supported)

    Returns:
      df_model1, df_model2, lstTrainIndices
    """
    # import dependencies
    import random
    import numpy as np
    import pandas as pd
    from myfunc import get_scores                                               # CHANGE DEPENDING ON KANGMING'S FILE NAME 
    import time

  
    name1, name2, name3 = outcome_names
    valid = {name1, name2, name3}
    if target1 not in valid or target2 not in valid:
        raise ValueError(f"target1/target2 must be in {valid}. Got: {target1}, {target2}")

    # path
    if strSaveDir != "" and strSaveDir[-1] != "/":
        strSaveDir += "/"

    #rng = np.random.default_rng(intRandomSeed)



    # set the random state
    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    # intialize the size of the training set
    intTempTrainSize = int(len(xTrainVal)*fltStepFrac)
    # intialize the step size
    #intStepSize = int(len(xTrainVal)*fltStepFrac)

    






    # init sizes
    init_size = max(1, int(len(xTrainVal) * fltStepFrac))
    step_size = max(1, int(len(xTrainVal) * fltStepFrac))

    

    # initialize a list with the indices of the first randomly selected training set
    lstTrainIndices = np.random.choice(xTrainVal.index, intTempTrainSize, replace=False).tolist()

    # pack y's into one df for splitTrainVal_fn
    outcome_list = [name1, name2, name3]



    # pack y's into one df for splitTrainVal_fn (MATCHES your splitTrainVal expectations)
    yTrainVal_df = pd.DataFrame({
        "y1TrainVal": y1TrainVal,
        "y2TrainVal": y2TrainVal,
        "y3TrainVal": y3TrainVal,
    })

    # map semantic outcome names -> these internal column keys
    name1, name2, name3 = outcome_names  # e.g. ("e_form","bulk_modulus","bandgap")

    # IMPORTANT: splitTrainVal RETURNS these names:
    outcome_to_traincol = {name1: "y1Train", name2: "y2Train", name3: "y3Train"}
    outcome_to_valcol   = {name1: "y1Val",   name2: "y2Val",   name3: "y3Val"}


    # test targets in same mapping
    yTest_map = {
        name1: y1Test,
        name2: y2Test,
        name3: y3Test,
    }








    # logs
    df_model1 = pd.DataFrame()
    df_model2 = pd.DataFrame()

    # loop
    while True:
        #train_frac = len(lstTrainIndices) / len(xTrainVal)
        #if train_frac >= fltTrainFracStop:
        #    break

        # split
        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(xTrainVal, yTrainVal_df, lstTrainIndices)

        print("\n------------------------------------------------------------")
        print("Training set size:", len(xTrain_temp))
        print("Training %:       ", "{:.3%}".format(len(xTrain_temp) / len(xTrainVal)), "\n")

        #if len(xVal_temp) == 0:
        #    print("No validation candidates left. Stopping.")
        #    break

        # train/predict for ALL 3 outcomes (for logging + to have pred_val for QBC)
        pred_val_1 = {}
        pred_val_2 = {}
        metrics_1 = {}
        metrics_2 = {}

        for nm in outcome_list:
            ytr = yTrain_temp[outcome_to_traincol[nm]]
            yte = yTest_map[nm]

            pv1, mae1, rmse1, r21 = get_scores(model1, xTrain_temp, ytr, xTest, yte, xVal_temp)
            pv2, mae2, rmse2, r22 = get_scores(model2, xTrain_temp, ytr, xTest, yte, xVal_temp)

            pred_val_1[nm] = np.asarray(pv1)
            pred_val_2[nm] = np.asarray(pv2)

            metrics_1[nm] = (mae1, rmse1, r21)
            metrics_2[nm] = (mae2, rmse2, r22)

        # log rows (same schema style; outcome-specific column names)
        row1 = {"train_ratio": round(len(xTrain_temp) / len(xTrainVal), n_train_ratio), "train_size": len(xTrain_temp)}
        row2 = {"train_ratio": round(len(xTrain_temp) / len(xTrainVal), n_train_ratio), "train_size": len(xTrain_temp)}
        for nm in outcome_list:
            mae1, rmse1, r21 = metrics_1[nm]
            mae2, rmse2, r22 = metrics_2[nm]
            row1[f"mae_{nm}"] = round(float(mae1), n_round)
            row1[f"rmse_{nm}"] = round(float(rmse1), n_round)
            row1[f"r2_{nm}"] = round(float(r21), n_round)
            row2[f"mae_{nm}"] = round(float(mae2), n_round)
            row2[f"rmse_{nm}"] = round(float(rmse2), n_round)
            row2[f"r2_{nm}"] = round(float(r22), n_round)

        df_model1 = pd.concat([df_model1, pd.DataFrame([row1])], ignore_index=True)
        df_model2 = pd.concat([df_model2, pd.DataFrame([row2])], ignore_index=True)

        # ---- QBC diffs for chosen 2 targets (ABS), then min-max normalize per target ----
        diff1_abs = np.abs(pred_val_1[target1] - pred_val_2[target1])
        diff2_abs = np.abs(pred_val_1[target2] - pred_val_2[target2])

        q1 = minMaxStandarize(diff1_abs)
        q2 = minMaxStandarize(diff2_abs)

        diffs_df = pd.DataFrame({"q1": q1, "q2": q2}, index=xVal_temp.index)

        # ---- EA selection (2-objective ONLY, no diversity) ----
        to_add = EA_select_batch_twoQuality_only(
            candidates=xVal_temp,
            diffs_df=diffs_df,
            step_size=min(step_size, len(xVal_temp)),
            pop_size=pop_size,
            n_gen=n_gen,
            p_cx=p_cx,
            p_mut=p_mut,
            random_state=intRandomSeed,
        )

        if len(to_add) == 0:
            print("EA returned 0 points. Stopping.")
            break

        lstTrainIndices += to_add

        # stop if full
        #if len(lstTrainIndices) >= len(xTrainVal):
        #    lstTrainIndices = lstTrainIndices[: len(xTrainVal)]
        #    break

        # optional: save every iter (like your second code)
        df_model1.to_csv(strSaveDir + strSaveName + "_" + strModel1Name + str(intRandomSeed) + ".csv", index=False)
        df_model2.to_csv(strSaveDir + strSaveName + "_" + strModel2Name + str(intRandomSeed) + ".csv", index=False)
        pd.DataFrame(lstTrainIndices, columns=["trainIndices"]).to_csv(
            strSaveDir + strSaveName + str(intRandomSeed) + "_trainIndices.csv",
            index=False,
        )

    # final save
    df_model1.to_csv(strSaveDir + strSaveName + "_" + strModel1Name + str(intRandomSeed) + ".csv", index=False)
    df_model2.to_csv(strSaveDir + strSaveName + "_" + strModel2Name + str(intRandomSeed) + ".csv", index=False)
    pd.DataFrame(lstTrainIndices, columns=["trainIndices"]).to_csv(
        strSaveDir + strSaveName + str(intRandomSeed) + "_trainIndices.csv",
        index=False,
    )

    return 



def EA_select_batch_twoQuality_plus_diversity(
    trainingdata: pd.DataFrame,
    candidates: pd.DataFrame,
    diffs_df: pd.DataFrame,          # columns ['q1','q2'] aligned to candidates.index
    step_size: int,
    do_scale: bool = True,
    normalize: bool = True,
    pop_size: int = 120,
    n_gen: int = 60,
    p_cx: float = 0.9,
    p_mut: float = 0.25,
    random_state: int = 0,
    # diversity mixing
    w_train: float = 0.5,
    w_within: float = 0.5,
):
    """
    Objectives (maximize):
      1) mean q1 over batch
      2) mean q2 over batch
      3) diversity = w_train * div_train + w_within * div_within

    div_train(B)  = mean_{i in B}(1 - mean_{j in T} cos(x_i, x_j))
    div_within(B) = 1 - mean_{i!=k in B} cos(x_i, x_k)

    Returns: list of candidates.index (length <= step_size)
    """
    rng = np.random.default_rng(random_state)

    # ----- geometry (scale on stacked train+pool, then normalize rows) -----
    X_train = trainingdata.values.astype(np.float32, copy=False)
    X_pool  = candidates.values.astype(np.float32, copy=False)

    if do_scale:
        scaler = StandardScaler()
        X_all0 = np.vstack([X_train, X_pool])
        scaler.fit(X_all0)
        X_train = scaler.transform(X_train).astype(np.float32, copy=False)
        X_pool  = scaler.transform(X_pool).astype(np.float32, copy=False)

    if normalize:
        X_train = sk_normalize(X_train, axis=1)
        X_pool  = sk_normalize(X_pool, axis=1)

    n_train = X_train.shape[0]
    n_pool  = X_pool.shape[0]
    B = int(min(step_size, n_pool))
    if B <= 0:
        return []

    # ----- quality vectors aligned -----
    diffs_df = diffs_df.reindex(candidates.index)
    q1 = np.nan_to_num(diffs_df["q1"].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    q2 = np.nan_to_num(diffs_df["q2"].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # ----- precompute for diversity -----
    # pool-to-train mean cosine similarity for each pool point
    if n_train > 0:
        sim_pool_train_mean = (X_pool @ X_train.T).mean(axis=1).astype(np.float32)  # (n_pool,)
    else:
        sim_pool_train_mean = np.zeros(n_pool, dtype=np.float32)

    # pool-to-pool cosine Gram (for within-batch cosine)
    G_pp = (X_pool @ X_pool.T).astype(np.float32, copy=False)

    def fitness(chrom):
        idx = np.asarray(chrom, dtype=int)

        # Objective 1/2: quality
        qual1 = float(np.mean(q1[idx]))
        qual2 = float(np.mean(q2[idx]))

        # div_train(B) = mean_i (1 - mean_j cos(i, j in train))
        div_train = float(np.mean(1.0 - sim_pool_train_mean[idx]))

        # div_within(B) = 1 - mean_{i!=k} cos(i,k)
        if len(idx) <= 1:
            div_within = 1.0
        else:
            S = G_pp[np.ix_(idx, idx)]
            off = S[~np.eye(len(idx), dtype=bool)]
            div_within = float(1.0 - np.mean(off))

        div = float(w_train * div_train + w_within * div_within)

        return qual1, qual2, div  # 3 objectives

    # ----- init population -----
    pop = [rng.choice(n_pool, size=B, replace=False).astype(int) for _ in range(pop_size)]

    # ----- evolution loop -----
    for _ in range(n_gen):
        F = np.array([fitness(ind) for ind in pop], dtype=float)  # (pop,3)
        fronts, rank = _fast_non_dominated_sort(F)
        crowd = {}
        for fr in fronts:
            crowd.update(_crowding_distance(F, fr))

        offspring = []
        pop_indices = np.arange(len(pop))

        while len(offspring) < pop_size:
            p1 = pop[_tournament_select(pop_indices, rank, crowd, rng)]
            p2 = pop[_tournament_select(pop_indices, rank, crowd, rng)]

            c1, c2 = p1.copy(), p2.copy()
            if rng.random() < p_cx:
                c1, c2 = _ordered_crossover(c1, c2, rng)

            c1 = _make_unique(c1, n_pool, rng)
            c2 = _make_unique(c2, n_pool, rng)

            c1 = _mutate_swap(c1, n_pool, rng, p_mut=p_mut)
            c2 = _mutate_swap(c2, n_pool, rng, p_mut=p_mut)

            c1 = _make_unique(c1, n_pool, rng)
            c2 = _make_unique(c2, n_pool, rng)

            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        combined = pop + offspring
        F2 = np.array([fitness(ind) for ind in combined], dtype=float)
        fronts2, _ = _fast_non_dominated_sort(F2)

        new_pop = []
        for fr in fronts2:
            if len(new_pop) + len(fr) <= pop_size:
                new_pop.extend([combined[i] for i in fr])
            else:
                cd = _crowding_distance(F2, fr)
                fr_sorted = sorted(fr, key=lambda i: cd.get(i, 0.0), reverse=True)
                needed = pop_size - len(new_pop)
                new_pop.extend([combined[i] for i in fr_sorted[:needed]])
                break
        pop = new_pop

    # ----- pick one solution from Pareto front -----
    F_final = np.array([fitness(ind) for ind in pop], dtype=float)
    fronts, _ = _fast_non_dominated_sort(F_final)
    front0 = fronts[0]
    f0 = F_final[front0]  # (k,3)

    # normalize each objective on front and pick max average
    fmin = f0.min(axis=0)
    fmax = f0.max(axis=0)
    denom = np.where((fmax - fmin) > 1e-12, (fmax - fmin), 1.0)
    f0n = (f0 - fmin) / denom
    score = f0n.mean(axis=1)  # equal weights for (q1,q2,div)
    best = front0[int(np.argmax(score))]

    best_local = pop[best]
    return candidates.index[np.asarray(best_local, dtype=int)].tolist()







# ============================================================
#   - NSGA-II selection on (q1,q2) -- WITH diversity
# ============================================================

def grow_EA_QBC_2Obj_twoTargets_plusDiversity_from3Outcomes(
    model1, model2,
    xTrainVal, y1TrainVal, y2TrainVal, y3TrainVal,
    xTest, y1Test, y2Test, y3Test,
    target1: str,
    target2: str,
    # names of the 3 outcomes in THIS dataset:
    outcome_names=("e_form", "bulk_modulus", "bandgap"),
    strSaveDir="",
    strSaveName="growing_EA_2obj_qbc_only",
    strModel1Name="model1",
    strModel2Name="model2",
    # loop control
    fltStepFrac=0.01,
    fltTrainFracStop=1.0,
    intRandomSeed=0,
    # EA params
    pop_size=120,
    n_gen=60,
    p_cx=0.9,
    p_mut=0.25,
    # rounding for logs
    n_round=6,
    n_train_ratio=6,
):
    """
    Keeps EXACTLY the EA logic of your first code, but with the "3 outcomes passed separately" interface.

    Requirements:
      - splitTrainVal_fn(XTrainVal, YTrainVal_df, lstTrainIndices) -> xTrain_temp, xVal_temp, yTrain_temp_df, yVal_temp_df
      - get_scores_fn(model, X_train, y_train, X_test, y_test, X_val=...) -> pred_val, mae, rmse, r2
        (positional X_val also supported)

    Returns:
      df_model1, df_model2, lstTrainIndices
    """
    # import dependencies
    import random
    import numpy as np
    import pandas as pd
    from myfunc import get_scores                                               # CHANGE DEPENDING ON KANGMING'S FILE NAME 
    import time

  
    name1, name2, name3 = outcome_names
    valid = {name1, name2, name3}
    if target1 not in valid or target2 not in valid:
        raise ValueError(f"target1/target2 must be in {valid}. Got: {target1}, {target2}")

    # path
    if strSaveDir != "" and strSaveDir[-1] != "/":
        strSaveDir += "/"

    
    #rng = np.random.default_rng(intRandomSeed)

    # init sizes
    #init_size = max(1, int(len(xTrainVal) * fltStepFrac))
    #step_size = max(1, int(len(xTrainVal) * fltStepFrac))

    # initial random training indices
    #lstTrainIndices = rng.choice(xTrainVal.index.to_numpy(), size=init_size, replace=False).tolist()

    # set the random state
    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    # initialize the size of the training set
    intTempTrainSize = int(len(xTrainVal)*fltStepFrac)
    # initialize the step size
    step_size = int(len(xTrainVal)*fltStepFrac)

    # initialize a list with the indices of the first randomly selected training set
    lstTrainIndices = np.random.choice(xTrainVal.index, intTempTrainSize, replace=False).tolist()






    # pack y's into one df for splitTrainVal_fn
    outcome_list = [name1, name2, name3]



    # pack y's into one df for splitTrainVal_fn (MATCHES your splitTrainVal expectations)
    yTrainVal_df = pd.DataFrame({
        "y1TrainVal": y1TrainVal,
        "y2TrainVal": y2TrainVal,
        "y3TrainVal": y3TrainVal,
    })

    # map semantic outcome names -> these internal column keys
    name1, name2, name3 = outcome_names  # e.g. ("e_form","bulk_modulus","bandgap")

    # IMPORTANT: splitTrainVal RETURNS these names:
    outcome_to_traincol = {name1: "y1Train", name2: "y2Train", name3: "y3Train"}
    outcome_to_valcol   = {name1: "y1Val",   name2: "y2Val",   name3: "y3Val"}


    # test targets in same mapping
    yTest_map = {
        name1: y1Test,
        name2: y2Test,
        name3: y3Test,
    }








    # logs
    df_model1 = pd.DataFrame()
    df_model2 = pd.DataFrame()

    # loop
    while True:
        train_frac = len(lstTrainIndices) / len(xTrainVal)
        #if train_frac >= fltTrainFracStop:
        #    break

        # split
        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(xTrainVal, yTrainVal_df, lstTrainIndices)

        print("\n------------------------------------------------------------")
        print("Training set size:", len(xTrain_temp))
        print("Training %:       ", "{:.3%}".format(len(xTrain_temp) / len(xTrainVal)), "\n")

        #if len(xVal_temp) == 0:
        #    print("No validation candidates left. Stopping.")
        #    break

        # train/predict for ALL 3 outcomes (for logging + to have pred_val for QBC)
        pred_val_1 = {}
        pred_val_2 = {}
        metrics_1 = {}
        metrics_2 = {}

        for nm in outcome_list:
            ytr = yTrain_temp[outcome_to_traincol[nm]]
            yte = yTest_map[nm]

            pv1, mae1, rmse1, r21 = get_scores(model1, xTrain_temp, ytr, xTest, yte, xVal_temp)
            pv2, mae2, rmse2, r22 = get_scores(model2, xTrain_temp, ytr, xTest, yte, xVal_temp)

            pred_val_1[nm] = np.asarray(pv1)
            pred_val_2[nm] = np.asarray(pv2)

            metrics_1[nm] = (mae1, rmse1, r21)
            metrics_2[nm] = (mae2, rmse2, r22)

        # log rows (same schema style; outcome-specific column names)
        row1 = {"train_ratio": round(len(xTrain_temp) / len(xTrainVal), n_train_ratio), "train_size": len(xTrain_temp)}
        row2 = {"train_ratio": round(len(xTrain_temp) / len(xTrainVal), n_train_ratio), "train_size": len(xTrain_temp)}
        for nm in outcome_list:
            mae1, rmse1, r21 = metrics_1[nm]
            mae2, rmse2, r22 = metrics_2[nm]
            row1[f"mae_{nm}"] = round(float(mae1), n_round)
            row1[f"rmse_{nm}"] = round(float(rmse1), n_round)
            row1[f"r2_{nm}"] = round(float(r21), n_round)
            row2[f"mae_{nm}"] = round(float(mae2), n_round)
            row2[f"rmse_{nm}"] = round(float(rmse2), n_round)
            row2[f"r2_{nm}"] = round(float(r22), n_round)

        df_model1 = pd.concat([df_model1, pd.DataFrame([row1])], ignore_index=True)
        df_model2 = pd.concat([df_model2, pd.DataFrame([row2])], ignore_index=True)


        if len(xTrain_temp) / len(xTrainVal)==1:
            break 

        # ---- QBC diffs for chosen 2 targets (ABS), then min-max normalize per target ----
        diff1_abs = np.abs(pred_val_1[target1] - pred_val_2[target1])
        diff2_abs = np.abs(pred_val_1[target2] - pred_val_2[target2])

        q1 = minMaxStandarize(diff1_abs)
        q2 = minMaxStandarize(diff2_abs)

        diffs_df = pd.DataFrame({"q1": q1, "q2": q2}, index=xVal_temp.index)

        # ---- EA selection (2-objective WITH  diversity) ----
        to_add = EA_select_batch_twoQuality_plus_diversity(
        trainingdata=xTrain_temp,
        candidates=xVal_temp,
        diffs_df=diffs_df,
        step_size=min(step_size, len(xVal_temp)),
        do_scale=True,          # or pass a parameter
        normalize=True,         # or pass a parameter
        pop_size=pop_size,
        n_gen=n_gen,
        p_cx=p_cx,
        p_mut=p_mut,
        random_state=intRandomSeed,
        w_train=0.5,            # you can tune
        w_within=0.5            # you can tune 
        )


        if len(to_add) == 0:
            print("EA returned 0 points. Stopping.")
            break

        lstTrainIndices += to_add

        # stop if full
        #if len(lstTrainIndices) >= len(xTrainVal):
        #    lstTrainIndices = lstTrainIndices[: len(xTrainVal)]
        #    break

        # optional: save every iter (like your second code)
        df_model1.to_csv(strSaveDir + strSaveName + "_" + strModel1Name + str(intRandomSeed) + ".csv", index=False)
        df_model2.to_csv(strSaveDir + strSaveName + "_" + strModel2Name + str(intRandomSeed) + ".csv", index=False)
        pd.DataFrame(lstTrainIndices, columns=["trainIndices"]).to_csv(
            strSaveDir + strSaveName + str(intRandomSeed) + "_trainIndices.csv",
            index=False,
        )

    # final save
    df_model1.to_csv(strSaveDir + strSaveName + "_" + strModel1Name + str(intRandomSeed) + ".csv", index=False)
    df_model2.to_csv(strSaveDir + strSaveName + "_" + strModel2Name + str(intRandomSeed) + ".csv", index=False)
    pd.DataFrame(lstTrainIndices, columns=["trainIndices"]).to_csv(
        strSaveDir + strSaveName + str(intRandomSeed) + "_trainIndices.csv",
        index=False,
    )

    return 




import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize as sk_normalize

def EA_select_batch_quality_plus_diversity(
    trainingdata: pd.DataFrame,
    candidates: pd.DataFrame,
    q: pd.Series,                    # quality aligned to candidates.index (e.g., normalized abs QBC diff)
    step_size: int,
    do_scale: bool = True,
    normalize: bool = True,
    pop_size: int = 120,
    n_gen: int = 60,
    p_cx: float = 0.9,
    p_mut: float = 0.25,
    random_state: int = 0,
    w_train: float = 0.5,
    w_within: float = 0.5,
):
    """
    Objectives (maximize):
      1) quality  = mean(q[batch])
      2) diversity = w_train * div_train + w_within * div_within

    div_train(B)  = mean_{i in B}(1 - mean_{j in T} cos(x_i, x_j))
    div_within(B) = 1 - mean_{i!=k in B} cos(x_i, x_k)

    Returns: list of candidates.index (length <= step_size)
    """
    rng = np.random.default_rng(random_state)

    # ---------- geometry ----------
    X_train = trainingdata.values.astype(np.float32, copy=False)
    X_pool  = candidates.values.astype(np.float32, copy=False)

    if do_scale:
        scaler = StandardScaler()
        X_all0 = np.vstack([X_train, X_pool])
        scaler.fit(X_all0)
        X_train = scaler.transform(X_train).astype(np.float32, copy=False)
        X_pool  = scaler.transform(X_pool).astype(np.float32, copy=False)

    if normalize:
        X_train = sk_normalize(X_train, axis=1)
        X_pool  = sk_normalize(X_pool, axis=1)

    n_train = X_train.shape[0]
    n_pool  = X_pool.shape[0]
    B = int(min(step_size, n_pool))
    if B <= 0:
        return []

    # ---------- quality aligned ----------
    if isinstance(q, pd.Series):
        qv = q.reindex(candidates.index).values.astype(np.float32)
    else:
        qv = np.asarray(q, dtype=np.float32).reshape(-1)
    qv = np.nan_to_num(qv, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------- precompute diversity terms ----------
    if n_train > 0:
        sim_pool_train_mean = (X_pool @ X_train.T).mean(axis=1).astype(np.float32)  # (n_pool,)
    else:
        sim_pool_train_mean = np.zeros(n_pool, dtype=np.float32)

    # pool-to-pool cosine Gram (within-batch)
    G_pp = (X_pool @ X_pool.T).astype(np.float32, copy=False)

    def fitness(chrom):
        idx = np.asarray(chrom, dtype=int)

        # Obj1: quality
        qual = float(np.mean(qv[idx]))

        # div_train
        div_train = float(np.mean(1.0 - sim_pool_train_mean[idx]))

        # div_within
        if len(idx) <= 1:
            div_within = 1.0
        else:
            S = G_pp[np.ix_(idx, idx)]
            off = S[~np.eye(len(idx), dtype=bool)]
            div_within = float(1.0 - np.mean(off))

        div = float(w_train * div_train + w_within * div_within)
        return qual, div  # 2 objectives

    # =========================
    # NSGA-II loop (uses your existing helpers)
    # =========================
    pop = [rng.choice(n_pool, size=B, replace=False).astype(int) for _ in range(pop_size)]

    for _ in range(n_gen):
        F = np.array([fitness(ind) for ind in pop], dtype=float)  # (pop,2)
        fronts, rank = _fast_non_dominated_sort(F)
        crowd = {}
        for fr in fronts:
            crowd.update(_crowding_distance(F, fr))

        offspring = []
        pop_indices = np.arange(len(pop))

        while len(offspring) < pop_size:
            p1 = pop[_tournament_select(pop_indices, rank, crowd, rng)]
            p2 = pop[_tournament_select(pop_indices, rank, crowd, rng)]

            c1, c2 = p1.copy(), p2.copy()
            if rng.random() < p_cx:
                c1, c2 = _ordered_crossover(c1, c2, rng)

            c1 = _make_unique(c1, n_pool, rng)
            c2 = _make_unique(c2, n_pool, rng)

            c1 = _mutate_swap(c1, n_pool, rng, p_mut=p_mut)
            c2 = _mutate_swap(c2, n_pool, rng, p_mut=p_mut)

            c1 = _make_unique(c1, n_pool, rng)
            c2 = _make_unique(c2, n_pool, rng)

            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        combined = pop + offspring
        F2 = np.array([fitness(ind) for ind in combined], dtype=float)
        fronts2, _ = _fast_non_dominated_sort(F2)

        new_pop = []
        for fr in fronts2:
            if len(new_pop) + len(fr) <= pop_size:
                new_pop.extend([combined[i] for i in fr])
            else:
                cd = _crowding_distance(F2, fr)
                fr_sorted = sorted(fr, key=lambda i: cd.get(i, 0.0), reverse=True)
                needed = pop_size - len(new_pop)
                new_pop.extend([combined[i] for i in fr_sorted[:needed]])
                break
        pop = new_pop

    # ===== pick one solution from Pareto front =====
    F_final = np.array([fitness(ind) for ind in pop], dtype=float)
    fronts, _ = _fast_non_dominated_sort(F_final)
    front0 = fronts[0]
    f0 = F_final[front0]  # (k,2)

    # normalize objectives on front; pick max average
    fmin = f0.min(axis=0)
    fmax = f0.max(axis=0)
    denom = np.where((fmax - fmin) > 1e-12, (fmax - fmin), 1.0)
    f0n = (f0 - fmin) / denom
    score = f0n.mean(axis=1)
    best = front0[int(np.argmax(score))]

    best_local = pop[best]
    return candidates.index[np.asarray(best_local, dtype=int)].tolist()





import numpy as np
import pandas as pd

# ============================================================
# Single-objective EA (maximize: quality + lambda_div * diversity)
#   quality  = mean(q[idx])  where q is a single QBC disagreement vector
#   diversity = w_train*div_train + w_within*div_within
# ============================================================
def EA_select_batch_singleObjective_with_diversity(
    trainingdata: pd.DataFrame,
    candidates: pd.DataFrame,
    q: pd.Series,                    # single objective quality aligned to candidates.index
    step_size: int,
    do_scale: bool = True,
    normalize: bool = True,
    pop_size: int = 120,
    n_gen: int = 60,
    p_cx: float = 0.9,
    p_mut: float = 0.25,
    random_state: int = 0,
    # diversity mixing
    w_train: float = 0.5,
    w_within: float = 0.5,
    # combine into single objective
    lambda_div: float = 0.5,         # weight on diversity term
):
    """
    Returns: list of candidates.index of selected batch

    Maximize:
        J(B) = mean(q[B]) + lambda_div * (w_train*div_train(B) + w_within*div_within(B))

    div_train(B)  = mean_{i in B} (1 - mean_{j in T} cos(x_i, x_j))
    div_within(B) = 1 - mean_{i != k in B} cos(x_i, x_k)
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler, normalize as sk_normalize

    rng = np.random.default_rng(random_state)

    X_train = trainingdata.values.astype(np.float32, copy=False)
    X_pool  = candidates.values.astype(np.float32, copy=False)

    # consistent geometry
    if do_scale:
        scaler = StandardScaler()
        X_all0 = np.vstack([X_train, X_pool])
        scaler.fit(X_all0)
        X_train = scaler.transform(X_train).astype(np.float32, copy=False)
        X_pool  = scaler.transform(X_pool).astype(np.float32, copy=False)

    if normalize:
        X_train = sk_normalize(X_train, axis=1)
        X_pool  = sk_normalize(X_pool, axis=1)

    n_train = X_train.shape[0]
    n_pool  = X_pool.shape[0]
    B = int(min(step_size, n_pool))
    if B <= 0:
        return []

    # quality aligned
    if isinstance(q, pd.Series):
        qv = q.reindex(candidates.index).values.astype(np.float32)
    else:
        qv = np.asarray(q, dtype=np.float32).reshape(-1)
    qv = np.nan_to_num(qv, nan=0.0, posinf=0.0, neginf=0.0)

    # precompute similarities for diversity
    if n_train > 0:
        sim_pool_train_mean = (X_pool @ X_train.T).mean(axis=1).astype(np.float32)  # (n_pool,)
    else:
        sim_pool_train_mean = np.zeros(n_pool, dtype=np.float32)

    G_pp = (X_pool @ X_pool.T).astype(np.float32, copy=False)  # (n_pool,n_pool)

    def diversity(idx):
        # div_train
        div_train = float(np.mean(1.0 - sim_pool_train_mean[idx]))

        # div_within
        if len(idx) <= 1:
            div_within = 1.0
        else:
            S = G_pp[np.ix_(idx, idx)]
            off = S[~np.eye(len(idx), dtype=bool)]
            div_within = float(1.0 - np.mean(off))

        return w_train * div_train + w_within * div_within

    def objective(chrom):
        idx = np.asarray(chrom, dtype=int)
        qual = float(np.mean(qv[idx]))
        div  = diversity(idx)
        return qual + lambda_div * div

    # --- population init ---
    pop = [rng.choice(n_pool, size=B, replace=False).astype(int) for _ in range(pop_size)]

    # --- simple GA loop (single objective) ---
    for _ in range(n_gen):
        # evaluate
        scores = np.array([objective(ind) for ind in pop], dtype=float)

        # elitism: keep top 20%
        elite_k = max(2, int(0.2 * pop_size))
        elite_idx = np.argsort(scores)[::-1][:elite_k]
        elites = [pop[i] for i in elite_idx]

        # offspring by tournament on scalar objective
        offspring = []
        while len(offspring) < (pop_size - elite_k):
            i1, i2 = rng.integers(0, pop_size), rng.integers(0, pop_size)
            p1 = pop[i1] if scores[i1] >= scores[i2] else pop[i2]

            j1, j2 = rng.integers(0, pop_size), rng.integers(0, pop_size)
            p2 = pop[j1] if scores[j1] >= scores[j2] else pop[j2]

            c1, c2 = p1.copy(), p2.copy()
            if rng.random() < p_cx:
                c1, c2 = _ordered_crossover(c1, c2, rng)

            c1 = _make_unique(c1, n_pool, rng)
            c2 = _make_unique(c2, n_pool, rng)

            c1 = _mutate_swap(c1, n_pool, rng, p_mut=p_mut)
            c2 = _mutate_swap(c2, n_pool, rng, p_mut=p_mut)

            c1 = _make_unique(c1, n_pool, rng)
            c2 = _make_unique(c2, n_pool, rng)

            offspring.append(c1)
            if len(offspring) < (pop_size - elite_k):
                offspring.append(c2)

        pop = elites + offspring[: (pop_size - elite_k)]

    # pick best
    final_scores = np.array([objective(ind) for ind in pop], dtype=float)
    best = int(np.argmax(final_scores))
    best_local = pop[best]
    return candidates.index[np.asarray(best_local, dtype=int)].tolist()


# ============================================================
# Grow loop (3 outcomes): bandgap, e_form, bulk_modulus
# Single objective + diversity
#   - choose ONE target to optimize each run: target in {"bandgap","e_form","bulk_modulus"}
#   - QBC quality = minmax( |pred1 - pred2| ) for that target on validation
#   - batch chosen by EA maximizing quality + lambda_div*diversity
# ============================================================
def grow_EA_QBC_1Obj_withDiversity_from3Outcomes(
    model1, model2,
    xTrainVal, y1TrainVal, y2TrainVal, y3TrainVal,
    xTest, y1Test, y2Test, y3Test,
    target: str,  # one of outcome_names
    outcome_names=("e_form", "bulk_modulus", "bandgap"),
    strSaveDir="",
    strSaveName="growing_EA_1obj_qbc_div",
    strModel1Name="model1",
    strModel2Name="model2",
    # loop control
    fltStepFrac=0.01,
    fltTrainFracStop=1.0,
    intRandomSeed=0,
    # geometry for diversity
    do_scale_global=True,
    normalize_global=True,
    # EA params
    pop_size=120,
    n_gen=60,
    p_cx=0.9,
    p_mut=0.25,
    # diversity mixing + weight in objective
    w_train=0.5,
    w_within=0.5,
    lambda_div=0.5,
    # rounding for logs
    n_round=6,
    n_train_ratio=6,
):
    """
    Requires your existing functions:
      - splitTrainVal(xTrainVal, yTrainVal_df, lstTrainIndices)
      - get_scores(model, X_train, y_train, X_test, y_test, X_val)
    """
    import time
    import numpy as np
    import pandas as pd
    import random
    from myfunc import get_scores  # <-- adjust to your file

    name1, name2, name3 = outcome_names
    valid = {name1, name2, name3}
    if target not in valid:
        raise ValueError(f"target must be in {valid}. Got: {target}")

    if strSaveDir != "" and strSaveDir[-1] != "/":
        strSaveDir += "/"

    #rng = np.random.default_rng(intRandomSeed)
    #np.random.seed(intRandomSeed)
    #random.seed(intRandomSeed)

    #init_size = max(1, int(len(xTrainVal) * fltStepFrac))
    #step_size = max(1, int(len(xTrainVal) * fltStepFrac))

    #lstTrainIndices = rng.choice(xTrainVal.index.to_numpy(), size=init_size, replace=False).tolist()


     # set the random state
    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    # initialize the size of the training set
    intTempTrainSize = int(len(xTrainVal)*fltStepFrac)
    # initialize the step size
    step_size = int(len(xTrainVal)*fltStepFrac)

    # initialize a list with the indices of the first randomly selected training set
    lstTrainIndices = np.random.choice(xTrainVal.index, intTempTrainSize, replace=False).tolist()


    # pack y's into df for splitTrainVal
    yTrainVal_df = pd.DataFrame({
        "y1TrainVal": y1TrainVal,
        "y2TrainVal": y2TrainVal,
        "y3TrainVal": y3TrainVal,
    })

    # IMPORTANT: your splitTrainVal returns these names
    outcome_to_traincol = {name1: "y1Train", name2: "y2Train", name3: "y3Train"}
    yTest_map = {name1: y1Test, name2: y2Test, name3: y3Test}
    outcome_list = [name1, name2, name3]

    df_model1 = pd.DataFrame()
    df_model2 = pd.DataFrame()

    while True:
        t0 = time.time()
        train_frac = len(lstTrainIndices) / len(xTrainVal)
        #if train_frac >= fltTrainFracStop:
        #    break

        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(xTrainVal, yTrainVal_df, lstTrainIndices)

        remaining_idx = xTrainVal.index.difference(pd.Index(lstTrainIndices))
        print("remaining:", len(remaining_idx))
        print("xVal_temp:", len(xVal_temp))
        print("missing remaining not in xVal_temp:", len(set(remaining_idx) - set(xVal_temp.index)))
        print("overlap train/val:", len(set(lstTrainIndices) & set(xVal_temp.index)))
        print("unique train indices:", len(set(lstTrainIndices)), "vs list length:", len(lstTrainIndices))


        print("\n------------------------------------------------------------")
        print("Training set size:", len(xTrain_temp))
        print("Training %:       ", "{:.3%}".format(len(xTrain_temp) / len(xTrainVal)), "\n")

       

        # train/predict for all outcomes (needed for logging + to compute QBC)
        pred_val_1, pred_val_2 = {}, {}
        metrics_1, metrics_2 = {}, {}

        for nm in outcome_list:
            ytr = yTrain_temp[outcome_to_traincol[nm]]
            yte = yTest_map[nm]

            pv1, mae1, rmse1, r21 = get_scores(model1, xTrain_temp, ytr, xTest, yte, xVal_temp)
            pv2, mae2, rmse2, r22 = get_scores(model2, xTrain_temp, ytr, xTest, yte, xVal_temp)

            pred_val_1[nm] = np.asarray(pv1)
            pred_val_2[nm] = np.asarray(pv2)

            metrics_1[nm] = (mae1, rmse1, r21)
            metrics_2[nm] = (mae2, rmse2, r22)

        # log rows
        row1 = {"train_ratio": round(len(xTrain_temp) / len(xTrainVal), n_train_ratio), "train_size": len(xTrain_temp)}
        row2 = {"train_ratio": round(len(xTrain_temp) / len(xTrainVal), n_train_ratio), "train_size": len(xTrain_temp)}

        for nm in outcome_list:
            mae1, rmse1, r21 = metrics_1[nm]
            mae2, rmse2, r22 = metrics_2[nm]
            row1[f"mae_{nm}"] = round(float(mae1), n_round)
            row1[f"rmse_{nm}"] = round(float(rmse1), n_round)
            row1[f"r2_{nm}"] = round(float(r21), n_round)
            row2[f"mae_{nm}"] = round(float(mae2), n_round)
            row2[f"rmse_{nm}"] = round(float(rmse2), n_round)
            row2[f"r2_{nm}"] = round(float(r22), n_round)

        df_model1 = pd.concat([df_model1, pd.DataFrame([row1])], ignore_index=True)
        df_model2 = pd.concat([df_model2, pd.DataFrame([row2])], ignore_index=True)

        # -------- single-target QBC quality --------
        diff_abs = np.abs(pred_val_1[target] - pred_val_2[target])
        q = minMaxStandarize(diff_abs)
        q_series = pd.Series(q, index=xVal_temp.index)

        # -------- EA selection (single objective + diversity) --------
        remaining = len(xTrainVal) - len(lstTrainIndices)
        if remaining <= 0:
            break

        step_now = min(step_size, remaining, len(xVal_temp))
        if step_now <= 0:
            break

        to_add = EA_select_batch_singleObjective_with_diversity(
            trainingdata=xTrain_temp,
            candidates=xVal_temp,
            q=q_series,
            step_size=step_now,
            do_scale=do_scale_global,
            normalize=normalize_global,
            pop_size=pop_size,
            n_gen=n_gen,
            p_cx=p_cx,
            p_mut=p_mut,
            random_state=intRandomSeed,
            w_train=w_train,
            w_within=w_within,
            lambda_div=lambda_div,
        )

        if len(to_add) == 0:
            print("EA returned 0 points. Stopping.")
            break

        lstTrainIndices += to_add
        lstTrainIndices = list(dict.fromkeys(lstTrainIndices))



        # stop if full
        # if len(lstTrainIndices) >= len(xTrainVal):
        #    lstTrainIndices = lstTrainIndices[: len(xTrainVal)]
        #    break

        # optional: save each iter
        df_model1.to_csv(strSaveDir + strSaveName + "_" + strModel1Name + str(intRandomSeed) + ".csv", index=False)
        df_model2.to_csv(strSaveDir + strSaveName + "_" + strModel2Name + str(intRandomSeed) + ".csv", index=False)
        pd.DataFrame(lstTrainIndices, columns=["trainIndices"]).to_csv(
            strSaveDir + strSaveName + str(intRandomSeed) + "_trainIndices.csv",
            index=False,
        )

        print("Time to run loop:", round(time.time() - t0, 2), "seconds", flush=True)

    # final save
    df_model1.to_csv(strSaveDir + strSaveName + "_" + strModel1Name + str(intRandomSeed) + ".csv", index=False)
    df_model2.to_csv(strSaveDir + strSaveName + "_" + strModel2Name + str(intRandomSeed) + ".csv", index=False)
    pd.DataFrame(lstTrainIndices, columns=["trainIndices"]).to_csv(
        strSaveDir + strSaveName + str(intRandomSeed) + "_trainIndices.csv",
        index=False,
    )

    return 





def grow2QBC(model1, model2,
                           xTrainVal, y1TrainVal,
                           y2TrainVal,y3TrainVal,
                           xTest, y1Test,
                           y2Test,y3Test,
                           target1,
                           target2,
                           strSaveDir = '',
                           strSaveName = 'grow2QBC',
                           strModel1Name = 'model1',
                           strModel2Name = 'model2',
                           fltStepFrac = 0.01,
                           fltTrainFracStop = 1.0,
                           intRandomSeed = 0):
     # import dependencies
    import random
    import numpy as np
    import pandas as pd
    from myfunc import get_scores                                               # CHANGE DEPENDING ON KANGMING'S FILE NAME
    import time

    # check that the path given ends with a '/'
    if strSaveDir != '':
        if strSaveDir[-1] != '/':
            strSaveDir += '/'

    # initialize dataframe to hold results
    # will have 'train_ratio', 'maes', 'rmse','r2', 'train_size' columns
    dfgrow_MultiObj_Max_model1=pd.DataFrame()
    dfgrow_MultiObj_Max_model2=pd.DataFrame()

    # set the random state
    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    # intialize the size of the training set
    intTempTrainSize = int(len(xTrainVal)*fltStepFrac)
    # intialize the step size
    intStepSize = int(len(xTrainVal)*fltStepFrac)

    # initialize a list with the indices of the first randomly selected training set
    lstTrainIndices = np.random.choice(xTrainVal.index, intTempTrainSize, replace=False).tolist()

    while intTempTrainSize < len(xTrainVal)+1:
        # start time
        timeStart = time.time()
        # --- CHECK STOP CRITERIA ---
        # check if the training set has met the stop criteria
        #if intTempTrainSize >= len(xTrainVal)*fltTrainFracStop:
            # if so, break the loop
        #    break
        # --- SUBSAMPLE THE TRAININGVAL SET ---


        yTrainVal=pd.DataFrame({'y1TrainVal':y1TrainVal,'y2TrainVal':y2TrainVal,'y3TrainVal':y3TrainVal})

        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(xTrainVal, yTrainVal, lstTrainIndices)

        # --- CALCULATE, PRINT AND STORE RESULTS ---
        print('\n--------------------------------------------------------------------------------------------')
        print('Training set size: ', len(xTrain_temp))
        print('Training set percentage: ', '{:.6%}'.format(len(xTrain_temp)/len(xTrainVal)), '\n')


        y1Train_temp=yTrain_temp["y1Train"]
        y2Train_temp=yTrain_temp["y2Train"]
        y3Train_temp=yTrain_temp["y3Train"]

        # get scores using function for the random forest model

        print("------------------MODEL1 METRICS FOR EFORM------------------ ")
        eform_pred_val1,_1maes_1, _1rmse_1, _1r2_1 = get_scores(model1,
                                          xTrain_temp,y1Train_temp,
                                          xTest,y1Test,xVal_temp)
        print("------------------MODEL1 METRICS FOR BULK MODULUS ------------------")

        bulk_pred_val1,_2maes_1, _2rmse_1, _2r2_1 = get_scores(model1,
                                          xTrain_temp,y2Train_temp,
                                          xTest,y2Test,xVal_temp)
        print("------------------MODEL1 METRICS FOR BAND GAP------------------ ")

        bandgap_pred_val1,_3maes_1, _3rmse_1, _3r2_1 = get_scores(model1,
                                          xTrain_temp,y3Train_temp,
                                          xTest,y3Test,xVal_temp)
        


        print("------------------MODEL2 METRICS FOR EFORM------------------ ")
        eform_pred_val2,_1maes_2, _1rmse_2, _1r2_2 = get_scores(model2,
                                          xTrain_temp,y1Train_temp,
                                          xTest,y1Test,xVal_temp)
        print("------------------MODEL2 METRICS FOR BULK MODULUS ------------------")

        bulk_pred_val2,_2maes_2, _2rmse_2, _2r2_2 = get_scores(model2,
                                          xTrain_temp,y2Train_temp,
                                          xTest,y2Test,xVal_temp)
        print("------------------MODEL2 METRICS FOR BAND GAP------------------ ")

        bandgap_pred_val2,_3maes_2, _3rmse_2, _3r2_2 = get_scores(model2,
                                          xTrain_temp,y3Train_temp,
                                          xTest,y3Test,xVal_temp)
    

        # store the results for the random forest model predictions
        dfgrow_MultiObj_Max_model1 = pd.concat([dfgrow_MultiObj_Max_model1,
                                                       pd.DataFrame({'train_ratio': [round(len(xTrain_temp)/len(xTrainVal), 6)],
                                                                     'maes_eform': [round(_1maes_1, 6)],
                                                                     'rmse_eform': [round(_1rmse_1, 6)],
                                                                     'r2_eform': [round(_1r2_1, 6)],
                                                                     'maes_bulk_modulus':[round(_2maes_1, 6)],
                                                                     'rmse_bulk_modulus':[round(_2rmse_1, 6)],
                                                                     'r2_bulk_modulus':[round(_2r2_1, 6)],
                                                                     'maes_bandgap':[round(_3maes_1, 6)] ,
                                                                     'rmse_bandgap':[round(_3rmse_1, 6)] ,
                                                                     'r2_bandgap':[round(_3r2_1, 6)] ,
                                                                     'train_size': [len(xTrain_temp)]})], 
                                                      ignore_index=True)
        # store the results for the extra model predictions
        dfgrow_MultiObj_Max_model2 = pd.concat([dfgrow_MultiObj_Max_model2,
                                                       pd.DataFrame({'train_ratio': [round(len(xTrain_temp)/len(xTrainVal), 3)],
                                                                     'maes_eform': [round(_1maes_2, 4)],
                                                                     'rmse_eform': [round(_1rmse_2, 4)],
                                                                     'r2_eform': [round(_1r2_2, 4)],
                                                                     'maes_bulk_modulus':[round(_2maes_2, 4)],
                                                                     'rmse_bulk_modulus':[round(_2rmse_2, 4)],
                                                                     'r2_bulk_modulus':[round(_2r2_2, 4)],
                                                                     'maes_bandgap':[round(_3maes_2, 4)] ,
                                                                     'rmse_bandgap':[round(_3rmse_2, 4)] ,
                                                                     'r2_bandgap':[round(_3r2_2, 4)] ,
                                                                     'train_size': [len(xTrain_temp)]})], 
                                                      ignore_index=True)
         # --- SAVE THE RESULTS ---
        # save the results to a csv file
        dfgrow_MultiObj_Max_model1.to_csv(strSaveDir + strSaveName + '_' + strModel1Name + '.csv')
        dfgrow_MultiObj_Max_model2.to_csv(strSaveDir + strSaveName + '_' + strModel2Name + '.csv')       
        # make a dataframe with the indices of the training set
        dfTrainIndices = pd.DataFrame(lstTrainIndices, columns=['trainIndices'])
        # save the indices of the training set to a csv file
        dfTrainIndices.to_csv(strSaveDir + strSaveName + '_trainIndices.csv')
         
        # --- GROWING CRITERIA ---
        # check if the training set is the full training+validation set
        if intTempTrainSize == len(xTrainVal):
            # if so, break the loop
            intTempTrainSize+=10
            break

    
        



        diff_1=abs(eform_pred_val1-eform_pred_val2)
        diff_2=abs(bulk_pred_val1-bulk_pred_val2)
        diff_3=abs(bandgap_pred_val1-bandgap_pred_val2)


        diff1_standarized= minMaxStandarize(diff_1)
        diff2_standarized=minMaxStandarize(diff_2)
        diff3_standarized=minMaxStandarize(diff_3)

        
        

    
        #y1=e_form 
        #y2=bulk_modulus
        #y3=bandgap


    # ------------------------ targets eform and bulk_modulus ----------------------------------------------


        if (target1=='e_form' and target2=='bulk_modulus') or (target1=='bulk_modulus' and target2=='e_form') :
            # get the difference between the two models' predictions
            yVal_temp_diff = diff1_standarized+diff2_standarized
            yVal_temp_diff = pd.Series(yVal_temp_diff, index = yVal_temp.index) 



    # ------------------------ targets bandgap and eform ----------------------------------------------
        if (target1=='bandgap' and target2=='e_form') or (target1=='e_form' and target2=='bandgap') :
            yVal_temp_diff = diff1_standarized+diff3_standarized
            yVal_temp_diff = pd.Series(yVal_temp_diff, index = yVal_temp.index)
            




    # ------------------------ targets bandgap and bulkmodulus----------------------------------------------
        if (target1=='bandgap' and target2=='bulk_modulus') or (target1=='bulk_modulus' and target2=='bandgap') :
            yVal_temp_diff = diff3_standarized+diff2_standarized
            yVal_temp_diff = pd.Series(yVal_temp_diff, index = yVal_temp.index)
        

        
        # get the indices of the intStepSize points with the highest difference
        lstMaxDiffIndices = yVal_temp_diff.nlargest(intStepSize).index.tolist()

        # --- UPDATE THE TRAINING SET ---
        # add the indices to the training set
        lstTrainIndices = lstTrainIndices + lstMaxDiffIndices
        # update the size of the training set
        intTempTrainSize = len(lstTrainIndices)

        print(intTempTrainSize)

        # end time
        timeEnd = time.time()

        # print the time it took to grow the training set
        print('Time to grow training set: ', round(timeEnd-timeStart, 2), ' seconds', flush=True)
        print('--------------------------------------------------------------------------------------------\n')
    
    dfgrow_MultiObj_Max_model1.to_csv(strSaveDir + strSaveName + '_' + strModel1Name + '.csv')
    dfgrow_MultiObj_Max_model2.to_csv(strSaveDir + strSaveName + '_' + strModel2Name + '.csv')
    # make a dataframe with the indices of the training set
    dfTrainIndices = pd.DataFrame(lstTrainIndices, columns=['trainIndices'])
    # save the indices of the training set to a csv file
    dfTrainIndices.to_csv(strSaveDir + strSaveName + '_trainIndices.csv')

###############
###############
###############
###############




def pam_kmedoids_indices(X, k, rng=None, max_iter=100):
    """
    PAM-style k-medoids.

    np.ndarray of selected indices with dtype=int.
    """

    import numpy as np

    if rng is None:
        rng = np.random.default_rng(0)

    X = np.asarray(X)
    n = X.shape[0]

    if k >= n:
        return np.arange(n, dtype=int)

    # Pairwise Euclidean distance matrix.
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

    # Initialize the first medoid near the centroid, like your greedy function.
    centroid = X.mean(axis=0, keepdims=True)
    d_to_centroid = np.linalg.norm(X - centroid, axis=1)
    first = int(np.argmin(d_to_centroid))

    medoids = [first]

    # Farthest-first initialization for the remaining medoids.
    min_dist = D[:, first]

    while len(medoids) < k:
        min_dist[medoids] = -np.inf
        next_idx = int(np.argmax(min_dist))
        medoids.append(next_idx)

        min_dist = np.minimum(min_dist, D[:, next_idx])

    medoids = np.asarray(medoids, dtype=int)

    def total_cost(current_medoids):
        return D[:, current_medoids].min(axis=1).sum()

    current_cost = total_cost(medoids)

    for _ in range(max_iter):
        best_cost = current_cost
        best_medoids = medoids.copy()

        non_medoids = np.setdiff1d(np.arange(n), medoids)

        for medoid_pos in range(k):
            for candidate in non_medoids:
                trial_medoids = medoids.copy()
                trial_medoids[medoid_pos] = candidate

                trial_cost = total_cost(trial_medoids)

                if trial_cost < best_cost:
                    best_cost = trial_cost
                    best_medoids = trial_medoids.copy()

        if best_cost < current_cost:
            medoids = best_medoids
            current_cost = best_cost
        else:
            break

    return np.asarray(medoids, dtype=int)


def select_batch_HV_topM_kmedoids_OneTarget(
    trainingdata,
    candidates,
    diffs_df,
    step_size,
    top_frac=0.10,
    do_scale=True,
    normalize_features=True,
    random_state=0,
    standardize_disagreement=True,
    ref_point=None,
):
    """RF/XGBoost disagreement top-M + k-medoids for one target."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import normalize as sk_normalize

    rng = np.random.default_rng(random_state)

    if len(candidates) == 0:
        return []

    n_pool = len(candidates)
    B = int(min(step_size, n_pool))

    if B <= 0:
        return []

    # Align disagreement values to candidate indices.
    diffs_df = diffs_df.reindex(candidates.index)

    if diffs_df.shape[1] < 1:
        raise ValueError("diffs_df must have at least 1 column: ['q1'].")

    Q = diffs_df.iloc[:, :1].values.astype(np.float64)
    Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)

    if standardize_disagreement:
        q_min = Q.min(axis=0, keepdims=True)
        q_max = Q.max(axis=0, keepdims=True)
        denom = np.where((q_max - q_min) > 1e-12, q_max - q_min, 1.0)
        Q = (Q - q_min) / denom

    if ref_point is None:
        ref = np.zeros(1, dtype=np.float64)
    else:
        ref = np.asarray(ref_point, dtype=np.float64)

    if ref.shape[0] != 1:
        raise ValueError("ref_point must have length 1 for q1.")

    improvement = np.maximum(Q - ref.reshape(1, -1), 0.0)
    hv_scores = improvement[:, 0]
    hv_scores = np.nan_to_num(
        hv_scores,
        nan=-np.inf,
        posinf=np.finfo(np.float64).max,
        neginf=-np.inf,
    )

    M_10pct = int(np.floor(top_frac * n_pool))

    if M_10pct < B:
        filtered_local_idx = np.arange(n_pool)
    else:
        M = max(B, M_10pct)
        filtered_local_idx = np.argsort(-hv_scores)[:M]

    filtered_candidates = candidates.iloc[filtered_local_idx]

    X_train_np = trainingdata.values.astype(np.float32, copy=False)
    X_pool_np = candidates.values.astype(np.float32, copy=False)

    if do_scale:
        scaler = StandardScaler()

        if len(trainingdata) > 0:
            X_all = np.vstack([X_train_np, X_pool_np])
        else:
            X_all = X_pool_np

        scaler.fit(X_all)
        X_pool_np = scaler.transform(X_pool_np).astype(np.float32, copy=False)

    if normalize_features:
        X_pool_np = sk_normalize(X_pool_np, axis=1)

    X_filtered = X_pool_np[filtered_local_idx]
    n_filtered = X_filtered.shape[0]

    if n_filtered <= B:
        return filtered_candidates.index.tolist()

    selected_filtered_positions = pam_kmedoids_indices(
        X_filtered,
        B,
        rng=rng,
    )

    selected_local_idx = filtered_local_idx[
        np.asarray(selected_filtered_positions, dtype=int)
    ]

    return candidates.index[selected_local_idx].tolist()


def grow_disagreementHV_topM_kmedoids_OneTarget(
    model1, model2,
    xTrainVal, y1TrainVal,
    y2TrainVal, y3TrainVal,
    xTest, y1Test,
    y2Test, y3Test,
    target1,
    strSaveDir='',
    strSaveName='growing_HV_topM_kmedoids_1target',
    strModel1Name='rf',
    strModel2Name='xgb',
    fltStepFrac=0.01,
    intRandomSeed=0,
    do_scale_global=True,
    normalize_global=True,
    top_frac=0.10,
):
    """
    DFT implementation following grow_QBC's data handling exactly.

    All three DFT targets are evaluated at each iteration. The target1 argument
    determines which RF/XGBoost disagreement is used for top-M filtering and
    feature-space k-medoids selection.
    """
    import random
    import time
    import numpy as np
    import pandas as pd
    from myfunc import get_scores

    valid_targets = ['e_form', 'bulk_modulus', 'bandgap']
    if target1 not in valid_targets:
        raise ValueError('target1 must be one of: ' + str(valid_targets))

    # Same path handling as grow_QBC.
    if strSaveDir != '':
        if strSaveDir[-1] != '/':
            strSaveDir += '/'

    # Same result dataframe initialization as grow_QBC.
    dfGrowing_model1 = pd.DataFrame()
    dfGrowing_model2 = pd.DataFrame()

    # Same random-state initialization as grow_QBC.
    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    # Same training and step sizes as grow_QBC.
    intTempTrainSize = int(len(xTrainVal) * fltStepFrac)
    intStepSize = int(len(xTrainVal) * fltStepFrac)

    # Avoid an empty initial set only when fltStepFrac is too small.
    if intTempTrainSize < 1:
        intTempTrainSize = 1
    if intStepSize < 1:
        intStepSize = 1

    # Same initial random selection as grow_QBC.
    lstTrainIndices = np.random.choice(
        xTrainVal.index,
        intTempTrainSize,
        replace=False,
    ).tolist()

    while intTempTrainSize < (len(xTrainVal) + 1):
        timeStart = time.time()

        # CRITICAL: use the exact column names expected by splitTrainVal.
        # This is identical to grow_QBC.
        yTrainVal_all = pd.DataFrame({
            'y1TrainVal': y1TrainVal,
            'y2TrainVal': y2TrainVal,
            'y3TrainVal': y3TrainVal,
        })

        # Same splitTrainVal call as grow_QBC.
        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(
            xTrainVal,
            yTrainVal_all,
            lstTrainIndices,
        )

        print('\n--------------------------------------------------------------------------------------------')
        print('Training set size: ', len(xTrain_temp))
        print(
            'Training set percentage: ',
            '{:.1%}'.format(len(xTrain_temp) / len(xTrainVal)),
            '\n',
        )

        # CRITICAL: use the exact output columns produced by splitTrainVal.
        # This is identical to grow_QBC.
        y1Train_temp = yTrain_temp['y1Train']
        y2Train_temp = yTrain_temp['y2Train']
        y3Train_temp = yTrain_temp['y3Train']

        # -------------------------
        # Model 1: evaluate all three DFT targets
        # -------------------------
        print('------------------MODEL1 METRICS FOR EFORM------------------ ')
        eform_pred_val1, _1maes_1, _1rmse_1, _1r2_1 = get_scores(
            model1,
            xTrain_temp, y1Train_temp,
            xTest, y1Test,
            X_val=xVal_temp,
        )

        print('------------------MODEL1 METRICS FOR BULK MODULUS ------------------')
        bulk_pred_val1, _2maes_1, _2rmse_1, _2r2_1 = get_scores(
            model1,
            xTrain_temp, y2Train_temp,
            xTest, y2Test,
            X_val=xVal_temp,
        )

        print('------------------MODEL1 METRICS FOR BAND GAP------------------ ')
        bandgap_pred_val1, _3maes_1, _3rmse_1, _3r2_1 = get_scores(
            model1,
            xTrain_temp, y3Train_temp,
            xTest, y3Test,
            X_val=xVal_temp,
        )

        # -------------------------
        # Model 2: evaluate all three DFT targets
        # -------------------------
        print('------------------MODEL2 METRICS FOR EFORM------------------ ')
        eform_pred_val2, _1maes_2, _1rmse_2, _1r2_2 = get_scores(
            model2,
            xTrain_temp, y1Train_temp,
            xTest, y1Test,
            X_val=xVal_temp,
        )

        print('------------------MODEL2 METRICS FOR BULK MODULUS ------------------')
        bulk_pred_val2, _2maes_2, _2rmse_2, _2r2_2 = get_scores(
            model2,
            xTrain_temp, y2Train_temp,
            xTest, y2Test,
            X_val=xVal_temp,
        )

        print('------------------MODEL2 METRICS FOR BAND GAP------------------ ')
        bandgap_pred_val2, _3maes_2, _3rmse_2, _3r2_2 = get_scores(
            model2,
            xTrain_temp, y3Train_temp,
            xTest, y3Test,
            X_val=xVal_temp,
        )

        # Store results using the same column names as grow_QBC.
        dfGrowing_model1 = pd.concat([
            dfGrowing_model1,
            pd.DataFrame({
                'train_ratio': [round(len(xTrain_temp) / len(xTrainVal), n_train_ratio)],
                'maes_eform': [round(_1maes_1, n_round)],
                'rmse_eform': [round(_1rmse_1, n_round)],
                'r2_eform': [round(_1r2_1, n_round)],
                'maes_bulk_modulus': [round(_2maes_1, n_round)],
                'rmse_bulk_modulus': [round(_2rmse_1, n_round)],
                'r2_bulk_modulus': [round(_2r2_1, n_round)],
                'maes_bandgap': [round(_3maes_1, n_round)],
                'rmse_bandgap': [round(_3rmse_1, n_round)],
                'r2_bandgap': [round(_3r2_1, n_round)],
                'train_size': [len(xTrain_temp)],
            }),
        ], ignore_index=True)

        dfGrowing_model2 = pd.concat([
            dfGrowing_model2,
            pd.DataFrame({
                'train_ratio': [round(len(xTrain_temp) / len(xTrainVal), n_train_ratio)],
                'maes_eform': [round(_1maes_2, n_round)],
                'rmse_eform': [round(_1rmse_2, n_round)],
                'r2_eform': [round(_1r2_2, n_round)],
                'maes_bulk_modulus': [round(_2maes_2, n_round)],
                'rmse_bulk_modulus': [round(_2rmse_2, n_round)],
                'r2_bulk_modulus': [round(_2r2_2, n_round)],
                'maes_bandgap': [round(_3maes_2, n_round)],
                'rmse_bandgap': [round(_3rmse_2, n_round)],
                'r2_bandgap': [round(_3r2_2, n_round)],
                'train_size': [len(xTrain_temp)],
            }),
        ], ignore_index=True)

        # Same completion check used by grow_QBC.
        if intTempTrainSize == len(xTrainVal):
            intTempTrainSize += 10
            break

        # Select the validation predictions for the requested DFT target.
        if target1 == 'e_form':
            xVal_temp_pred1 = eform_pred_val1
            xVal_temp_pred2 = eform_pred_val2

        if target1 == 'bulk_modulus':
            xVal_temp_pred1 = bulk_pred_val1
            xVal_temp_pred2 = bulk_pred_val2

        if target1 == 'bandgap':
            xVal_temp_pred1 = bandgap_pred_val1
            xVal_temp_pred2 = bandgap_pred_val2

        # Same disagreement construction as grow_QBC.
        yVal_temp_diff = abs(xVal_temp_pred1 - xVal_temp_pred2)
        yVal_temp_diff = pd.Series(yVal_temp_diff, index=yVal_temp.index)

        # Standardize disagreement exactly once before selection.
        d1s = minMaxStandarize(yVal_temp_diff.values)
        diffs_df = pd.DataFrame({'q1': d1s}, index=yVal_temp.index)

        # Apply the experimental-data top-M + k-medoids acquisition logic.
        to_add = select_batch_HV_topM_kmedoids_OneTarget(
            trainingdata=xTrain_temp,
            candidates=xVal_temp,
            diffs_df=diffs_df,
            step_size=min(intStepSize, len(xVal_temp)),
            top_frac=top_frac,
            do_scale=do_scale_global,
            normalize_features=normalize_global,
            random_state=intRandomSeed,
            standardize_disagreement=False,
            ref_point=[0.0],
        )

        if len(to_add) == 0:
            print('RF-disagreement top-M k-medoids returned 0 points. Stopping.')
            break

        # Same update pattern as grow_QBC.
        lstTrainIndices = lstTrainIndices + to_add
        intTempTrainSize = len(lstTrainIndices)

        timeEnd = time.time()
        print('Selected batch size: ', len(to_add))
        print('Time to run loop: ', round(timeEnd - timeStart, 2), ' seconds', flush=True)
        print('\n--------------------------------------------------------------------------------------------')

    # Save results with seed-aware filenames, as in the experimental version.
    dfGrowing_model1.to_csv(
        strSaveDir + strSaveName + '_' + strModel1Name + str(intRandomSeed) + '.csv',
        index=False,
    )
    dfGrowing_model2.to_csv(
        strSaveDir + strSaveName + '_' + strModel2Name + str(intRandomSeed) + '.csv',
        index=False,
    )
    pd.DataFrame(lstTrainIndices, columns=['trainIndices']).to_csv(
        strSaveDir + strSaveName + str(intRandomSeed) + '_trainIndices.csv',
        index=False,
    )

    return



def select_batch_HV_topM_kmedoids_TwoTargets(
    trainingdata,
    candidates,
    diffs_df,
    step_size,
    top_frac=0.10,
    do_scale=True,
    normalize_features=True,
    random_state=0,
    standardize_disagreement=True,
    ref_point=None,
):
    """
    RF/XGBoost disagreement hypervolume top-M + feature-space k-medoids
    acquisition for TWO DFT targets.

    Parameters
    ----------
    trainingdata : pd.DataFrame
        Current labeled/training feature set.
    candidates : pd.DataFrame
        Current unlabeled/validation candidate feature set.
    diffs_df : pd.DataFrame
        Two disagreement columns, q1 and q2, aligned to candidates.index.
    step_size : int
        Number of candidates to select.
    top_frac : float, default=0.10
        Fraction of the pool retained after ranking by disagreement HV.
    do_scale : bool, default=True
        Standardize feature columns before k-medoids.
    normalize_features : bool, default=True
        L2-normalize feature vectors after scaling.
    random_state : int, default=0
        Random state passed to the PAM helper.
    standardize_disagreement : bool, default=True
        If True, min-max standardize q1 and q2 here. In the grow function the
        disagreements are already standardized, so this is passed as False.
    ref_point : sequence of length 2 or None
        Hypervolume reference point. Default is [0, 0].

    Returns
    -------
    list
        Original dataframe indices of the selected candidates.
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import normalize as sk_normalize

    rng = np.random.default_rng(random_state)

    if len(candidates) == 0:
        return []

    n_pool = len(candidates)
    B = int(min(step_size, n_pool))

    if B <= 0:
        return []

    # Keep disagreement rows in exactly the same order as candidate rows.
    diffs_df = diffs_df.reindex(candidates.index)

    if diffs_df.shape[1] < 2:
        raise ValueError("diffs_df must have at least 2 columns: ['q1', 'q2'].")

    Q = diffs_df.iloc[:, :2].values.astype(np.float64)
    Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)

    # Optional local scaling. The grow function already calls minMaxStandarize
    # independently for the two target disagreements, so it disables this.
    if standardize_disagreement:
        q_min = Q.min(axis=0, keepdims=True)
        q_max = Q.max(axis=0, keepdims=True)
        denom = np.where((q_max - q_min) > 1e-12, q_max - q_min, 1.0)
        Q = (Q - q_min) / denom

    if ref_point is None:
        ref = np.zeros(2, dtype=np.float64)
    else:
        ref = np.asarray(ref_point, dtype=np.float64)

    if ref.shape[0] != 2:
        raise ValueError("ref_point must have length 2 for q1 and q2.")

    # Per-candidate 2D disagreement hypervolume:
    # HV(x) = max(q1-r1, 0) * max(q2-r2, 0)
    improvement = np.maximum(Q - ref.reshape(1, -1), 0.0)
    hv_scores = np.prod(improvement, axis=1)
    hv_scores = np.nan_to_num(
        hv_scores,
        nan=-np.inf,
        posinf=np.finfo(np.float64).max,
        neginf=-np.inf,
    )

    # Keep the top top_frac of candidates according to 2D disagreement HV.
    # As in the experimental implementation, if that subset would contain
    # fewer candidates than the requested batch, use the whole candidate pool.
    M_top = int(np.floor(top_frac * n_pool))

    if M_top < B:
        filtered_local_idx = np.arange(n_pool)
    else:
        M = max(B, M_top)
        filtered_local_idx = np.argsort(-hv_scores)[:M]

    filtered_candidates = candidates.iloc[filtered_local_idx]

    # Feature-space preprocessing follows the one-target DFT implementation.
    X_train_np = trainingdata.values.astype(np.float32, copy=False)
    X_pool_np = candidates.values.astype(np.float32, copy=False)

    if do_scale:
        scaler = StandardScaler()

        if len(trainingdata) > 0:
            X_all = np.vstack([X_train_np, X_pool_np])
        else:
            X_all = X_pool_np

        scaler.fit(X_all)
        X_pool_np = scaler.transform(X_pool_np).astype(np.float32, copy=False)

    if normalize_features:
        X_pool_np = sk_normalize(X_pool_np, axis=1)

    X_filtered = X_pool_np[filtered_local_idx]

    if X_filtered.shape[0] <= B:
        return filtered_candidates.index.tolist()

    selected_filtered_positions = pam_kmedoids_indices(
        X_filtered,
        B,
        rng=rng,
    )

    selected_local_idx = filtered_local_idx[
        np.asarray(selected_filtered_positions, dtype=int)
    ]

    return candidates.index[selected_local_idx].tolist()


def grow_disagreementHV_topM_kmedoids_TwoTargets(
    model1, model2,
    xTrainVal, y1TrainVal,
    y2TrainVal, y3TrainVal,
    xTest, y1Test,
    y2Test, y3Test,
    target1,
    target2,
    strSaveDir='',
    strSaveName='growing_HV_topM_kmedoids_2targets',
    strModel1Name='model1',
    strModel2Name='model2',
    fltStepFrac=0.01,
    fltTrainFracStop=1.0,   # kept to match the DFT two-target interface
    intRandomSeed=0,
    do_scale_global=True,
    normalize_global=True,
    top_frac=0.10,
):
    """
    DFT two-target RF/XGBoost disagreement-HV -> top-M -> k-medoids grow loop.

    The data handling follows the existing DFT two-target grow2QBC structure:
      * y1 = e_form
      * y2 = bulk_modulus
      * y3 = bandgap
      * all three properties are trained/evaluated at every AL iteration
      * target1 and target2 only determine the acquisition objectives
      * the same training indices are used for all three targets
      * CSV results and trainIndices are saved during the loop and again at end

    The acquisition follows the experimental two-target implementation:
      1) compute RF/XGBoost disagreement independently for target1 and target2
      2) min-max standardize each disagreement
      3) retain them as separate q1 and q2 objectives
      4) score each candidate by 2D hypervolume q1*q2 relative to [0, 0]
      5) retain the top-M fraction
      6) use feature-space k-medoids to choose the next batch
    """
    import random
    import time
    import numpy as np
    import pandas as pd
    from myfunc import get_scores

    valid_targets = ['e_form', 'bulk_modulus', 'bandgap']

    if target1 not in valid_targets or target2 not in valid_targets:
        raise ValueError(
            'target1 and target2 must each be one of: ' + str(valid_targets)
        )

    if target1 == target2:
        raise ValueError('target1 and target2 must be different targets.')

    # Same path handling used by the existing DFT grow functions.
    if strSaveDir != '':
        if strSaveDir[-1] != '/':
            strSaveDir += '/'

    # Same DFT result structure: one dataframe per predictive model.
    dfgrow_MultiObj_Max_model1 = pd.DataFrame()
    dfgrow_MultiObj_Max_model2 = pd.DataFrame()

    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    intTempTrainSize = int(len(xTrainVal) * fltStepFrac)
    intStepSize = int(len(xTrainVal) * fltStepFrac)

    # Protect against a zero-sized initial set or AL increment.
    if intTempTrainSize < 1:
        intTempTrainSize = 1
    if intStepSize < 1:
        intStepSize = 1

    lstTrainIndices = np.random.choice(
        xTrainVal.index,
        intTempTrainSize,
        replace=False,
    ).tolist()

    while intTempTrainSize < len(xTrainVal) + 1:
        timeStart = time.time()

        # DFT data handling: build the exact dataframe expected by splitTrainVal.
        yTrainVal_all = pd.DataFrame({
            'y1TrainVal': y1TrainVal,
            'y2TrainVal': y2TrainVal,
            'y3TrainVal': y3TrainVal,
        })

        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(
            xTrainVal,
            yTrainVal_all,
            lstTrainIndices,
        )

        print('\n--------------------------------------------------------------------------------------------')
        print('Training set size: ', len(xTrain_temp))
        print(
            'Training set percentage: ',
            '{:.6%}'.format(len(xTrain_temp) / len(xTrainVal)),
            '\n',
        )

        # Exact splitTrainVal output names used by the DFT implementation.
        y1Train_temp = yTrain_temp['y1Train']
        y2Train_temp = yTrain_temp['y2Train']
        y3Train_temp = yTrain_temp['y3Train']

        # ------------------------------------------------------------------
        # MODEL 1: evaluate all three DFT properties
        # ------------------------------------------------------------------
        print('------------------MODEL1 METRICS FOR EFORM------------------ ')
        eform_pred_val1, _1maes_1, _1rmse_1, _1r2_1 = get_scores(
            model1,
            xTrain_temp, y1Train_temp,
            xTest, y1Test,
            X_val=xVal_temp,
        )

        print('------------------MODEL1 METRICS FOR BULK MODULUS ------------------')
        bulk_pred_val1, _2maes_1, _2rmse_1, _2r2_1 = get_scores(
            model1,
            xTrain_temp, y2Train_temp,
            xTest, y2Test,
            X_val=xVal_temp,
        )

        print('------------------MODEL1 METRICS FOR BAND GAP------------------ ')
        bandgap_pred_val1, _3maes_1, _3rmse_1, _3r2_1 = get_scores(
            model1,
            xTrain_temp, y3Train_temp,
            xTest, y3Test,
            X_val=xVal_temp,
        )

        # ------------------------------------------------------------------
        # MODEL 2: evaluate all three DFT properties
        # ------------------------------------------------------------------
        print('------------------MODEL2 METRICS FOR EFORM------------------ ')
        eform_pred_val2, _1maes_2, _1rmse_2, _1r2_2 = get_scores(
            model2,
            xTrain_temp, y1Train_temp,
            xTest, y1Test,
            X_val=xVal_temp,
        )

        print('------------------MODEL2 METRICS FOR BULK MODULUS ------------------')
        bulk_pred_val2, _2maes_2, _2rmse_2, _2r2_2 = get_scores(
            model2,
            xTrain_temp, y2Train_temp,
            xTest, y2Test,
            X_val=xVal_temp,
        )

        print('------------------MODEL2 METRICS FOR BAND GAP------------------ ')
        bandgap_pred_val2, _3maes_2, _3rmse_2, _3r2_2 = get_scores(
            model2,
            xTrain_temp, y3Train_temp,
            xTest, y3Test,
            X_val=xVal_temp,
        )

        # ------------------------------------------------------------------
        # Store metrics using the same DFT two-target output schema.
        # ------------------------------------------------------------------
        dfgrow_MultiObj_Max_model1 = pd.concat([
            dfgrow_MultiObj_Max_model1,
            pd.DataFrame({
                'train_ratio': [round(len(xTrain_temp) / len(xTrainVal), 6)],
                'maes_eform': [round(_1maes_1, 6)],
                'rmse_eform': [round(_1rmse_1, 6)],
                'r2_eform': [round(_1r2_1, 6)],
                'maes_bulk_modulus': [round(_2maes_1, 6)],
                'rmse_bulk_modulus': [round(_2rmse_1, 6)],
                'r2_bulk_modulus': [round(_2r2_1, 6)],
                'maes_bandgap': [round(_3maes_1, 6)],
                'rmse_bandgap': [round(_3rmse_1, 6)],
                'r2_bandgap': [round(_3r2_1, 6)],
                'train_size': [len(xTrain_temp)],
            }),
        ], ignore_index=True)

        dfgrow_MultiObj_Max_model2 = pd.concat([
            dfgrow_MultiObj_Max_model2,
            pd.DataFrame({
                'train_ratio': [round(len(xTrain_temp) / len(xTrainVal), 3)],
                'maes_eform': [round(_1maes_2, 4)],
                'rmse_eform': [round(_1rmse_2, 4)],
                'r2_eform': [round(_1r2_2, 4)],
                'maes_bulk_modulus': [round(_2maes_2, 4)],
                'rmse_bulk_modulus': [round(_2rmse_2, 4)],
                'r2_bulk_modulus': [round(_2r2_2, 4)],
                'maes_bandgap': [round(_3maes_2, 4)],
                'rmse_bandgap': [round(_3rmse_2, 4)],
                'r2_bandgap': [round(_3r2_2, 4)],
                'train_size': [len(xTrain_temp)],
            }),
        ], ignore_index=True)

        # ------------------------------------------------------------------
        # SAVE DURING EACH ITERATION -- same logic/naming as grow2QBC.
        # ------------------------------------------------------------------
        dfgrow_MultiObj_Max_model1.to_csv(
            strSaveDir + strSaveName + '_' + strModel1Name + '.csv'
        )
        dfgrow_MultiObj_Max_model2.to_csv(
            strSaveDir + strSaveName + '_' + strModel2Name + '.csv'
        )

        dfTrainIndices = pd.DataFrame(lstTrainIndices, columns=['trainIndices'])
        dfTrainIndices.to_csv(
            strSaveDir + strSaveName + '_trainIndices.csv'
        )

        # Stop after the full train+validation pool has been collected.
        if intTempTrainSize == len(xTrainVal):
            intTempTrainSize += 10
            break

        if len(xVal_temp) == 0:
            print('No validation candidates left. Stopping.')
            break

        # ------------------------------------------------------------------
        # Map each DFT target to the corresponding validation predictions.
        # ------------------------------------------------------------------
        pred_map_model1 = {
            'e_form': eform_pred_val1,
            'bulk_modulus': bulk_pred_val1,
            'bandgap': bandgap_pred_val1,
        }

        pred_map_model2 = {
            'e_form': eform_pred_val2,
            'bulk_modulus': bulk_pred_val2,
            'bandgap': bandgap_pred_val2,
        }

        # Two independent RF/XGB disagreements.
        diff_1 = np.abs(pred_map_model1[target1] - pred_map_model2[target1])
        diff_2 = np.abs(pred_map_model1[target2] - pred_map_model2[target2])

        # Standardize each target separately, exactly as required before
        # constructing the two-objective disagreement HV.
        diff1_standardized = minMaxStandarize(diff_1)
        diff2_standardized = minMaxStandarize(diff_2)

        diffs_df = pd.DataFrame(
            {
                'q1': diff1_standardized,
                'q2': diff2_standardized,
            },
            index=yVal_temp.index,
        )

        # Experimental two-target acquisition logic applied to DFT features.
        to_add = select_batch_HV_topM_kmedoids_TwoTargets(
            trainingdata=xTrain_temp,
            candidates=xVal_temp,
            diffs_df=diffs_df,
            step_size=min(intStepSize, len(xVal_temp)),
            top_frac=top_frac,
            do_scale=do_scale_global,
            normalize_features=normalize_global,
            random_state=intRandomSeed,
            standardize_disagreement=False,
            ref_point=[0.0, 0.0],
        )

        if len(to_add) == 0:
            print('RF-disagreement HV top-M k-medoids returned 0 points. Stopping.')
            break

        # Same DFT list-based training-set update.
        lstTrainIndices = lstTrainIndices + to_add
        intTempTrainSize = len(lstTrainIndices)

        print(intTempTrainSize)
        print('Selected batch size: ', len(to_add))

        timeEnd = time.time()
        print(
            'Time to grow training set: ',
            round(timeEnd - timeStart, 2),
            ' seconds',
            flush=True,
        )
        print('--------------------------------------------------------------------------------------------\n')

    # Final save, matching the existing DFT two-target saving structure.
    dfgrow_MultiObj_Max_model1.to_csv(
        strSaveDir + strSaveName + '_' + strModel1Name + '.csv'
    )
    dfgrow_MultiObj_Max_model2.to_csv(
        strSaveDir + strSaveName + '_' + strModel2Name + '.csv'
    )

    dfTrainIndices = pd.DataFrame(lstTrainIndices, columns=['trainIndices'])
    dfTrainIndices.to_csv(
        strSaveDir + strSaveName + '_trainIndices.csv'
    )

    return






def grow_TS_QBC_no_diversity_OneTarget(
    model1, model2,
    xTrainVal,
    y1TrainVal, y2TrainVal, y3TrainVal,
    xTest,
    y1Test, y2Test, y3Test,
    target1: str,
    strSaveDir='',
    strSaveName='growing_TS_QBC_no_diversity_1target',
    strModel1Name='rf',
    strModel2Name='xgb',
    fltStepFrac=0.01,
    intRandomSeed=0,
    tau=1.0,
):
    """
    DFT grow loop using:

        One-target QBC disagreement
        -> Thompson-sampled disagreement landscape
        -> top-k selection

    This is a TS-QBC baseline WITHOUT diversity for one DFT target.

    The three DFT targets are evaluated at every iteration:
      - e_form
      - bulk_modulus
      - bandgap

    Only target1 controls candidate acquisition.

    The implementation follows the same data layout as grow_QBC:
      1) Build a target dataframe with columns
         y1TrainVal, y2TrainVal, y3TrainVal.
      2) Call splitTrainVal(...).
      3) Read the split training targets from
         y1Train, y2Train, y3Train.
      4) Evaluate both models on all three DFT targets.
      5) Use the selected target disagreement for TS-QBC acquisition.

    Requires existing project-level objects:
      - splitTrainVal(...)
      - get_scores(...)
      - minMaxStandarize(...)
      - n_round
      - n_train_ratio
      - select_batch_TS_QBC_no_diversity(...)
    """

    import random
    import time
    import numpy as np
    import pandas as pd


    import numpy as np
    from myfunc import get_scores                                               # CHANGE DEPENDING ON KANGMING'S FILE NAME

    valid_targets = ['e_form', 'bulk_modulus', 'bandgap']

    if target1 not in valid_targets:
        raise ValueError(f"target1 must be one of: {valid_targets}")

    if strSaveDir != '':
        if strSaveDir[-1] != '/':
            strSaveDir += '/'

    # Initialize result dataframes exactly as in grow_QBC.
    dfGrowing_TS_QBC_model1 = pd.DataFrame()
    dfGrowing_TS_QBC_model2 = pd.DataFrame()

    # Reproducibility.
    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)
    rng = np.random.default_rng(intRandomSeed)

    # Initial and incremental training sizes.
    intTempTrainSize = max(1, int(len(xTrainVal) * fltStepFrac))
    intStepSize = max(1, int(len(xTrainVal) * fltStepFrac))

    # Initial random training subset.
    lstTrainIndices = np.random.choice(
        xTrainVal.index,
        intTempTrainSize,
        replace=False
    ).tolist()

    while intTempTrainSize < (len(xTrainVal) + 1):
        timeStart = time.time()

        # IMPORTANT: this matches grow_QBC and splitTrainVal exactly.
        yTrainVal_all = pd.DataFrame({
            'y1TrainVal': y1TrainVal,
            'y2TrainVal': y2TrainVal,
            'y3TrainVal': y3TrainVal,
        })

        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(
            xTrainVal,
            yTrainVal_all,
            lstTrainIndices
        )

        print('\n--------------------------------------------------------------------------------------------')
        print('Training set size: ', len(xTrain_temp))
        print(
            'Training set percentage: ',
            '{:.1%}'.format(len(xTrain_temp) / len(xTrainVal)),
            '\n'
        )

        # IMPORTANT: these names are produced by your existing splitTrainVal.
        y1Train_temp = yTrain_temp['y1Train']
        y2Train_temp = yTrain_temp['y2Train']
        y3Train_temp = yTrain_temp['y3Train']

        # ------------------------------------------------------------------
        # Model 1: evaluate all three DFT targets and predict validation pool
        # ------------------------------------------------------------------
        print('------------------MODEL1 METRICS FOR EFORM------------------')
        eform_pred_val1, eform_mae_1, eform_rmse_1, eform_r2_1 = get_scores(
            model1,
            xTrain_temp, y1Train_temp,
            xTest, y1Test,
            X_val=xVal_temp
        )

        print('------------------MODEL1 METRICS FOR BULK MODULUS------------------')
        bulk_pred_val1, bulk_mae_1, bulk_rmse_1, bulk_r2_1 = get_scores(
            model1,
            xTrain_temp, y2Train_temp,
            xTest, y2Test,
            X_val=xVal_temp
        )

        print('------------------MODEL1 METRICS FOR BAND GAP------------------')
        bandgap_pred_val1, bandgap_mae_1, bandgap_rmse_1, bandgap_r2_1 = get_scores(
            model1,
            xTrain_temp, y3Train_temp,
            xTest, y3Test,
            X_val=xVal_temp
        )

        # ------------------------------------------------------------------
        # Model 2: evaluate all three DFT targets and predict validation pool
        # ------------------------------------------------------------------
        print('------------------MODEL2 METRICS FOR EFORM------------------')
        eform_pred_val2, eform_mae_2, eform_rmse_2, eform_r2_2 = get_scores(
            model2,
            xTrain_temp, y1Train_temp,
            xTest, y1Test,
            X_val=xVal_temp
        )

        print('------------------MODEL2 METRICS FOR BULK MODULUS------------------')
        bulk_pred_val2, bulk_mae_2, bulk_rmse_2, bulk_r2_2 = get_scores(
            model2,
            xTrain_temp, y2Train_temp,
            xTest, y2Test,
            X_val=xVal_temp
        )

        print('------------------MODEL2 METRICS FOR BAND GAP------------------')
        bandgap_pred_val2, bandgap_mae_2, bandgap_rmse_2, bandgap_r2_2 = get_scores(
            model2,
            xTrain_temp, y3Train_temp,
            xTest, y3Test,
            X_val=xVal_temp
        )

        # -------------------------
        # Store model 1 results
        # -------------------------
        dfGrowing_TS_QBC_model1 = pd.concat([
            dfGrowing_TS_QBC_model1,
            pd.DataFrame({
                'train_ratio': [round(len(xTrain_temp) / len(xTrainVal), n_train_ratio)],
                'maes_eform': [round(eform_mae_1, n_round)],
                'rmse_eform': [round(eform_rmse_1, n_round)],
                'r2_eform': [round(eform_r2_1, n_round)],
                'maes_bulk_modulus': [round(bulk_mae_1, n_round)],
                'rmse_bulk_modulus': [round(bulk_rmse_1, n_round)],
                'r2_bulk_modulus': [round(bulk_r2_1, n_round)],
                'maes_bandgap': [round(bandgap_mae_1, n_round)],
                'rmse_bandgap': [round(bandgap_rmse_1, n_round)],
                'r2_bandgap': [round(bandgap_r2_1, n_round)],
                'train_size': [len(xTrain_temp)],
            })
        ], ignore_index=True)

        # -------------------------
        # Store model 2 results
        # -------------------------
        dfGrowing_TS_QBC_model2 = pd.concat([
            dfGrowing_TS_QBC_model2,
            pd.DataFrame({
                'train_ratio': [round(len(xTrain_temp) / len(xTrainVal), n_train_ratio)],
                'maes_eform': [round(eform_mae_2, n_round)],
                'rmse_eform': [round(eform_rmse_2, n_round)],
                'r2_eform': [round(eform_r2_2, n_round)],
                'maes_bulk_modulus': [round(bulk_mae_2, n_round)],
                'rmse_bulk_modulus': [round(bulk_rmse_2, n_round)],
                'r2_bulk_modulus': [round(bulk_r2_2, n_round)],
                'maes_bandgap': [round(bandgap_mae_2, n_round)],
                'rmse_bandgap': [round(bandgap_rmse_2, n_round)],
                'r2_bandgap': [round(bandgap_r2_2, n_round)],
                'train_size': [len(xTrain_temp)],
            })
        ], ignore_index=True)

        # Stop after evaluating the full training+validation set.
        if intTempTrainSize == len(xTrainVal):
            intTempTrainSize += 10
            break

        if len(xVal_temp) == 0:
            print('No validation candidates left. Stopping.')
            break

        # -------------------------------------------------------
        # Select validation predictions for the acquisition target
        # -------------------------------------------------------
        if target1 == 'e_form':
            xVal_temp_pred1 = eform_pred_val1
            xVal_temp_pred2 = eform_pred_val2

        elif target1 == 'bulk_modulus':
            xVal_temp_pred1 = bulk_pred_val1
            xVal_temp_pred2 = bulk_pred_val2

        else:  # target1 == 'bandgap'
            xVal_temp_pred1 = bandgap_pred_val1
            xVal_temp_pred2 = bandgap_pred_val2

        # -------------------------
        # One-target QBC disagreement
        # -------------------------
        d1 = np.abs(
            np.asarray(xVal_temp_pred1, dtype=float)
            - np.asarray(xVal_temp_pred2, dtype=float)
        )

        # Standardize disagreement exactly as in the experimental version.
        d1s = minMaxStandarize(d1)
        u = np.asarray(d1s, dtype=float).reshape(-1)

        # -------------------------
        # Thompson sample
        # -------------------------
        u_ts = rng.normal(
            loc=u,
            scale=tau * u + 1e-12
        )
        u_ts = np.maximum(u_ts, 0.0)

        # -------------------------
        # Top-k TS-QBC selection, no diversity
        # -------------------------
        to_add = select_batch_TS_QBC_no_diversity(
            trainingdata=xTrain_temp,
            candidates=xVal_temp,
            quality_scores=u_ts,
            step_size=min(intStepSize, len(xVal_temp)),
            tau=tau,
            random_state=intRandomSeed,
        )

        if len(to_add) == 0:
            print('TS-QBC no-diversity returned 0 points. Stopping.')
            break

        # Update the active-learning training set exactly as grow_QBC does.
        lstTrainIndices = lstTrainIndices + to_add
        intTempTrainSize = len(lstTrainIndices)

        timeEnd = time.time()
        print('Selected batch size: ', len(to_add))
        print('Selected target: ', target1)
        print('Time to run loop: ', round(timeEnd - timeStart, 2), ' seconds', flush=True)
        print('\n--------------------------------------------------------------------------------------------')

    # -------------------------
    # Save results
    # -------------------------
    dfGrowing_TS_QBC_model1.to_csv(
        strSaveDir + strSaveName + '_' + strModel1Name + str(intRandomSeed) + '.csv',
        index=False
    )

    dfGrowing_TS_QBC_model2.to_csv(
        strSaveDir + strSaveName + '_' + strModel2Name + str(intRandomSeed) + '.csv',
        index=False
    )

    pd.DataFrame(lstTrainIndices, columns=['trainIndices']).to_csv(
        strSaveDir + strSaveName + str(intRandomSeed) + '_trainIndices.csv',
        index=False
    )

    return dfGrowing_TS_QBC_model1, dfGrowing_TS_QBC_model2, lstTrainIndices





 
def select_batch_TS_QBC_no_diversity(
    trainingdata,
    candidates,
    quality_scores=None,
    diffs_df=None,
    step_size=1,
    tau=1.0,
    random_state=0,
    standardize_disagreement=True,
    eps=1e-10,
):
    """
    Select a batch using Thompson-sampled QBC disagreement WITHOUT diversity.

    Works for 1, 2, or 3 targets.

    Selection rule:
        1) Compute the disagreement landscape u_i.
        2) Thompson sample
               u_ts_i ~ Normal(u_i, tau^2 * u_i^2)
        3) Select the top-k candidates by u_ts_i.

    No DPP, k-medoids, or feature-space diversity is used.
    """
    rng = np.random.default_rng(random_state)

    if len(candidates) == 0:
        return []

    step_size = min(step_size, len(candidates))

    if quality_scores is None and diffs_df is None:
        raise ValueError("You must provide either quality_scores or diffs_df.")

    if quality_scores is not None and diffs_df is not None:
        raise ValueError("Provide only one of quality_scores or diffs_df, not both.")

    # Option A: build Thompson-sampled quality scores from disagreement columns.
    if diffs_df is not None:
        if not isinstance(diffs_df, pd.DataFrame):
            raise TypeError("diffs_df must be a pandas DataFrame.")

        if len(diffs_df) != len(candidates):
            raise ValueError(
                "diffs_df must have the same length as candidates. "
                f"Got {len(diffs_df)} rows and {len(candidates)} candidates."
            )

        if not diffs_df.index.equals(candidates.index):
            diffs_df = diffs_df.reindex(candidates.index)

        if diffs_df.isnull().any().any():
            raise ValueError(
                "diffs_df contains NaN values after reindexing. "
                "Check that diffs_df and candidates have matching indices."
            )

        disagreement_matrix = diffs_df.values.astype(float)

        if disagreement_matrix.ndim == 1:
            disagreement_matrix = disagreement_matrix.reshape(-1, 1)

        if disagreement_matrix.shape[1] < 1:
            raise ValueError("diffs_df must contain at least one disagreement column.")

        if disagreement_matrix.shape[1] > 3:
            raise ValueError(
                "This selector is intended for 1, 2, or 3 targets. "
                f"Got {disagreement_matrix.shape[1]} disagreement columns."
            )

        if standardize_disagreement:
            disagreement_matrix_std = np.zeros_like(disagreement_matrix, dtype=float)

            for j in range(disagreement_matrix.shape[1]):
                col = disagreement_matrix[:, j]
                cmin = np.min(col)
                cmax = np.max(col)
                disagreement_matrix_std[:, j] = (
                    (col - cmin) / (cmax - cmin + eps)
                )

            disagreement_matrix = disagreement_matrix_std

        # Average standardized disagreement across selected targets.
        u = np.mean(disagreement_matrix, axis=1)

        # Thompson sample from disagreement landscape.
        quality_scores = rng.normal(
            loc=u,
            scale=tau * u + eps,
        )
        quality_scores = np.maximum(quality_scores, 0.0)

    # Option B: use already computed Thompson/quality scores.
    else:
        quality_scores = np.asarray(quality_scores).reshape(-1)

        if len(quality_scores) != len(candidates):
            raise ValueError(
                "quality_scores must have the same length as candidates. "
                f"Got {len(quality_scores)} scores and {len(candidates)} candidates."
            )

        quality_scores = np.asarray(quality_scores, dtype=float)
        quality_scores = np.maximum(quality_scores, 0.0)

    # Select top-k by Thompson-sampled QBC score.
    selected_local_idx = np.argsort(quality_scores)[-step_size:]

    # Return selected candidates from highest to lowest sampled score.
    selected_local_idx = selected_local_idx[
        np.argsort(quality_scores[selected_local_idx])[::-1]
    ]

    return candidates.index[selected_local_idx].tolist()








def select_batch_DPP_TS_QBC(
    trainingdata,
    candidates,
    quality_scores=None,
    diffs_df=None,
    step_size=1,
    top_frac=0.10,
    do_scale=True,
    normalize_features=True,
    sigma=None,
    tau=1.0,
    random_state=0,
    standardize_disagreement=True,
    eps=1e-10,
):
    """
    Select a batch using DPP-TS-QBC.

    This function works for 1, 2, or 3 targets.

    You can call it in either of two ways:

    Option A: pass precomputed Thompson-sampled quality scores

        select_batch_DPP_TS_QBC(
            candidates=xVal_temp,
            quality_scores=u_ts,
            ...
        )

    Option B: pass target disagreement columns in diffs_df

        diffs_df = pd.DataFrame({
            "q1": d1,
            "q2": d2,
            "q3": d3,
        }, index=xVal_temp.index)

        select_batch_DPP_TS_QBC(
            candidates=xVal_temp,
            diffs_df=diffs_df,
            ...
        )

    If diffs_df is passed, this function:
      1) optionally standardizes each disagreement column,
      2) averages across q1, q2, q3, ...
      3) Thompson-samples the disagreement landscape:

             u_ts_i ~ Normal(u_i, tau^2 * u_i^2)

      4) uses u_ts_i as DPP quality.

    DPP kernel:

        L_ij = q_i S_ij q_j

    where:
        q_i = normalized Thompson-sampled disagreement
        S_ij = RBF similarity between candidate features

    Parameters
    ----------
    trainingdata : pd.DataFrame
        Current training features. Kept for API compatibility.
        Not directly used by this selector.

    candidates : pd.DataFrame
        Candidate pool features.

    quality_scores : array-like or None
        Precomputed 1D quality scores, usually Thompson-sampled disagreement.
        Length must equal len(candidates).

    diffs_df : pd.DataFrame or None
        DataFrame containing one or more disagreement columns.
        For 1 target: columns may be ["q1"].
        For 2 targets: columns may be ["q1", "q2"].
        For 3 targets: columns may be ["q1", "q2", "q3"].

    step_size : int
        Number of candidates to select.

    top_frac : float
        Fraction of candidates to keep before DPP selection.

    do_scale : bool
        Whether to standard-scale candidate features before computing similarity.

    normalize_features : bool
        Whether to min-max normalize candidate features after optional standard scaling.

    sigma : float or None
        RBF bandwidth. If None, uses median-distance heuristic.

    tau : float
        Thompson sampling noise scale.

    random_state : int
        Random seed.

    standardize_disagreement : bool
        Whether to min-max standardize each disagreement column in diffs_df.

    eps : float
        Numerical stability constant.

    Returns
    -------
    selected_indices : list
        Original index labels from candidates.
    """

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics.pairwise import euclidean_distances

    rng = np.random.default_rng(random_state)

    if len(candidates) == 0:
        return []

    step_size = min(step_size, len(candidates))

    # ------------------------------------------------------------
    # 1. Build or validate quality scores
    # ------------------------------------------------------------
    if quality_scores is None and diffs_df is None:
        raise ValueError("You must provide either quality_scores or diffs_df.")

    if quality_scores is not None and diffs_df is not None:
        raise ValueError("Provide only one of quality_scores or diffs_df, not both.")

    if diffs_df is not None:
        if not isinstance(diffs_df, pd.DataFrame):
            raise TypeError("diffs_df must be a pandas DataFrame.")

        if len(diffs_df) != len(candidates):
            raise ValueError(
                "diffs_df must have the same length as candidates. "
                f"Got {len(diffs_df)} rows and {len(candidates)} candidates."
            )

        if not diffs_df.index.equals(candidates.index):
            diffs_df = diffs_df.reindex(candidates.index)

        if diffs_df.isnull().any().any():
            raise ValueError(
                "diffs_df contains NaN values after reindexing. "
                "Check that diffs_df and candidates have matching indices."
            )

        disagreement_matrix = diffs_df.values.astype(float)

        if disagreement_matrix.ndim == 1:
            disagreement_matrix = disagreement_matrix.reshape(-1, 1)

        if disagreement_matrix.shape[1] < 1:
            raise ValueError("diffs_df must contain at least one disagreement column.")

        if disagreement_matrix.shape[1] > 3:
            raise ValueError(
                "This selector is intended for 1, 2, or 3 targets. "
                f"Got {disagreement_matrix.shape[1]} disagreement columns."
            )

        if standardize_disagreement:
            disagreement_matrix_std = np.zeros_like(disagreement_matrix, dtype=float)

            for j in range(disagreement_matrix.shape[1]):
                col = disagreement_matrix[:, j]
                cmin = np.min(col)
                cmax = np.max(col)
                disagreement_matrix_std[:, j] = (col - cmin) / (cmax - cmin + eps)

            disagreement_matrix = disagreement_matrix_std

        # Average disagreement across 1, 2, or 3 targets
        u = np.mean(disagreement_matrix, axis=1)

        # Thompson sample from disagreement landscape
        quality_scores = rng.normal(
            loc=u,
            scale=tau * u + eps
        )

        quality_scores = np.maximum(quality_scores, 0.0)

    else:
        quality_scores = np.asarray(quality_scores).reshape(-1)

        if len(quality_scores) != len(candidates):
            raise ValueError(
                "quality_scores must have the same length as candidates. "
                f"Got {len(quality_scores)} scores and {len(candidates)} candidates."
            )

        quality_scores = np.asarray(quality_scores, dtype=float)
        quality_scores = np.maximum(quality_scores, 0.0)

    # ------------------------------------------------------------
    # 2. Top-M filtering by DPP-TS-QBC quality
    # ------------------------------------------------------------
    top_m = max(step_size, int(np.ceil(len(candidates) * top_frac)))
    top_m = min(top_m, len(candidates))

    top_local_idx = np.argsort(quality_scores)[-top_m:]

    cand_top = candidates.iloc[top_local_idx].copy()
    q_top_raw = quality_scores[top_local_idx]

    # ------------------------------------------------------------
    # 3. Feature preprocessing for DPP similarity
    # ------------------------------------------------------------
    X_top = cand_top.values

    if do_scale:
        X_top = StandardScaler().fit_transform(X_top)

    if normalize_features:
        X_top = MinMaxScaler().fit_transform(X_top)

    # ------------------------------------------------------------
    # 4. RBF similarity matrix
    # ------------------------------------------------------------
    D2 = euclidean_distances(X_top, squared=True)

    if sigma is None:
        nonzero = D2[D2 > 0]

        if len(nonzero) == 0:
            sigma = 1.0
        else:
            sigma = np.sqrt(np.median(nonzero))

    if sigma is None or sigma <= 0:
        sigma = 1.0

    S = np.exp(-D2 / (2.0 * sigma ** 2))

    # ------------------------------------------------------------
    # 5. Normalize DPP quality scores
    # ------------------------------------------------------------
    q_min = np.min(q_top_raw)
    q_max = np.max(q_top_raw)

    q = (q_top_raw - q_min) / (q_max - q_min + eps)
    q = eps + q

    # ------------------------------------------------------------
    # 6. Quality-weighted DPP kernel
    # ------------------------------------------------------------
    L = q[:, None] * S * q[None, :]

    # ------------------------------------------------------------
    # 7. Greedy MAP approximation to k-DPP
    # ------------------------------------------------------------
    selected_local_top = greedy_dpp_map(
        L,
        k=step_size,
        eps=eps
    )

    selected_original_indices = cand_top.index[selected_local_top].tolist()

    return selected_original_indices


def greedy_dpp_map(L, k, eps=1e-10):
    """
    Greedy MAP approximation for k-DPP subset selection.

    The selected subset approximately maximizes:

        det(L_B)

    where L_B is the DPP kernel submatrix for the selected batch.

    This favors points that are both:
      1) high quality, through the diagonal/quality weights, and
      2) mutually diverse, through the determinant.

    Parameters
    ----------
    L : array-like, shape (n_candidates, n_candidates)
        DPP kernel matrix.

    k : int
        Number of points to select.

    eps : float
        Numerical stability constant.

    Returns
    -------
    selected : np.ndarray
        Local integer indices of selected points.
    """

    import numpy as np

    L = np.asarray(L, dtype=float)

    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L must be a square matrix.")

    n = L.shape[0]

    if n == 0:
        return np.array([], dtype=int)

    k = min(int(k), n)

    if k <= 0:
        return np.array([], dtype=int)

    # Symmetrize for numerical stability
    L = 0.5 * (L + L.T)

    # Add small diagonal jitter once
    L = L + eps * np.eye(n)

    selected = []
    remaining = list(range(n))

    for _ in range(k):
        best_item = None
        best_score = -np.inf

        for i in remaining:
            candidate = selected + [i]
            submatrix = L[np.ix_(candidate, candidate)]

            sign, logdet = np.linalg.slogdet(submatrix)

            if sign > 0 and np.isfinite(logdet) and logdet > best_score:
                best_score = logdet
                best_item = i

        if best_item is None:
            # Fallback: choose the remaining item with largest diagonal quality
            remaining_diag = np.diag(L)[remaining]
            best_item = remaining[int(np.argmax(remaining_diag))]

        selected.append(best_item)
        remaining.remove(best_item)

    return np.array(selected, dtype=int)


def grow_DPP_TS_QBC_OneTarget(
    model1, model2,
    xTrainVal,
    y1TrainVal, y2TrainVal, y3TrainVal,
    xTest,
    y1Test, y2Test, y3Test,
    target1: str,
    strSaveDir='',
    strSaveName='growing_DPP_TS_QBC_1target',
    strModel1Name='rf',
    strModel2Name='xgb',
    fltStepFrac=0.01,
    intRandomSeed=0,
    do_scale_global=True,
    normalize_global=True,
    top_frac=0.10,
    tau=1.0,
    dpp_sigma=None,
):
    """
    DFT grow loop using:

        One-target QBC disagreement
        -> Thompson-sampled disagreement landscape
        -> top-M filtering
        -> quality-weighted DPP selection

    This is a DPP-TS-QBC method with feature-space diversity for one DFT target.

    The three DFT targets are evaluated at every iteration:
      - e_form
      - bulk_modulus
      - bandgap

    Only target1 controls candidate acquisition.

    The implementation follows the same data layout as grow_QBC:
      1) Build a target dataframe with columns
         y1TrainVal, y2TrainVal, y3TrainVal.
      2) Call splitTrainVal(...).
      3) Read the split training targets from
         y1Train, y2Train, y3Train.
      4) Evaluate both models on all three DFT targets.
      5) Use the selected target disagreement for TS-QBC acquisition.

    Requires existing project-level objects:
      - splitTrainVal(...)
      - get_scores(...)
      - minMaxStandarize(...)
      - n_round
      - n_train_ratio
      - select_batch_DPP_TS_QBC(...)
      - greedy_dpp_map(...)
    """

    import random
    import time
    import numpy as np
    import pandas as pd

    valid_targets = ['e_form', 'bulk_modulus', 'bandgap']

    if target1 not in valid_targets:
        raise ValueError(f"target1 must be one of: {valid_targets}")

    if strSaveDir != '':
        if strSaveDir[-1] != '/':
            strSaveDir += '/'

    # Initialize result dataframes exactly as in grow_QBC.
    dfGrowing_DPP_TS_QBC_model1 = pd.DataFrame()
    dfGrowing_DPP_TS_QBC_model2 = pd.DataFrame()

    # Reproducibility.
    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)
    rng = np.random.default_rng(intRandomSeed)

    # Initial and incremental training sizes.
    intTempTrainSize = max(1, int(len(xTrainVal) * fltStepFrac))
    intStepSize = max(1, int(len(xTrainVal) * fltStepFrac))

    # Initial random training subset.
    lstTrainIndices = np.random.choice(
        xTrainVal.index,
        intTempTrainSize,
        replace=False
    ).tolist()

    while intTempTrainSize < (len(xTrainVal) + 1):
        timeStart = time.time()

        # IMPORTANT: this matches grow_QBC and splitTrainVal exactly.
        yTrainVal_all = pd.DataFrame({
            'y1TrainVal': y1TrainVal,
            'y2TrainVal': y2TrainVal,
            'y3TrainVal': y3TrainVal,
        })

        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(
            xTrainVal,
            yTrainVal_all,
            lstTrainIndices
        )

        print('\n--------------------------------------------------------------------------------------------')
        print('Training set size: ', len(xTrain_temp))
        print(
            'Training set percentage: ',
            '{:.1%}'.format(len(xTrain_temp) / len(xTrainVal)),
            '\n'
        )

        # IMPORTANT: these names are produced by your existing splitTrainVal.
        y1Train_temp = yTrain_temp['y1Train']
        y2Train_temp = yTrain_temp['y2Train']
        y3Train_temp = yTrain_temp['y3Train']

        # ------------------------------------------------------------------
        # Model 1: evaluate all three DFT targets and predict validation pool
        # ------------------------------------------------------------------
        print('------------------MODEL1 METRICS FOR EFORM------------------')
        eform_pred_val1, eform_mae_1, eform_rmse_1, eform_r2_1 = get_scores(
            model1,
            xTrain_temp, y1Train_temp,
            xTest, y1Test,
            X_val=xVal_temp
        )

        print('------------------MODEL1 METRICS FOR BULK MODULUS------------------')
        bulk_pred_val1, bulk_mae_1, bulk_rmse_1, bulk_r2_1 = get_scores(
            model1,
            xTrain_temp, y2Train_temp,
            xTest, y2Test,
            X_val=xVal_temp
        )

        print('------------------MODEL1 METRICS FOR BAND GAP------------------')
        bandgap_pred_val1, bandgap_mae_1, bandgap_rmse_1, bandgap_r2_1 = get_scores(
            model1,
            xTrain_temp, y3Train_temp,
            xTest, y3Test,
            X_val=xVal_temp
        )

        # ------------------------------------------------------------------
        # Model 2: evaluate all three DFT targets and predict validation pool
        # ------------------------------------------------------------------
        print('------------------MODEL2 METRICS FOR EFORM------------------')
        eform_pred_val2, eform_mae_2, eform_rmse_2, eform_r2_2 = get_scores(
            model2,
            xTrain_temp, y1Train_temp,
            xTest, y1Test,
            X_val=xVal_temp
        )

        print('------------------MODEL2 METRICS FOR BULK MODULUS------------------')
        bulk_pred_val2, bulk_mae_2, bulk_rmse_2, bulk_r2_2 = get_scores(
            model2,
            xTrain_temp, y2Train_temp,
            xTest, y2Test,
            X_val=xVal_temp
        )

        print('------------------MODEL2 METRICS FOR BAND GAP------------------')
        bandgap_pred_val2, bandgap_mae_2, bandgap_rmse_2, bandgap_r2_2 = get_scores(
            model2,
            xTrain_temp, y3Train_temp,
            xTest, y3Test,
            X_val=xVal_temp
        )

        # -------------------------
        # Store model 1 results
        # -------------------------
        dfGrowing_DPP_TS_QBC_model1 = pd.concat([
            dfGrowing_DPP_TS_QBC_model1,
            pd.DataFrame({
                'train_ratio': [round(len(xTrain_temp) / len(xTrainVal), n_train_ratio)],
                'maes_eform': [round(eform_mae_1, n_round)],
                'rmse_eform': [round(eform_rmse_1, n_round)],
                'r2_eform': [round(eform_r2_1, n_round)],
                'maes_bulk_modulus': [round(bulk_mae_1, n_round)],
                'rmse_bulk_modulus': [round(bulk_rmse_1, n_round)],
                'r2_bulk_modulus': [round(bulk_r2_1, n_round)],
                'maes_bandgap': [round(bandgap_mae_1, n_round)],
                'rmse_bandgap': [round(bandgap_rmse_1, n_round)],
                'r2_bandgap': [round(bandgap_r2_1, n_round)],
                'train_size': [len(xTrain_temp)],
            })
        ], ignore_index=True)

        # -------------------------
        # Store model 2 results
        # -------------------------
        dfGrowing_DPP_TS_QBC_model2 = pd.concat([
            dfGrowing_DPP_TS_QBC_model2,
            pd.DataFrame({
                'train_ratio': [round(len(xTrain_temp) / len(xTrainVal), n_train_ratio)],
                'maes_eform': [round(eform_mae_2, n_round)],
                'rmse_eform': [round(eform_rmse_2, n_round)],
                'r2_eform': [round(eform_r2_2, n_round)],
                'maes_bulk_modulus': [round(bulk_mae_2, n_round)],
                'rmse_bulk_modulus': [round(bulk_rmse_2, n_round)],
                'r2_bulk_modulus': [round(bulk_r2_2, n_round)],
                'maes_bandgap': [round(bandgap_mae_2, n_round)],
                'rmse_bandgap': [round(bandgap_rmse_2, n_round)],
                'r2_bandgap': [round(bandgap_r2_2, n_round)],
                'train_size': [len(xTrain_temp)],
            })
        ], ignore_index=True)

        # Stop after evaluating the full training+validation set.
        if intTempTrainSize == len(xTrainVal):
            intTempTrainSize += 10
            break

        if len(xVal_temp) == 0:
            print('No validation candidates left. Stopping.')
            break

        # -------------------------------------------------------
        # Select validation predictions for the acquisition target
        # -------------------------------------------------------
        if target1 == 'e_form':
            xVal_temp_pred1 = eform_pred_val1
            xVal_temp_pred2 = eform_pred_val2

        elif target1 == 'bulk_modulus':
            xVal_temp_pred1 = bulk_pred_val1
            xVal_temp_pred2 = bulk_pred_val2

        else:  # target1 == 'bandgap'
            xVal_temp_pred1 = bandgap_pred_val1
            xVal_temp_pred2 = bandgap_pred_val2

        # -------------------------
        # One-target QBC disagreement
        # -------------------------
        d1 = np.abs(
            np.asarray(xVal_temp_pred1, dtype=float)
            - np.asarray(xVal_temp_pred2, dtype=float)
        )

        # Standardize disagreement exactly as in the experimental version.
        d1s = minMaxStandarize(d1)
        u = np.asarray(d1s, dtype=float).reshape(-1)

        # -------------------------
        # Thompson sample
        # -------------------------
        u_ts = rng.normal(
            loc=u,
            scale=tau * u + 1e-12
        )
        u_ts = np.maximum(u_ts, 0.0)

        # -------------------------
        # DPP-TS-QBC selection
        # -------------------------
        to_add = select_batch_DPP_TS_QBC(
            trainingdata=xTrain_temp,
            candidates=xVal_temp,
            quality_scores=u_ts,
            step_size=min(intStepSize, len(xVal_temp)),
            top_frac=top_frac,
            do_scale=do_scale_global,
            normalize_features=normalize_global,
            sigma=dpp_sigma,
            tau=tau,
            random_state=intRandomSeed,
            standardize_disagreement=False,
        )

        if len(to_add) == 0:
            print('DPP-TS-QBC returned 0 points. Stopping.')
            break

        # Update the active-learning training set exactly as grow_QBC does.
        lstTrainIndices = lstTrainIndices + to_add
        intTempTrainSize = len(lstTrainIndices)

        timeEnd = time.time()
        print('Selected batch size: ', len(to_add))
        print('Selected target: ', target1)
        print('Time to run loop: ', round(timeEnd - timeStart, 2), ' seconds', flush=True)
        print('\n--------------------------------------------------------------------------------------------')

    # -------------------------
    # Save results
    # -------------------------
    dfGrowing_DPP_TS_QBC_model1.to_csv(
        strSaveDir + strSaveName + '_' + strModel1Name + str(intRandomSeed) + '.csv',
        index=False
    )

    dfGrowing_DPP_TS_QBC_model2.to_csv(
        strSaveDir + strSaveName + '_' + strModel2Name + str(intRandomSeed) + '.csv',
        index=False
    )

    pd.DataFrame(lstTrainIndices, columns=['trainIndices']).to_csv(
        strSaveDir + strSaveName + str(intRandomSeed) + '_trainIndices.csv',
        index=False
    )

    return dfGrowing_DPP_TS_QBC_model1, dfGrowing_DPP_TS_QBC_model2, lstTrainIndices





def grow_DPP_TS_QBC_TwoTargets(
    model1, model2,
    xTrainVal, y1TrainVal,
    y2TrainVal, y3TrainVal,
    xTest, y1Test,
    y2Test, y3Test,
    target1: str,
    target2: str,
    strSaveDir='',
    strSaveName='grow_DPP_TS_QBC_2targets',
    strModel1Name='rf',
    strModel2Name='xgb',
    fltStepFrac=0.01,
    intRandomSeed=0,
    do_scale_global=True,
    normalize_global=True,
    top_frac=0.10,
    tau=1.0,
    dpp_sigma=None,
):
    """
    DFT implementation of the experimental-data
    grow_DPP_TS_QBC_TwoTargets algorithm.

    DFT target mapping:
        y1 -> e_form
        y2 -> bulk_modulus
        y3 -> bandgap

    Acquisition algorithm is unchanged:

        Two-target QBC disagreement
        -> target-wise min-max standardization
        -> average standardized disagreement landscape
        -> Thompson sampling
        -> top-M filtering
        -> quality-weighted DPP diversity selection

    At each iteration:
      1) Train/evaluate model1 and model2 on all three DFT targets.
      2) Compute RF/XGB disagreement for target1 and target2.
      3) Min-max standardize each target disagreement independently.
      4) Average the two standardized disagreements:
             u_i = mean(d1_std_i, d2_std_i)
      5) Thompson sample:
             u_ts_i ~ Normal(u_i, tau^2 * u_i^2)
      6) Clip sampled quality scores to nonnegative values.
      7) Use u_ts_i as DPP quality.
      8) Use RBF feature similarity as DPP similarity.
      9) Select batch using greedy DPP MAP.

    Requires existing project functions/variables:
      - splitTrainVal(...)
      - get_scores(...)
      - minMaxStandarize(...)
      - n_round
      - n_train_ratio
    """

    import random
    import time
    import numpy as np
    import pandas as pd

    target_cols = [
        "e_form",
        "bulk_modulus",
        "bandgap"
    ]

    selected_targets = [target1, target2]

    for t in selected_targets:
        if t not in target_cols:
            raise ValueError(f"{t} must be one of: {target_cols}")

    if len(set(selected_targets)) != 2:
        raise ValueError("target1 and target2 must be two distinct targets.")

    if strSaveDir != '' and strSaveDir[-1] != '/':
        strSaveDir += '/'

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)
    rng = np.random.default_rng(intRandomSeed)

    init_size = max(1, int(len(xTrainVal) * fltStepFrac))
    step_size = max(1, int(len(xTrainVal) * fltStepFrac))

    lstTrainIndices = np.random.choice(
        xTrainVal.index,
        init_size,
        replace=False
    ).tolist()

    while True:
        t0 = time.time()

        # IMPORTANT:
        # This follows the DFT splitTrainVal interface exactly.
        yTrainVal_all = pd.DataFrame({
            "y1TrainVal": y1TrainVal,
            "y2TrainVal": y2TrainVal,
            "y3TrainVal": y3TrainVal
        })

        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(
            xTrainVal,
            yTrainVal_all,
            lstTrainIndices
        )

        print("\n------------------------------------------------------------")
        print("Training set size:", len(xTrain_temp))
        print(
            "Training %:       ",
            "{:.1%}".format(len(xTrain_temp) / len(xTrainVal)),
            "\n"
        )

        # -------------------------
        # Unpack DFT train targets
        # -------------------------
        y1Train_temp = yTrain_temp["y1Train"]
        y2Train_temp = yTrain_temp["y2Train"]
        y3Train_temp = yTrain_temp["y3Train"]

        # -------------------------
        # Model 1 scores and validation predictions
        # -------------------------
        print("------------------MODEL1 METRICS FOR EFORM------------------")
        eform_pred_val1, eform_mae_1, eform_rmse_1, eform_r2_1 = get_scores(
            model1,
            xTrain_temp, y1Train_temp,
            xTest, y1Test,
            X_val=xVal_temp
        )

        print("------------------MODEL1 METRICS FOR BULK MODULUS------------------")
        bulk_pred_val1, bulk_mae_1, bulk_rmse_1, bulk_r2_1 = get_scores(
            model1,
            xTrain_temp, y2Train_temp,
            xTest, y2Test,
            X_val=xVal_temp
        )

        print("------------------MODEL1 METRICS FOR BAND GAP------------------")
        bandgap_pred_val1, bandgap_mae_1, bandgap_rmse_1, bandgap_r2_1 = get_scores(
            model1,
            xTrain_temp, y3Train_temp,
            xTest, y3Test,
            X_val=xVal_temp
        )

        # -------------------------
        # Model 2 scores and validation predictions
        # -------------------------
        print("------------------MODEL2 METRICS FOR EFORM------------------")
        eform_pred_val2, eform_mae_2, eform_rmse_2, eform_r2_2 = get_scores(
            model2,
            xTrain_temp, y1Train_temp,
            xTest, y1Test,
            X_val=xVal_temp
        )

        print("------------------MODEL2 METRICS FOR BULK MODULUS------------------")
        bulk_pred_val2, bulk_mae_2, bulk_rmse_2, bulk_r2_2 = get_scores(
            model2,
            xTrain_temp, y2Train_temp,
            xTest, y2Test,
            X_val=xVal_temp
        )

        print("------------------MODEL2 METRICS FOR BAND GAP------------------")
        bandgap_pred_val2, bandgap_mae_2, bandgap_rmse_2, bandgap_r2_2 = get_scores(
            model2,
            xTrain_temp, y3Train_temp,
            xTest, y3Test,
            X_val=xVal_temp
        )

        # -------------------------
        # Store model 1 results
        # -------------------------
        df1 = pd.concat([
            df1,
            pd.DataFrame({
                "train_ratio": [
                    round(len(xTrain_temp) / len(xTrainVal), n_train_ratio)
                ],
                "maes_eform": [round(eform_mae_1, n_round)],
                "rmse_eform": [round(eform_rmse_1, n_round)],
                "r2_eform": [round(eform_r2_1, n_round)],
                "maes_bulk_modulus": [round(bulk_mae_1, n_round)],
                "rmse_bulk_modulus": [round(bulk_rmse_1, n_round)],
                "r2_bulk_modulus": [round(bulk_r2_1, n_round)],
                "maes_bandgap": [round(bandgap_mae_1, n_round)],
                "rmse_bandgap": [round(bandgap_rmse_1, n_round)],
                "r2_bandgap": [round(bandgap_r2_1, n_round)],
                "train_size": [len(xTrain_temp)]
            })
        ], ignore_index=True)

        # -------------------------
        # Store model 2 results
        # -------------------------
        df2 = pd.concat([
            df2,
            pd.DataFrame({
                "train_ratio": [
                    round(len(xTrain_temp) / len(xTrainVal), n_train_ratio)
                ],
                "maes_eform": [round(eform_mae_2, n_round)],
                "rmse_eform": [round(eform_rmse_2, n_round)],
                "r2_eform": [round(eform_r2_2, n_round)],
                "maes_bulk_modulus": [round(bulk_mae_2, n_round)],
                "rmse_bulk_modulus": [round(bulk_rmse_2, n_round)],
                "r2_bulk_modulus": [round(bulk_r2_2, n_round)],
                "maes_bandgap": [round(bandgap_mae_2, n_round)],
                "rmse_bandgap": [round(bandgap_rmse_2, n_round)],
                "r2_bandgap": [round(bandgap_r2_2, n_round)],
                "train_size": [len(xTrain_temp)]
            })
        ], ignore_index=True)

        if len(xVal_temp) == 0:
            print("No validation candidates left. Stopping.")
            break

        # -------------------------
        # Map validation predictions
        # -------------------------
        pred_map_1 = {
            "e_form": eform_pred_val1,
            "bulk_modulus": bulk_pred_val1,
            "bandgap": bandgap_pred_val1,
        }

        pred_map_2 = {
            "e_form": eform_pred_val2,
            "bulk_modulus": bulk_pred_val2,
            "bandgap": bandgap_pred_val2,
        }

        # -------------------------
        # Two-target QBC disagreement
        # EXACT same algorithm as experimental-data version
        # -------------------------
        d1 = np.abs(pred_map_1[target1] - pred_map_2[target1])
        d2 = np.abs(pred_map_1[target2] - pred_map_2[target2])

        # Standardize disagreements target-wise so one target does not dominate
        d1s = minMaxStandarize(d1)
        d2s = minMaxStandarize(d2)

        # Average standardized disagreement landscape
        u = np.mean(
            np.vstack([d1s, d2s]),
            axis=0
        )

        # -------------------------
        # Thompson sample from disagreement landscape
        # -------------------------
        u_ts = rng.normal(
            loc=u,
            scale=tau * u + 1e-12
        )

        # DPP quality weights must be nonnegative
        u_ts = np.maximum(u_ts, 0.0)

        # -------------------------
        # DPP-TS-QBC selection
        # -------------------------
        to_add = select_batch_DPP_TS_QBC(
            trainingdata=xTrain_temp,
            candidates=xVal_temp,
            quality_scores=u_ts,
            step_size=min(step_size, len(xVal_temp)),
            top_frac=top_frac,
            do_scale=do_scale_global,
            normalize_features=normalize_global,
            sigma=dpp_sigma,
            random_state=intRandomSeed,
        )

        if len(to_add) == 0:
            print("DPP-TS-QBC returned 0 points. Stopping.")
            break

        lstTrainIndices = lstTrainIndices + to_add

        print("Selected batch size:", len(to_add))
        print("Selected targets:", selected_targets)
        print(
            "Time to run loop:",
            round(time.time() - t0, 2),
            "seconds",
            flush=True
        )

    # -------------------------
    # Save
    # -------------------------
    df1.to_csv(
        strSaveDir + strSaveName + "_" + strModel1Name +
        str(intRandomSeed) + ".csv",
        index=False
    )

    df2.to_csv(
        strSaveDir + strSaveName + "_" + strModel2Name +
        str(intRandomSeed) + ".csv",
        index=False
    )

    pd.DataFrame(
        lstTrainIndices,
        columns=["trainIndices"]
    ).to_csv(
        strSaveDir + strSaveName +
        str(intRandomSeed) + "_trainIndices.csv",
        index=False
    )

    return df1, df2, lstTrainIndices



#################
#################
#################
#################
#################



def grow_TS_QBC_no_diversity_TwoTargets(
    model1, model2,
    xTrainVal, y1TrainVal,
    y2TrainVal, y3TrainVal,
    xTest, y1Test,
    y2Test, y3Test,
    target1,
    target2,
    strSaveDir='',
    strSaveName='growing_TS_QBC_no_diversity_2targets',
    strModel1Name='rf',
    strModel2Name='xgb',
    fltStepFrac=0.01,
    intRandomSeed=0,
    tau=1.0):
    """
    DFT two-target TS-QBC baseline WITHOUT diversity.

    DFT targets:
        y1 -> e_form
        y2 -> bulk_modulus
        y3 -> bandgap

    At every active-learning iteration:
        1) Train/evaluate model1 and model2 on ALL THREE DFT targets.
        2) Compute RF/XGBoost disagreement for target1 and target2 only.
        3) Min-max standardize each selected target disagreement separately.
        4) Average the two standardized disagreement vectors:

               u_i = mean(q1_i, q2_i)

        5) Thompson sample:

               u_ts_i ~ Normal(u_i, tau^2 * u_i^2)

        6) Select the top-k candidates by u_ts_i.

    No DPP, k-medoids, or feature-space diversity is used.

    Requires existing project functions/globals:
        - splitTrainVal(...)
        - get_scores(...)
        - minMaxStandarize(...)
        - n_round
        - n_train_ratio
    """
    import random
    import time
    from myfunc import get_scores

    valid_targets = ['e_form', 'bulk_modulus', 'bandgap']

    for t in [target1, target2]:
        if t not in valid_targets:
            raise ValueError(
                'target1 and target2 must be selected from: ' + str(valid_targets)
            )

    if target1 == target2:
        raise ValueError('target1 and target2 must be different targets.')

    # Same path handling used by the DFT grow functions.
    if strSaveDir != '':
        if strSaveDir[-1] != '/':
            strSaveDir += '/'

    dfGrowing_model1 = pd.DataFrame()
    dfGrowing_model2 = pd.DataFrame()

    # Reproducible initial set and Thompson samples.
    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)
    rng = np.random.default_rng(intRandomSeed)

    intTempTrainSize = int(len(xTrainVal) * fltStepFrac)
    intStepSize = int(len(xTrainVal) * fltStepFrac)

    if intTempTrainSize < 1:
        intTempTrainSize = 1
    if intStepSize < 1:
        intStepSize = 1

    lstTrainIndices = np.random.choice(
        xTrainVal.index,
        intTempTrainSize,
        replace=False,
    ).tolist()

    while intTempTrainSize < (len(xTrainVal) + 1):
        timeStart = time.time()

        # Preserve the exact DFT target-column structure expected by splitTrainVal.
        yTrainVal_all = pd.DataFrame({
            'y1TrainVal': y1TrainVal,
            'y2TrainVal': y2TrainVal,
            'y3TrainVal': y3TrainVal,
        })

        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(
            xTrainVal,
            yTrainVal_all,
            lstTrainIndices,
        )

        print('\n--------------------------------------------------------------------------------------------')
        print('Training set size: ', len(xTrain_temp))
        print(
            'Training set percentage: ',
            '{:.1%}'.format(len(xTrain_temp) / len(xTrainVal)),
            '\n',
        )

        # Exact output names produced by the DFT splitTrainVal function.
        y1Train_temp = yTrain_temp['y1Train']
        y2Train_temp = yTrain_temp['y2Train']
        y3Train_temp = yTrain_temp['y3Train']

        # ------------------------------------------------------------------
        # Model 1: evaluate all three DFT targets.
        # ------------------------------------------------------------------
        print('------------------MODEL1 METRICS FOR EFORM------------------ ')
        eform_pred_val1, _1maes_1, _1rmse_1, _1r2_1 = get_scores(
            model1,
            xTrain_temp, y1Train_temp,
            xTest, y1Test,
            X_val=xVal_temp,
        )

        print('------------------MODEL1 METRICS FOR BULK MODULUS ------------------')
        bulk_pred_val1, _2maes_1, _2rmse_1, _2r2_1 = get_scores(
            model1,
            xTrain_temp, y2Train_temp,
            xTest, y2Test,
            X_val=xVal_temp,
        )

        print('------------------MODEL1 METRICS FOR BAND GAP------------------ ')
        bandgap_pred_val1, _3maes_1, _3rmse_1, _3r2_1 = get_scores(
            model1,
            xTrain_temp, y3Train_temp,
            xTest, y3Test,
            X_val=xVal_temp,
        )

        # ------------------------------------------------------------------
        # Model 2: evaluate all three DFT targets.
        # ------------------------------------------------------------------
        print('------------------MODEL2 METRICS FOR EFORM------------------ ')
        eform_pred_val2, _1maes_2, _1rmse_2, _1r2_2 = get_scores(
            model2,
            xTrain_temp, y1Train_temp,
            xTest, y1Test,
            X_val=xVal_temp,
        )

        print('------------------MODEL2 METRICS FOR BULK MODULUS ------------------')
        bulk_pred_val2, _2maes_2, _2rmse_2, _2r2_2 = get_scores(
            model2,
            xTrain_temp, y2Train_temp,
            xTest, y2Test,
            X_val=xVal_temp,
        )

        print('------------------MODEL2 METRICS FOR BAND GAP------------------ ')
        bandgap_pred_val2, _3maes_2, _3rmse_2, _3r2_2 = get_scores(
            model2,
            xTrain_temp, y3Train_temp,
            xTest, y3Test,
            X_val=xVal_temp,
        )

        # ------------------------------------------------------------------
        # Store model-1 metrics using the same DFT column names.
        # ------------------------------------------------------------------
        dfGrowing_model1 = pd.concat([
            dfGrowing_model1,
            pd.DataFrame({
                'train_ratio': [round(len(xTrain_temp) / len(xTrainVal), n_train_ratio)],
                'maes_eform': [round(_1maes_1, n_round)],
                'rmse_eform': [round(_1rmse_1, n_round)],
                'r2_eform': [round(_1r2_1, n_round)],
                'maes_bulk_modulus': [round(_2maes_1, n_round)],
                'rmse_bulk_modulus': [round(_2rmse_1, n_round)],
                'r2_bulk_modulus': [round(_2r2_1, n_round)],
                'maes_bandgap': [round(_3maes_1, n_round)],
                'rmse_bandgap': [round(_3rmse_1, n_round)],
                'r2_bandgap': [round(_3r2_1, n_round)],
                'train_size': [len(xTrain_temp)],
            }),
        ], ignore_index=True)

        # ------------------------------------------------------------------
        # Store model-2 metrics using the same DFT column names.
        # ------------------------------------------------------------------
        dfGrowing_model2 = pd.concat([
            dfGrowing_model2,
            pd.DataFrame({
                'train_ratio': [round(len(xTrain_temp) / len(xTrainVal), n_train_ratio)],
                'maes_eform': [round(_1maes_2, n_round)],
                'rmse_eform': [round(_1rmse_2, n_round)],
                'r2_eform': [round(_1r2_2, n_round)],
                'maes_bulk_modulus': [round(_2maes_2, n_round)],
                'rmse_bulk_modulus': [round(_2rmse_2, n_round)],
                'r2_bulk_modulus': [round(_2r2_2, n_round)],
                'maes_bandgap': [round(_3maes_2, n_round)],
                'rmse_bandgap': [round(_3rmse_2, n_round)],
                'r2_bandgap': [round(_3r2_2, n_round)],
                'train_size': [len(xTrain_temp)],
            }),
        ], ignore_index=True)

        # --------------------------------------------------------------
        # Save after every iteration, following the DFT grow2QBC style.
        # --------------------------------------------------------------
        dfGrowing_model1.to_csv(
            strSaveDir + strSaveName + '_' + strModel1Name + str(intRandomSeed) + '.csv',
            index=False,
        )
        dfGrowing_model2.to_csv(
            strSaveDir + strSaveName + '_' + strModel2Name + str(intRandomSeed) + '.csv',
            index=False,
        )
        pd.DataFrame(lstTrainIndices, columns=['trainIndices']).to_csv(
            strSaveDir + strSaveName + str(intRandomSeed) + '_trainIndices.csv',
            index=False,
        )

        # Full pool reached.
        if intTempTrainSize == len(xTrainVal):
            intTempTrainSize += 10
            break

        if len(xVal_temp) == 0:
            print('No validation candidates left. Stopping.')
            break

        # ------------------------------------------------------------------
        # Map each DFT target to the corresponding model predictions.
        # ------------------------------------------------------------------
        pred_map_1 = {
            'e_form': eform_pred_val1,
            'bulk_modulus': bulk_pred_val1,
            'bandgap': bandgap_pred_val1,
        }
        pred_map_2 = {
            'e_form': eform_pred_val2,
            'bulk_modulus': bulk_pred_val2,
            'bandgap': bandgap_pred_val2,
        }

        # ------------------------------------------------------------------
        # Two-target QBC disagreement.
        # ------------------------------------------------------------------
        d1 = np.abs(pred_map_1[target1] - pred_map_2[target1])
        d2 = np.abs(pred_map_1[target2] - pred_map_2[target2])

        # Standardize independently so target magnitude cannot dominate.
        d1s = minMaxStandarize(d1)
        d2s = minMaxStandarize(d2)

        # Average the two standardized target disagreements.
        u = np.mean(
            np.vstack([d1s, d2s]),
            axis=0,
        )

        # ------------------------------------------------------------------
        # Thompson sampling from the disagreement landscape.
        # ------------------------------------------------------------------
        u_ts = rng.normal(
            loc=u,
            scale=tau * u + 1e-12,
        )
        u_ts = np.maximum(u_ts, 0.0)

        # ------------------------------------------------------------------
        # Select top-k Thompson-sampled candidates. No diversity term.
        # ------------------------------------------------------------------
        to_add = select_batch_TS_QBC_no_diversity(
            trainingdata=xTrain_temp,
            candidates=xVal_temp,
            quality_scores=u_ts,
            step_size=min(intStepSize, len(xVal_temp)),
            tau=tau,
            random_state=intRandomSeed,
        )

        if len(to_add) == 0:
            print('TS-QBC no-diversity returned 0 points. Stopping.')
            break

        lstTrainIndices = lstTrainIndices + to_add
        intTempTrainSize = len(lstTrainIndices)

        timeEnd = time.time()
        print('Selected batch size: ', len(to_add))
        print('Selected targets: ', [target1, target2])
        print('Time to run loop: ', round(timeEnd - timeStart, 2), ' seconds', flush=True)
        print('\n--------------------------------------------------------------------------------------------')

    # Final save.
    dfGrowing_model1.to_csv(
        strSaveDir + strSaveName + '_' + strModel1Name + str(intRandomSeed) + '.csv',
        index=False,
    )
    dfGrowing_model2.to_csv(
        strSaveDir + strSaveName + '_' + strModel2Name + str(intRandomSeed) + '.csv',
        index=False,
    )
    pd.DataFrame(lstTrainIndices, columns=['trainIndices']).to_csv(
        strSaveDir + strSaveName + str(intRandomSeed) + '_trainIndices.csv',
        index=False,
    )

    return dfGrowing_model1, dfGrowing_model2, lstTrainIndices


####################
####################
####################
####################
####################
####################


def select_batch_HV_no_diversity_TwoTargets(
    trainingdata,
    candidates,
    diffs_df,
    step_size=1,
    top_frac=0.10,
    do_scale=True,
    normalize_features=True,
    random_state=0,
    standardize_disagreement=False,
    ref_point=None,
    eps=1e-10,
):
    """Two-target RF/XGBoost disagreement hypervolume WITHOUT diversity."""
    import numpy as np
    import pandas as pd

    if len(candidates) == 0:
        return []

    step_size = min(int(step_size), len(candidates))
    if step_size <= 0:
        return []

    if not isinstance(diffs_df, pd.DataFrame):
        raise TypeError("diffs_df must be a pandas DataFrame.")

    required_cols = ["q1", "q2"]
    for col in required_cols:
        if col not in diffs_df.columns:
            raise ValueError(f"diffs_df must contain column '{col}'.")

    if len(diffs_df) != len(candidates):
        raise ValueError(
            "diffs_df must have the same length as candidates. "
            f"Got {len(diffs_df)} rows and {len(candidates)} candidates."
        )

    if not diffs_df.index.equals(candidates.index):
        diffs_df = diffs_df.reindex(candidates.index)

    if diffs_df[required_cols].isnull().any().any():
        raise ValueError(
            "diffs_df contains NaN values after reindexing. "
            "Check that diffs_df and candidates have matching indices."
        )

    Q = diffs_df[required_cols].to_numpy(dtype=float)
    Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)

    if standardize_disagreement:
        Q_std = np.zeros_like(Q, dtype=float)
        for j in range(Q.shape[1]):
            col = Q[:, j]
            Q_std[:, j] = (col - np.min(col)) / (np.max(col) - np.min(col) + eps)
        Q = Q_std

    if ref_point is None:
        ref = np.array([0.0, 0.0], dtype=float)
    else:
        ref = np.asarray(ref_point, dtype=float).reshape(-1)
        if len(ref) != 2:
            raise ValueError("ref_point must have length 2 for two targets.")

    gains = np.maximum(Q - ref.reshape(1, -1), 0.0)
    hv_score = gains[:, 0] * gains[:, 1]
    hv_score = np.nan_to_num(
        hv_score,
        nan=-np.inf,
        posinf=np.finfo(np.float64).max,
        neginf=-np.inf,
    )

    selected_local_idx = np.argsort(hv_score)[-step_size:]
    selected_local_idx = selected_local_idx[
        np.argsort(hv_score[selected_local_idx])[::-1]
    ]

    return candidates.index[selected_local_idx].tolist()


def grow_disagreementHV_no_diversity_TwoTargets(
    model1, model2,
    xTrainVal, y1TrainVal,
    y2TrainVal, y3TrainVal,
    xTest, y1Test,
    y2Test, y3Test,
    target1,
    target2,
    strSaveDir='',
    strSaveName='growing_HV_no_diversity_2targets',
    strModel1Name='rf',
    strModel2Name='xgb',
    fltStepFrac=0.01,
    intRandomSeed=0,
    do_scale_global=True,
    normalize_global=True,
    pop_size=120,
    n_gen=60,
    p_cx=0.9,
    p_mut=0.25,
    w_train=0.5,
    w_within=0.5,
    top_frac=0.10,
):
    """
    DFT two-target RF/XGBoost disagreement hypervolume WITHOUT diversity.

    DFT targets:
        y1 = e_form
        y2 = bulk_modulus
        y3 = bandgap

    Acquisition:
        1) disagreement on target1 and target2
        2) min-max standardize each independently
        3) HV_i = q1_i * q2_i with reference point [0, 0]
        4) select top-k by HV

    No feature diversity, k-medoids, DPP, or top-M filtering.
    """
    import random
    import time
    import numpy as np
    import pandas as pd
    from myfunc import get_scores

    valid_targets = ['e_form', 'bulk_modulus', 'bandgap']
    selected_targets = [target1, target2]

    for target in selected_targets:
        if target not in valid_targets:
            raise ValueError(f"{target} must be one of: {valid_targets}")

    if target1 == target2:
        raise ValueError("target1 and target2 must be two distinct targets.")

    if strSaveDir != '':
        if strSaveDir[-1] != '/':
            strSaveDir += '/'

    dfGrowing_model1 = pd.DataFrame()
    dfGrowing_model2 = pd.DataFrame()

    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    intTempTrainSize = int(len(xTrainVal) * fltStepFrac)
    intStepSize = int(len(xTrainVal) * fltStepFrac)

    if intTempTrainSize < 1:
        intTempTrainSize = 1
    if intStepSize < 1:
        intStepSize = 1

    lstTrainIndices = np.random.choice(
        xTrainVal.index,
        intTempTrainSize,
        replace=False
    ).tolist()

    while intTempTrainSize < (len(xTrainVal) + 1):
        timeStart = time.time()

        yTrainVal_all = pd.DataFrame({
            'y1TrainVal': y1TrainVal,
            'y2TrainVal': y2TrainVal,
            'y3TrainVal': y3TrainVal,
        })

        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(
            xTrainVal,
            yTrainVal_all,
            lstTrainIndices,
        )

        print('\n--------------------------------------------------------------------------------------------')
        print('Training set size: ', len(xTrain_temp))
        print(
            'Training set percentage: ',
            '{:.1%}'.format(len(xTrain_temp) / len(xTrainVal)),
            '\n',
        )

        y1Train_temp = yTrain_temp['y1Train']
        y2Train_temp = yTrain_temp['y2Train']
        y3Train_temp = yTrain_temp['y3Train']

        print('------------------MODEL1 METRICS FOR EFORM------------------ ')
        eform_pred_val1, _1maes_1, _1rmse_1, _1r2_1 = get_scores(
            model1, xTrain_temp, y1Train_temp, xTest, y1Test, X_val=xVal_temp
        )

        print('------------------MODEL1 METRICS FOR BULK MODULUS ------------------')
        bulk_pred_val1, _2maes_1, _2rmse_1, _2r2_1 = get_scores(
            model1, xTrain_temp, y2Train_temp, xTest, y2Test, X_val=xVal_temp
        )

        print('------------------MODEL1 METRICS FOR BAND GAP------------------ ')
        bandgap_pred_val1, _3maes_1, _3rmse_1, _3r2_1 = get_scores(
            model1, xTrain_temp, y3Train_temp, xTest, y3Test, X_val=xVal_temp
        )

        print('------------------MODEL2 METRICS FOR EFORM------------------ ')
        eform_pred_val2, _1maes_2, _1rmse_2, _1r2_2 = get_scores(
            model2, xTrain_temp, y1Train_temp, xTest, y1Test, X_val=xVal_temp
        )

        print('------------------MODEL2 METRICS FOR BULK MODULUS ------------------')
        bulk_pred_val2, _2maes_2, _2rmse_2, _2r2_2 = get_scores(
            model2, xTrain_temp, y2Train_temp, xTest, y2Test, X_val=xVal_temp
        )

        print('------------------MODEL2 METRICS FOR BAND GAP------------------ ')
        bandgap_pred_val2, _3maes_2, _3rmse_2, _3r2_2 = get_scores(
            model2, xTrain_temp, y3Train_temp, xTest, y3Test, X_val=xVal_temp
        )

        dfGrowing_model1 = pd.concat([
            dfGrowing_model1,
            pd.DataFrame({
                'train_ratio': [round(len(xTrain_temp) / len(xTrainVal), n_train_ratio)],
                'maes_eform': [round(_1maes_1, n_round)],
                'rmse_eform': [round(_1rmse_1, n_round)],
                'r2_eform': [round(_1r2_1, n_round)],
                'maes_bulk_modulus': [round(_2maes_1, n_round)],
                'rmse_bulk_modulus': [round(_2rmse_1, n_round)],
                'r2_bulk_modulus': [round(_2r2_1, n_round)],
                'maes_bandgap': [round(_3maes_1, n_round)],
                'rmse_bandgap': [round(_3rmse_1, n_round)],
                'r2_bandgap': [round(_3r2_1, n_round)],
                'train_size': [len(xTrain_temp)],
            }),
        ], ignore_index=True)

        dfGrowing_model2 = pd.concat([
            dfGrowing_model2,
            pd.DataFrame({
                'train_ratio': [round(len(xTrain_temp) / len(xTrainVal), n_train_ratio)],
                'maes_eform': [round(_1maes_2, n_round)],
                'rmse_eform': [round(_1rmse_2, n_round)],
                'r2_eform': [round(_1r2_2, n_round)],
                'maes_bulk_modulus': [round(_2maes_2, n_round)],
                'rmse_bulk_modulus': [round(_2rmse_2, n_round)],
                'r2_bulk_modulus': [round(_2r2_2, n_round)],
                'maes_bandgap': [round(_3maes_2, n_round)],
                'rmse_bandgap': [round(_3rmse_2, n_round)],
                'r2_bandgap': [round(_3r2_2, n_round)],
                'train_size': [len(xTrain_temp)],
            }),
        ], ignore_index=True)

        # Save after every iteration.
        dfGrowing_model1.to_csv(
            strSaveDir + strSaveName + '_' + strModel1Name + str(intRandomSeed) + '.csv',
            index=False,
        )
        dfGrowing_model2.to_csv(
            strSaveDir + strSaveName + '_' + strModel2Name + str(intRandomSeed) + '.csv',
            index=False,
        )
        pd.DataFrame(lstTrainIndices, columns=['trainIndices']).to_csv(
            strSaveDir + strSaveName + str(intRandomSeed) + '_trainIndices.csv',
            index=False,
        )

        if intTempTrainSize == len(xTrainVal):
            intTempTrainSize += 10
            break

        if len(xVal_temp) == 0:
            print('No validation candidates left. Stopping.')
            break

        pred_map_1 = {
            'e_form': eform_pred_val1,
            'bulk_modulus': bulk_pred_val1,
            'bandgap': bandgap_pred_val1,
        }
        pred_map_2 = {
            'e_form': eform_pred_val2,
            'bulk_modulus': bulk_pred_val2,
            'bandgap': bandgap_pred_val2,
        }

        d1 = np.abs(pred_map_1[target1] - pred_map_2[target1])
        d2 = np.abs(pred_map_1[target2] - pred_map_2[target2])

        d1s = minMaxStandarize(d1)
        d2s = minMaxStandarize(d2)

        diffs_df = pd.DataFrame(
            {'q1': d1s, 'q2': d2s},
            index=yVal_temp.index,
        )

        to_add = select_batch_HV_no_diversity_TwoTargets(
            trainingdata=xTrain_temp,
            candidates=xVal_temp,
            diffs_df=diffs_df,
            step_size=min(intStepSize, len(xVal_temp)),
            top_frac=top_frac,
            do_scale=do_scale_global,
            normalize_features=normalize_global,
            random_state=intRandomSeed,
            standardize_disagreement=False,
            ref_point=[0.0, 0.0],
        )

        if len(to_add) == 0:
            print('RF-disagreement HV no-diversity returned 0 points. Stopping.')
            break

        lstTrainIndices = lstTrainIndices + to_add
        intTempTrainSize = len(lstTrainIndices)

        timeEnd = time.time()
        print('Selected batch size: ', len(to_add))
        print('Selected targets: ', selected_targets)
        print('Time to run loop: ', round(timeEnd - timeStart, 2), ' seconds', flush=True)
        print('\n--------------------------------------------------------------------------------------------')

    dfGrowing_model1.to_csv(
        strSaveDir + strSaveName + '_' + strModel1Name + str(intRandomSeed) + '.csv',
        index=False,
    )
    dfGrowing_model2.to_csv(
        strSaveDir + strSaveName + '_' + strModel2Name + str(intRandomSeed) + '.csv',
        index=False,
    )
    pd.DataFrame(lstTrainIndices, columns=['trainIndices']).to_csv(
        strSaveDir + strSaveName + str(intRandomSeed) + '_trainIndices.csv',
        index=False,
    )

    return