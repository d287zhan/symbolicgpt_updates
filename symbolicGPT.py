#!/usr/bin/env python
# coding: utf-8

# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

# load libraries
import os
import glob
import json
import math
import pickle
import random
import numpy as np
#from tqdm import tqdm
from numpy import * # to override the math functions

import torch
import torch.nn as nn
from torch.nn import functional as F
#from torch.utils.data import Dataset

from utils import set_seed, sample_from_model
from matplotlib import pyplot as plt
from trainer import Trainer, TrainerConfig
from models import GPT, GPTConfig, PointNetConfig
from scipy.optimize import minimize, least_squares
from utils import processDataFiles, CharDataset, relativeErr, mse, sqrt, divide, lossFunc
from gam import *

# set the random seed
set_seed(42)

# config
device='gpu'
scratch=True # if you want to ignore the cache and start for scratch
numEpochs = 20 # number of epochs to train the GPT+PT model
embeddingSize = 512 # the hidden dimension of the representation of both GPT and PT
numPoints=[200,201] # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
numVars=2 # the dimenstion of input points x, if you don't know then use the maximum
numYs=1 # the dimension of output points y = f(x), if you don't know then use the maximum
blockSize = 64 # spatial extent of the model for its context
testBlockSize = 400
batchSize = 128 # batch size of training data
target = 'Skeleton' #'Skeleton' #'EQ'
const_range = [-2.1, 2.1] # constant range to generate during training only if target is Skeleton
decimals = 8 # decimals of the points only if target is Skeleton
trainRange = [-3.0,3.0] # support range to generate during training only if target is Skeleton
dataDir = './datasets/'
dataInfo = 'XYE_{}Var_{}Points_{}EmbeddingSize'.format(numVars, numPoints, embeddingSize)
titleTemplate = "{} equations of {} variables - Benchmark"
target = 'Skeleton' #'Skeleton' #'EQ'
dataFolder = '2Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_200Points'
addr = './SavedModels/' # where to save model
method = 'EMB_SUM' # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT/EMB_CON -> whether to concat the embedding or use summation. 

# EMB_CAT: Concat point embedding to GPT token+pos embedding
# EMB_SUM: Add point embedding to GPT tokens+pos embedding
# OUT_CAT: Concat the output of the self-attention and point embedding
# OUT_SUM: Add the output of the self-attention and point embedding
# EMB_CON: Conditional Embedding, add the point embedding as the first token
variableEmbedding = 'NOT_VAR' # NOT_VAR/LEA_EMB/STR_VAR
# NOT_VAR: Do nothing, will not pass any information from the number of variables in the equation to the GPT
# LEA_EMB: Learnable embedding for the variables, added to the pointNET embedding
# STR_VAR: Add the number of variables to the first token
addVars = True if variableEmbedding == 'STR_VAR' else False
maxNumFiles = 100 # maximum number of file to load in memory for training the neural network
bestLoss = None # if there is any model to load as pre-trained one
fName = '{}_SymbolicGPT_{}_{}_{}_MINIMIZE.txt'.format(dataInfo, 
                                             'GPT_PT_{}_{}'.format(method, target), 
                                             'Padding',
                                             variableEmbedding)
perform_gam = True
get_full_train = False
get_full_val= False
get_full_test = False

ckptPath = '{}/{}.pt'.format(addr,fName.split('.txt')[0])
ckptPath_gam = '{}/gam/{}.pt'.format(addr,fName.split('.txt')[0])
try: 
    os.mkdir(addr)
except:
    print('Folder already exists!')

# load the train dataset
train_file = 'train_dataset_{}.pb'.format(fName)
if os.path.isfile(train_file) and not scratch:
    # just load the train set
    with open(train_file, 'rb') as f:
        train_dataset,trainText,chars = pickle.load(f)
else:
    # process training files from scratch
    path = '{}/{}/Train/*.json'.format(dataDir, dataFolder)
    
    # Break it down to only one covariate at a time
    # In order of data x1,x2,.., xn, y for the shape of points
    if perform_gam:
        for i in range(numVars):
            print(f"Creating dataset for x_{i}")
            outpath = '{}/{}/Train/gam/{}_vars_x_{}_dataset.json'.format(dataDir, dataFolder,numVars,i)
            create_gam_datasets( True, path, outpath, i)
    if get_full_train:
        files = glob.glob(path)[:maxNumFiles]
        text = processDataFiles(files)
        chars = sorted(list(set(text))+['_','T','<','>',':']) # extract unique characters from the text before converting the text to a list, # T is for the test data
        text = text.split('\n') # convert the raw text to a set of examples
        trainText_full = text[:-1] if len(text[-1]) == 0 else text
        random.shuffle(trainText_full) # shuffle the dataset, it's important specailly for the combined number of variables experiment
        train_dataset_full = CharDataset(text, blockSize, chars, numVars=numVars, 
                        numYs=numYs, numPoints=numPoints, target=target, addVars=addVars,
                        const_range=const_range, xRange=trainRange, decimals=decimals, augment=False) 
    # with open(train_file, 'wb') as f:
    #     pickle.dump([train_dataset,trainText,chars], f)


        # print a random sample
        idx = np.random.randint(train_dataset_full.__len__())
        #idx = 1
        inputs, outputs, points, variables = train_dataset_full.__getitem__(idx)
        print('inputs:{}'.format(inputs))
        inputs = ''.join([train_dataset_full.itos[int(i)] for i in inputs])
        outputs = ''.join([train_dataset_full.itos[int(i)] for i in outputs])
        print("Train")
        print('id:{}\ninputs:{}\noutputs:{}\npoints:{}\nvariables:{}'.format(idx,inputs,outputs,points, variables))

# load the val dataset
path = '{}/{}/Val/*.json'.format(dataDir,dataFolder)
if get_full_val:
    files = glob.glob(path)
    textVal_full = processDataFiles([files[0]])
    textVal_full = textVal_full.split('\n') # convert the raw text to a set of examples
    val_dataset_full = CharDataset(textVal_full, blockSize, chars, numVars=numVars, 
                    numYs=numYs, numPoints=numPoints, target=target, addVars=addVars,
                    const_range=const_range, xRange=trainRange, decimals=decimals)
    
    # print a random sample
    idx = np.random.randint(val_dataset_full.__len__())
    inputs, outputs, points, variables = val_dataset_full.__getitem__(idx)
    print(points.min(), points.max())
    inputs = ''.join([train_dataset_full.itos[int(i)] for i in inputs])
    outputs = ''.join([train_dataset_full.itos[int(i)] for i in outputs])
    print("Val")
    print('id:{}\ninputs:{}\noutputs:{}\npoints:{}\nvariables:{}'.format(idx,inputs,outputs,points, variables))

if perform_gam:
    for i in range(numVars):
        print(f"Creating dataset for x_{i}")
        outpath = '{}/{}/Val/gam/{}_vars_x_{}_dataset.json'.format(dataDir, dataFolder,numVars,i)
        create_gam_datasets(False, path, outpath, i)


# load the test data
path = f'{dataDir}/{dataFolder}/Test/*.json'
print(f'test path is {path}')

if get_full_test:
    files = glob.glob(path)
    textTest_full = processDataFiles(files)
    textTest_full = textTest_full.split('\n') # convert the raw text to a set of examples
    # test_dataset_target = CharDataset(textTest, blockSize, chars, target=target)
    test_dataset_full = CharDataset(textTest_full, testBlockSize, chars, numVars=numVars, 
                    numYs=numYs, numPoints=numPoints, addVars=addVars,
                    const_range=const_range, xRange=trainRange, decimals=decimals)

    # print a random sample
    idx = np.random.randint(test_dataset_full.__len__())
    inputs, outputs, points, variables = test_dataset_full.__getitem__(idx)
    print(points.min(), points.max())
    inputs = ''.join([train_dataset_full.itos[int(i)] for i in inputs])
    outputs = ''.join([train_dataset_full.itos[int(i)] for i in outputs])
    print("Test")
    print('id:{}\ninputs:{}\noutputs:{}\npoints:{}\nvariables:{}'.format(idx,inputs,outputs,points, variables))
if perform_gam:
    for i in range(numVars):
        print(f"Creating dataset for x_{i}")
        outpath = '{}/{}/Test/gam/{}_vars_x_{}_dataset.json'.format(dataDir, dataFolder,numVars,i)
        create_gam_datasets(False, path, outpath, i)

# Create own training loop to get a better control over training dynamics


gam_path = '{}/{}/Train/gam/*.json'.format(dataDir, dataFolder)


if perform_gam:
    additive_functions = []
    residuals = {}

    train_gam_path = '{}/{}/Train/gam/*.json'.format(dataDir, dataFolder)
    val_gam_path = '{}/{}/Val/gam/*.json'.format(dataDir, dataFolder)
    test_gam_path = '{}/{}/Test/gam/*.json'.format(dataDir, dataFolder)
    
    val_files = glob.glob(val_gam_path)
    test_files = glob.glob(test_gam_path)

    

    for i in range(numVars):
    # Create the Torch CharDataset
    # If not x_1 then update the next dataset with y = residuals
        print(f"Training on x_{i}")
        if i == 0:
            print(f"Reading from {train_gam_path}")
            train_files = [glob.glob(train_gam_path)[i]]
            print(f"Reading file {train_files}")
            # Do similar thing with val and test
            trainText, train_chars, train_data = gam_backfitting_preprocess(False, True, train_files, blockSize, 1, numYs,
                                                    numPoints, target, addVars, const_range, 
                                                    trainRange, decimals, None)
            
        else:
            print(f"Reading from {train_gam_path}")
            train_files = [glob.glob(train_gam_path)[i]]
            # update with residuals
            outpath = '{}/{}/Train/gam/{}_vars_x_{}_dataset_copy.json'.format(dataDir, dataFolder,numVars,i)
            for file in train_files:
                read_json_lines_and_update_y(file, residuals, outpath)
            trainText, train_chars, train_data = gam_backfitting_preprocess(False, True, outpath, blockSize, 1, numYs,
                                                    numPoints, target, addVars, const_range, 
                                                    trainRange, decimals, None)

        val_data = gam_backfitting_preprocess(False, False, val_files, blockSize, 1, numYs,
                                                    numPoints, target, addVars, const_range, 
                                                    trainRange, decimals, train_chars)
    
        textTest, test_data = gam_backfitting_preprocess(True, False, test_files, blockSize,1, numYs,
                                                    numPoints, target, addVars, const_range, 
                                                    trainRange, decimals, train_chars)
        
        # Instantiate the model
        pconf = PointNetConfig(embeddingSize=embeddingSize, 
                       numberofPoints=numPoints[1]-1, 
                       #numberofVars=numVars, # Try changing this to 1 for the purpose of training?
                       numberofVars=1, # 1 because we are training 1 variable at a time
                       numberofYs=numYs, 
                       method=method,
                       variableEmbedding=variableEmbedding)
        mconf = GPTConfig(train_data.vocab_size, train_data.block_size,
                  n_layer=8, n_head=8, n_embd=embeddingSize, 
                  padding_idx=train_data.paddingID)
        model = GPT(mconf, pconf)
        
        # Loading best model after training once 
        if os.path.exists(ckptPath_gam):
            model.load_state_dict(torch.load(ckptPath_gam))

        


        tconf = TrainerConfig(max_epochs=1, batch_size=batchSize, 
                      learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, 
                      final_tokens=2*len(train_data)*blockSize,
                      num_workers=0, ckpt_path=ckptPath_gam)
        
        # Train the model on the train data
        print("Training ==>")
        trainer = Trainer(model, train_data, val_data, tconf, bestLoss, device = device)
        if i != 0:
            trainer.train()

        # Evaluate model on train data to get residuals and the predicted function
        print('The following model {} has been loaded!'.format(ckptPath_gam))
        model.load_state_dict(torch.load(ckptPath_gam))
        model = model.eval().to(trainer.device)

        loader = torch.utils.data.DataLoader(
                                train_data, 
                                shuffle=False, 
                                pin_memory=True,
                                batch_size=1,
                                num_workers=0)
        
        resultDict_tr = {}
        # Compute Residuals

        try:
            with open(fName, 'w', encoding="utf-8") as o:
                resultDict_tr[fName] = {'SymbolicGPT':{'Error': [], 'Residuals': []}}
                for i, batch in enumerate(loader):
                        
                    inputs,outputs,points,variables = batch
                    print('Train Case {}.'.format(i))
                    o.write('Train Case {}/{}.\n'.format(i,len(trainText)-1))

                    tr = json.loads(trainText[i])
                    inputs = inputs[:,0:1].to(trainer.device)
                    points = points.to(trainer.device)
                    variables = variables.to(trainer.device)
                    outputsHat = sample_from_model(
                                model, 
                                inputs, 
                                blockSize, 
                                points=points,
                                variables=variables,
                                temperature=1.0, 
                                sample=True, 
                                top_k=0.0,
                                top_p=0.7)[0]



                    # filter out predicted
                    target = ''.join([train_data.itos[int(i)] for i in outputs[0]])
                    predicted = ''.join([train_data.itos[int(i)] for i in outputsHat])

                    if variableEmbedding == 'STR_VAR':
                        target = target.split(':')[-1]
                        predicted = predicted.split(':')[-1]

                    target = target.strip(train_data.paddingToken).split('>')
                    target = target[0] #if len(target[0])>=1 else target[1]
                    target = target.strip('<').strip(">")
                    predicted = predicted.strip(train_data.paddingToken).split('>')
                    predicted = predicted[0] #if len(predicted[0])>=1 else predicted[1]
                    predicted = predicted.strip('<').strip(">")
                    
                    print('Target:{}\nSkeleton:{}'.format(target, predicted))
                    
                    o.write('{}\n'.format(target))
                    o.write('{}:\n'.format('SymbolicGPT'))
                    o.write('{}\n'.format(predicted))

                    # train a regressor to find the constants (too slow)
                    c = [1.0 for i,x in enumerate(predicted) if x=='C'] # initialize coefficients as 1
                    # c[-1] = 0 # initialize the constant as zero
                    b = [(-2,2) for i,x in enumerate(predicted) if x=='C']  # bounds on variables
                    try:
                        if len(c) != 0:
                            # This is the bottleneck in our algorithm
                            # for easier comparison, we are using minimize package  
                            cHat = minimize(lossFunc, c, #bounds=b,
                                        args=(predicted, tr['X'], tr['Y'])) 
                            
                            predicted = predicted.replace('C','{}').format(*cHat.x)
                    except ValueError:
                        raise 'Err: Wrong Equation {}'.format(predicted)
                    except Exception as e:
                        raise 'Err: Wrong Equation {}, Err: {}'.format(predicted, e)

                    # TODO: let's enjoy GPU

                    print('Skeleton+LS:{}'.format(predicted))

                    Ys_tr = [] #t['YT']
                    Yhats_tr = []
                    for xs in tr['X']:
                        try:
                            eqTmp = target + '' # copy eq
                            eqTmp = eqTmp.replace(' ','')
                            eqTmp = eqTmp.replace('\n','')
                            for i,x in enumerate(xs):
                                # replace xi with the value in the eq
                                if not perform_gam:
                                    eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
                                else:
                                    for idx in range(numVars):
                                        eqTmp = eqTmp.replace('x{}'.format(idx+1), str(x))
                                if ',' in eqTmp:
                                    assert 'There is a , in the equation!'
                            YEval = eval(eqTmp)
                            # YEval = 0 if np.isnan(YEval) else YEval
                            # YEval = 100 if np.isinf(YEval) else YEval
                        except:
                            print('TA: For some reason, we used the default value. Eq:{}'.format(eqTmp))
                            print("Utilizing EQ from data")
                            eqTmp = tr['EQ']
                            eqTmp = eqTmp.replace(' ','')
                            eqTmp = eqTmp.replace('\n','')
                            for i,x in enumerate(xs):
                                if not perform_gam:
                                    eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
                                else:
                                    for idx in range(numVars):
                                        eqTmp = eqTmp.replace('x{}'.format(idx+1), str(x))

                            YEval = eval(eqTmp)    
                            #continue # if there is any point in the target equation that has any problem, ignore it
                            #YEval = 100 #TODO: Maybe I have to punish the model for each wrong template not for each point
                        Ys_tr.append(YEval)
                        try:
                            eqTmp = predicted + '' # copy eq
                            eqTmp = eqTmp.replace(' ','')
                            eqTmp = eqTmp.replace('\n','')
                            for i,x in enumerate(xs):
                                # replace xi with the value in the eq
                                if not perform_gam:
                                    eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
                                else:
                                    for idx in range(numVars):
                                        eqTmp = eqTmp.replace('x{}'.format(idx+1), str(x))

                                if ',' in eqTmp:
                                    assert 'There is a , in the equation!'
                            Yhat = eval(eqTmp)
                            # Yhat = 0 if np.isnan(Yhat) else Yhat
                            # Yhat = 100 if np.isinf(Yhat) else Yhat
                        except:
                            print('PR: For some reason, we used the default value. Eq:{}'.format(eqTmp))
                            Yhat = 100
                        Yhats_tr.append(Yhat)   
                    err = relativeErr(Ys_tr,Yhats_tr, info=True)
                    res = compute_residuals(Ys_tr,Yhats_tr, info=True)
                    residuals[i] = res

                    if type(err) is np.complex128 or np.complex:
                        err = abs(err.real)

                    resultDict_tr[fName]['SymbolicGPT']['Error'].append(err)
                    resultDict_tr[fName]['SymbolicGPT']['Residuals'].append(res)
                    
                    o.write('{}\n{}\n\n'.format( 
                                            predicted,
                                            err
                                            ))

                    print('Err:{}'.format(err))
                    print('Residuals:{}'.format(res))
                    print('') # just an empty line
            print('Avg Err:{}'.format(np.mean(resultDict_tr[fName]['SymbolicGPT'])))
        
        except KeyboardInterrupt:
            print('KeyboardInterrupt')

        loader2 = torch.utils.data.DataLoader(
                                test_data, 
                                shuffle=False, 
                                pin_memory=True,
                                batch_size=1,
                                num_workers=0)


        resultDict = {}
        try:
            with open(fName, 'w', encoding="utf-8") as o:
                resultDict[fName] = {'SymbolicGPT':{'Error': [], 'Residuals':[]}}

                for i, batch in enumerate(loader2):
                        
                    inputs,outputs,points,variables = batch

                    print('Test Case {}.'.format(i))
                    o.write('Test Case {}/{}.\n'.format(i,len(textTest)-1))

                    t = json.loads(textTest[i])
                    inputs = inputs[:,0:1].to(trainer.device)
                    points = points.to(trainer.device)
                    variables = variables.to(trainer.device)
                    outputsHat = sample_from_model(
                                model, 
                                inputs, 
                                blockSize, 
                                points=points,
                                variables=variables,
                                temperature=1.0, 
                                sample=True, 
                                top_k=0.0,
                                top_p=0.7)[0]

                    # filter out predicted
                    target = ''.join([train_data.itos[int(i)] for i in outputs[0]])
                    predicted = ''.join([train_data.itos[int(i)] for i in outputsHat])

                    if variableEmbedding == 'STR_VAR':
                        target = target.split(':')[-1]
                        predicted = predicted.split(':')[-1]

                    target = target.strip(train_data.paddingToken).split('>')
                    target = target[0] #if len(target[0])>=1 else target[1]
                    target = target.strip('<').strip(">")
                    predicted = predicted.strip(train_data.paddingToken).split('>')
                    predicted = predicted[0] #if len(predicted[0])>=1 else predicted[1]
                    predicted = predicted.strip('<').strip(">")
                    
                    print('Target:{}\nSkeleton:{}'.format(target, predicted))
                    
                    o.write('{}\n'.format(target))
                    o.write('{}:\n'.format('SymbolicGPT'))
                    o.write('{}\n'.format(predicted))

                    # train a regressor to find the constants (too slow)
                    c = [1.0 for i,x in enumerate(predicted) if x=='C'] # initialize coefficients as 1
                    # c[-1] = 0 # initialize the constant as zero
                    b = [(-2,2) for i,x in enumerate(predicted) if x=='C']  # bounds on variables
                    try:
                        if len(c) != 0:
                            # This is the bottleneck in our algorithm
                            # for easier comparison, we are using minimize package  
                            cHat = minimize(lossFunc, c, #bounds=b,
                                        args=(predicted, t['X'], t['Y'])) 
                
                            predicted = predicted.replace('C','{}').format(*cHat.x)
                    except ValueError:
                        raise 'Err: Wrong Equation {}'.format(predicted)
                    except Exception as e:
                        raise 'Err: Wrong Equation {}, Err: {}'.format(predicted, e)

                    # TODO: let's enjoy GPU

                    print('Skeleton+LS:{}'.format(predicted))

                    Ys = [] #t['YT']
                    Yhats = []
                    for xs in t['XT']:
                        try:
                            eqTmp = target + '' # copy eq
                            eqTmp = eqTmp.replace(' ','')
                            eqTmp = eqTmp.replace('\n','')
                            for i,x in enumerate(xs):
                                # replace xi with the value in the eq
                                eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
                                if ',' in eqTmp:
                                    assert 'There is a , in the equation!'
                            YEval = eval(eqTmp)
                            # YEval = 0 if np.isnan(YEval) else YEval
                            # YEval = 100 if np.isinf(YEval) else YEval
                        except:
                            print('TA: For some reason, we used the default value. Eq:{}'.format(eqTmp))
                            print("Utilizing EQ from data")
                            eqTmp = t['EQ']
                            eqTmp = eqTmp.replace(' ','')
                            eqTmp = eqTmp.replace('\n','')
                            for i,x in enumerate(xs):
                                if not perform_gam:
                                    eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
                                else:
                                    for idx in range(numVars):
                                        eqTmp = eqTmp.replace('x{}'.format(idx+1), str(x))
                            continue # if there is any point in the target equation that has any problem, ignore it
                            YEval = 100 #TODO: Maybe I have to punish the model for each wrong template not for each point
                        Ys.append(YEval)
                        try:
                            eqTmp = predicted + '' # copy eq
                            eqTmp = eqTmp.replace(' ','')
                            eqTmp = eqTmp.replace('\n','')
                            for i,x in enumerate(xs):
                                # replace xi with the value in the eq
                                eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
                                if ',' in eqTmp:
                                    assert 'There is a , in the equation!'
                            Yhat = eval(eqTmp)
                            # Yhat = 0 if np.isnan(Yhat) else Yhat
                            # Yhat = 100 if np.isinf(Yhat) else Yhat
                        except:
                            print('PR: For some reason, we used the default value. Eq:{}'.format(eqTmp))
                            Yhat = 100
                        Yhats.append(Yhat)
                    err = relativeErr(Ys,Yhats, info=True)
                    # Don't need to compute residuals on test set
                    # res = compute_residuals(Ys,Yhats, info=True)

                    if type(err) is np.complex128 or np.complex:
                        err = abs(err.real)

                    resultDict[fName]['SymbolicGPT']['Error'].append(err)
                    #resultDict[fName]['SymbolicGPT']['Residuals'].append(res)
                    
                    o.write('{}\n{}\n\n'.format( 
                                            predicted,
                                            err
                                            #res
                                            ))

                    print('Err:{}'.format(err))
                    #print('Residuals:{}'.format(res))
                    print('') # just an empty line
            print('Avg Err:{}'.format(np.mean(resultDict[fName]['SymbolicGPT'])))
        
        except KeyboardInterrupt:
            print('KeyboardInterrupt')

else:
    try:
        # create the model
        pconf = PointNetConfig(embeddingSize=embeddingSize, 
                       numberofPoints=numPoints[1]-1, 
                       numberofVars=numVars, 
                       numberofYs=numYs,
                       method=method,
                       variableEmbedding=variableEmbedding)
        mconf = GPTConfig(train_dataset_full.vocab_size, train_dataset_full.block_size,
                  n_layer=8, n_head=8, n_embd=embeddingSize, 
                  padding_idx=train_dataset_full.paddingID)
        model = GPT(mconf, pconf)


        # initialize a trainer instance and kick off training
        tconf = TrainerConfig(max_epochs=numEpochs, batch_size=batchSize, 
                      learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, 
                      final_tokens=2*len(train_dataset_full)*blockSize,
                      num_workers=0, ckpt_path=ckptPath)
        trainer = Trainer(model, train_dataset_full, val_dataset_full, tconf, bestLoss, device=device)
        trainer.train()
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
 
        

# # load the best model before training
# print('The following model {} has been loaded!'.format(ckptPath))
# model.load_state_dict(torch.load(ckptPath))
# model = model.eval().to(trainer.device)

# try:
#     trainer.train()
# except KeyboardInterrupt:
#     print('KeyboardInterrupt')

    #  load the best model
    print('The following model {} has been loaded!'.format(ckptPath))
    model.load_state_dict(torch.load(ckptPath))
    model = model.eval().to(trainer.device)

    ## Test the model
    # alright, let's sample some character-level symbolic GPT 
    loader = torch.utils.data.DataLoader(
                                    test_dataset_full, 
                                    shuffle=False, 
                                    pin_memory=True,
                                    batch_size=1,
                                    num_workers=0)

    from utils import *
    resultDict = {}
    try:
        with open(fName, 'w', encoding="utf-8") as o:
            resultDict[fName] = {'SymbolicGPT':[]}

            for i, batch in enumerate(loader):
                    
                inputs,outputs,points,variables = batch

                print('Test Case {}.'.format(i))
                o.write('Test Case {}/{}.\n'.format(i,len(textTest_full)-1))

                t = json.loads(textTest_full[i])

                inputs = inputs[:,0:1].to(trainer.device)
                points = points.to(trainer.device)
                variables = variables.to(trainer.device)
                outputsHat = sample_from_model(
                            model, 
                            inputs, 
                            blockSize, 
                            points=points,
                            variables=variables,
                            temperature=1.0, 
                            sample=True, 
                            top_k=0.0,
                            top_p=0.7)[0]

                # filter out predicted
                target = ''.join([train_dataset_full.itos[int(i)] for i in outputs[0]])
                predicted = ''.join([train_dataset_full.itos[int(i)] for i in outputsHat])

                if variableEmbedding == 'STR_VAR':
                    target = target.split(':')[-1]
                    predicted = predicted.split(':')[-1]

                target = target.strip(train_dataset_full.paddingToken).split('>')
                target = target[0] #if len(target[0])>=1 else target[1]
                target = target.strip('<').strip(">")
                predicted = predicted.strip(train_dataset_full.paddingToken).split('>')
                predicted = predicted[0] #if len(predicted[0])>=1 else predicted[1]
                predicted = predicted.strip('<').strip(">")
                
                print('Target:{}\nSkeleton:{}'.format(target, predicted))
                
                o.write('{}\n'.format(target))
                o.write('{}:\n'.format('SymbolicGPT'))
                o.write('{}\n'.format(predicted))

                # train a regressor to find the constants (too slow)
                c = [1.0 for i,x in enumerate(predicted) if x=='C'] # initialize coefficients as 1
                # c[-1] = 0 # initialize the constant as zero
                b = [(-2,2) for i,x in enumerate(predicted) if x=='C']  # bounds on variables
                try:
                    if len(c) != 0:
                        # This is the bottleneck in our algorithm
                        # for easier comparison, we are using minimize package  
                        cHat = minimize(lossFunc, c, #bounds=b,
                                    args=(predicted, t['X'], t['Y'])) 
            
                        predicted = predicted.replace('C','{}').format(*cHat.x)
                except ValueError:
                    raise 'Err: Wrong Equation {}'.format(predicted)
                except Exception as e:
                    raise 'Err: Wrong Equation {}, Err: {}'.format(predicted, e)

                # TODO: let's enjoy GPU

                print('Skeleton+LS:{}'.format(predicted))

                Ys = [] #t['YT']
                Yhats = []
                for xs in t['XT']:
                    try:
                        eqTmp = target + '' # copy eq
                        eqTmp = eqTmp.replace(' ','')
                        eqTmp = eqTmp.replace('\n','')
                        for i,x in enumerate(xs):
                            # replace xi with the value in the eq
                            eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
                            if ',' in eqTmp:
                                assert 'There is a , in the equation!'
                        YEval = eval(eqTmp)
                        # YEval = 0 if np.isnan(YEval) else YEval
                        # YEval = 100 if np.isinf(YEval) else YEval
                    except:
                        print('TA: For some reason, we used the default value. Eq:{}'.format(eqTmp))
                        print(i)
                        raise
                        continue # if there is any point in the target equation that has any problem, ignore it
                        YEval = 100 #TODO: Maybe I have to punish the model for each wrong template not for each point
                    Ys.append(YEval)
                    try:
                        eqTmp = predicted + '' # copy eq
                        eqTmp = eqTmp.replace(' ','')
                        eqTmp = eqTmp.replace('\n','')
                        for i,x in enumerate(xs):
                            # replace xi with the value in the eq
                            eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
                            if ',' in eqTmp:
                                assert 'There is a , in the equation!'
                        Yhat = eval(eqTmp)
                        # Yhat = 0 if np.isnan(Yhat) else Yhat
                        # Yhat = 100 if np.isinf(Yhat) else Yhat
                    except:
                        print('PR: For some reason, we used the default value. Eq:{}'.format(eqTmp))
                        Yhat = 100
                    Yhats.append(Yhat)
                err = relativeErr(Ys,Yhats, info=True)

                if type(err) is np.complex128 or np.complex:
                    err = abs(err.real)

                resultDict[fName]['SymbolicGPT'].append(err)

                o.write('{}\n{}\n\n'.format( 
                                        predicted,
                                        err
                                        ))

                print('Err:{}'.format(err))
                
                print('') # just an empty line
        print('Avg Err:{}'.format(np.mean(resultDict[fName]['SymbolicGPT'])))
        
    except KeyboardInterrupt:
        print('KeyboardInterrupt')

# plot the error frequency for model comparison
if not perform_gam:
    num_eqns = len(resultDict[fName]['SymbolicGPT'])
else:
    num_eqns = len(resultDict[fName]['SymbolicGPT']['Error'])
num_vars = pconf.numberofVars
title = titleTemplate.format(num_eqns, num_vars)

models = list(key for key in resultDict[fName].keys() if len(resultDict[fName][key])==num_eqns)
lists_of_error_scores = [resultDict[fName][key] for key in models if len(resultDict[fName][key])==num_eqns]
linestyles = ["-","dashdot","dotted","--"]

eps = 0.00001
y, x, _ = plt.hist([np.log([max(min(x+eps, 1e5),1e-5) for x in e]) for e in lists_of_error_scores],
                   label=models,
                   cumulative=True, 
                   histtype="step", 
                   bins=2000, 
                   density=True,
                   log=False)
y = np.expand_dims(y,0)
plt.figure(figsize=(15, 10))

for idx, m in enumerate(models): 
    plt.plot(x[:-1], 
           y[idx] * 100, 
           linestyle=linestyles[idx], 
           label=m)

plt.legend(loc="upper left")
plt.title(title)
plt.xlabel("Log of Relative Mean Square Error")
plt.ylabel("Normalized Cumulative Frequency")

name = '{}.png'.format(fName.split('.txt')[0])
plt.savefig(name)