#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:35:01 2018

@author: abhijay
"""

import numpy as np
import pandas as pd
#import os
#import pickle
import copy
from sklearn import preprocessing
from sklearn import tree

#os.chdir('/home/abhijay/Documents/ML/hw_2/Q10')

class tree_node():
    def __init__(self, splitOnAttribute):
        self.attribute = splitOnAttribute
        self.children = []
        self.attributeValue = []
        self.attributeValueType = None

def make_target_variable(data):
    data['salary-bracket'] = data['salary-bracket'].apply(lambda y: 0 if y==" <=50K" else 1)
    return data

def find_categorical_continuous_features(data):
    categorical_features = [data.columns[col] for col, col_type in enumerate(data.dtypes) if col_type == np.dtype('O') ]
    continuous_features = list(set(data.columns) - set(categorical_features))
    return categorical_features, continuous_features

##### entropy calculation #####
def entropy(data):
    binCount = np.bincount(data) # frequency
    ind = np.nonzero(binCount)[0] # indices or attribute value
    stackedBinCounts = (np.vstack((ind, binCount[ind])).T).astype(float) # stack with attribute value against frequency
    stackedBinCounts[:,1] = stackedBinCounts[:,1]/data.shape[0]
    return sum([(-stackedBinCount[1] * np.log2(stackedBinCount[1])) for stackedBinCount in stackedBinCounts if stackedBinCount[1]!=0.0]) # Calculate entropy

##### entropy calculation for continuous variables #####
##### Getting the splits i.e. if continuous variable is 1,2,3,4
##### Then splits are 1.5, 2.5, 3.5 (This was mentioned in class)
def gain_continuousVar(y, x, split):
    entropy_ = entropy(y) # H(X) As mentioned in the lecture slides
    entropy_ -= ((x>split).sum()/len(x)) * entropy(y[x > split])
    entropy_ -= ((x<split).sum()/len(x)) * entropy(y[x < split])
    return entropy_ # return Info gain (weighted sum of entropy; do not confuse)

##### entropy calculation for categorical variables #####
def gain_categoricalVar(x,y):
    entropy_ = entropy(x) # H(X) As mentioned in the lecture slides
    binCount = np.bincount(x) # frequency
    ind = np.nonzero(binCount)[0] # indices
    stackedBinCounts = (np.vstack((ind, binCount[ind])).T).astype(float)
    stackedBinCounts[:,1] = stackedBinCounts[:,1]/x.shape[0]
    ##### Calculate info gain using entropy #####
    return entropy_ + sum([(-stackedBinCount[1] * entropy(y[x == stackedBinCount[0]])) for stackedBinCount in stackedBinCounts])

##### Traverse the tree and predict #####
def predict(node, x, y):
    yPred = np.array([])
    
    ##### Traverse the tree #####
    if node.attributeValueType == 0:
        ##### if categorical split #####
        for child, attributeVal in zip(node.children, node.attributeValue):
            if type(child) == np.ndarray:
                ##### if child is an array and not a pointer to a branch then predict #####
                y_ = y[x[:, int(node.attribute)] == attributeVal]
                y_predicted = np.full(y_.shape, np.sign(np.sum(child)))
                yPred = np.append(yPred,np.array(y_predicted == y_).astype(int))
            else:
                ##### if child is pointer to a branch then branch out #####
                if x.shape[0] > 0:
                    x_ = x[x[:, int(node.attribute)] == attributeVal, :]
                    y_ = y[x[:, int(node.attribute)] == attributeVal]
                    y_predicted = predict(child, x_, y_)
                    yPred = np.append(yPred, y_predicted)
    else:
        ##### if continuous split #####
        for i, child in enumerate(node.children):
            if i == 0:
                ##### if greater than split condition #####
                if type(child) == np.ndarray:
                    y_ = y[x[:, int(node.attribute.split('_')[0])] > float(node.attribute.split('_')[1])]
                    y_predicted = np.full(y_.shape, np.sign(np.sum(child)))
                    yPred = np.append(yPred, np.array(y_predicted == y_).astype(int))
                elif x.shape[0] > 0:
                    ##### if child is pointer to a branch then branch out #####
                    x_ = x[x[:, int(node.attribute.split('_')[0])] > float(node.attribute.split('_')[1]), :]
                    y_ = y[x[:, int(node.attribute.split('_')[0])] > float(node.attribute.split('_')[1])]
                    y_predicted = predict(child, x_, y_)
                    yPred = np.append(yPred, y_predicted)
            elif i == 1:
                ##### if less than split condition #####
                if type(child) == np.ndarray:
                    ##### if child is an array and not a pointer to a branch then predict #####
                    y_ = y[x[:, int(node.attribute.split('_')[0])] < float(node.attribute.split('_')[1])]
                    y_predicted = np.full(y_.shape, np.sign(np.sum(child)))
                    yPred = np.append(yPred, np.array(y_predicted == y_).astype(int))
                elif x.shape[0] > 0:
                    ##### if child is pointer to a branch then branch out #####
                    x_ = x[x[:, int(node.attribute.split('_')[0])] < float(node.attribute.split('_')[1]), :]
                    y_ = y[x[:, int(node.attribute.split('_')[0])] < float(node.attribute.split('_')[1])]
                    y_predicted = predict(child, x_, y_)
                    yPred = np.append(yPred, y_predicted)
    return yPred

##### Pruning #####
def pruning(node, x, y):
    prune = False
    yPred = np.array([])

    if x.shape[0] == 0:
        return yPred, node, prune
    
    ##### Traverse the tree #####
    if node.attributeValueType == 0:
        ##### if categorical split #####
        for i, (child, attributeVal) in enumerate(zip(node.children, node.attributeValue)):
            if type(child) == np.ndarray:
                y_ = y[x[:, int(node.attribute)] == attributeVal]
                y_predicted = np.full(y_.shape, np.sign(np.sum(child)))
                yPred = np.append(yPred, np.array(y_predicted == y_).astype(int))
            else:
                if x.shape[0] > 0:
                    x_ = x[x[:, int(node.attribute)] == attributeVal, :]
                    y_ = y[x[:, int(node.attribute)] == attributeVal]
                    y_predicted, node.children[i], prune = pruning(child, x_, y_)
                    if prune:
                        ##### Prune and assign class #####
                        node.children[i] = y_
                        yPred = np.append(yPred, y_)
                        prune = False
                    else:
                        yPred = np.append(yPred, y_predicted)
    else:
        ##### if continuous split #####
        for i, child in enumerate(node.children):
            ##### is greater than split condition #####
            if i == 0:
                if type(child) == np.ndarray:
                    y_ = y[x[:, int(node.attribute.split('_')[0])] > float(node.attribute.split('_')[1])]
                    y_predicted = np.full(y_.shape, np.sign(np.sum(child)))
                    yPred = np.append(yPred, np.array(y_predicted == y_).astype(int))
                elif x.shape[0] > 0:
                    x_ = x[x[:, int(node.attribute.split('_')[0])] > float(node.attribute.split('_')[1]), :]
                    y_ = y[x[:, int(node.attribute.split('_')[0])] > float(node.attribute.split('_')[1])]
                    y_predicted, node.children[i], prune = pruning(child, x_, y_)
                    if prune:
                        ##### Prune and assign class #####
                        node.children[i] = y_
                        yPred = np.append(yPred, y_)
                        prune = False
                    else:
                        yPred = np.append(yPred, y_predicted)
            elif i == 1:
                ##### if less than split condition #####
                if type(child) == np.ndarray:
                    y_ = y[x[:, int(node.attribute.split('_')[0])] < float(node.attribute.split('_')[1])]
                    y_predicted = np.full(y_.shape, np.sign(np.sum(child)))
                    yPred = np.append(yPred, np.array(y_predicted == y_).astype(int))
                elif x.shape[0] > 0:
                    x_ = x[x[:, int(node.attribute.split('_')[0])] < float(node.attribute.split('_')[1]), :]
                    y_ = y[x[:, int(node.attribute.split('_')[0])] < float(node.attribute.split('_')[1])]
                    y_predicted, node.children[i], prune = pruning(child, x_, y_)
                    if prune:
                        ##### Prune and assign class #####
                        node.children[i] = y_
                        yPred = np.append(yPred, y_)
                        prune = False
                    else:
                        yPred = np.append(yPred, y_predicted)

    ##### prune condition #####
    ##### If better validation accuracy can be achieved on pruning this branch #####
    if (yPred.sum() < y.sum()):
        prune = True
    return yPred, node, prune

def pure(y):
    return len(set(y)) == 1

# Reference: http://gabrielelanaro.github.io/blog/2016/03/03/decision-trees.html
# Took a little inspiration from the way partition is done for optimization while training ..mentioned as refer[1]
# Rest of the code is my own work; Please note every other function is my own implementation and very different from the code by gabrielelanaro.
# Even id3_train() function is heavily tailored to they way we were supposed to implement the algo and very little reference has been taken from gabrielelanaro's.
def partition(a):
    return {c: (a==c).nonzero()[0] for c in np.unique(a)}

def id3_train(x, y):
    
    if pure(y) or len(y) == 0:
        return y

    ##### Calculate gain of categorical variables #####
    gainCategoricalVar = np.apply_along_axis(gain_categoricalVar, 0, x[:,categorical_features], y)
    
    ##### Calculate gain of continuous variables #####
    gainContinuousVar = {}
    for continuous_feature in continuous_features:
        for split in splits[continuous_feature]:
            gainContinuousVar[str(continuous_feature)+'_'+str(split)] = gain_continuousVar(y,x[:,continuous_feature],split)
    
    ##### Is the featureToSplitOn categorical or continuous
    if np.max(gainCategoricalVar) > gainContinuousVar[max(gainContinuousVar, key=gainContinuousVar.get)]:
        
        ##### Categorical feature to split on #####
        featureToSplitOn = categorical_features[np.argmax(gainCategoricalVar)]
        
        if np.max(gainCategoricalVar) < 1e-4:
            return y
        
        # Refer ..[1]
        sets = partition(x[:, featureToSplitOn])
                
        node = tree_node(str(featureToSplitOn))
        node.attributeValueType = 0 # Categorical
        
        # Refer ..[1]
        for k, v in sets.items():
            y_subset = y.take(v, axis=0)
            x_subset = x.take(v, axis=0)
            node.attributeValue.append(k)
            node.children.append(id3_train(x_subset, y_subset))
        
        ##### Some attributes were getting missed add those splits #####
        for key in (set(featuresUniqueValues[featureToSplitOn]) - set(sets.keys())):
            node.attributeValue.append(key)
            node.children.append(y)
            
    else:
        
        ##### Coninuous feature to split on #####
        featureToSplitOn = max(gainContinuousVar, key=gainContinuousVar.get)
            
        if (gainContinuousVar[max(gainContinuousVar, key=gainContinuousVar.get)] < 1e-4):
            return y
        
        dataSplitIndices = [ x[:, int(featureToSplitOn.split('_')[0])]>float(featureToSplitOn.split('_')[1]), x[:, int(featureToSplitOn.split('_')[0])]<float(featureToSplitOn.split('_')[1])]
        node = tree_node(featureToSplitOn)
        node.attributeValueType = 1
        for indices in dataSplitIndices:
            y_subset = y[indices]
            x_subset = x[indices]            
            node.children.append(id3_train(x_subset, y_subset))
    return node

if __name__ == "__main__":
    
    col_names = ["age","workclass","education","marital-status","occupation","race","gender","hours-per-week","native-country","salary-bracket"]
    
    ##### Load data #####
    train_data = pd.read_csv("income-data/income.train.txt", names = col_names)
    dev_data = pd.read_csv("income-data/income.dev.txt", names = col_names)
    test_data = pd.read_csv("income-data/income.test.txt", names = col_names)
    
    train_data = make_target_variable(train_data)
    test_data = make_target_variable(test_data)
    dev_data = make_target_variable(dev_data)
        
    categorical_features_, continuous_features_ = find_categorical_continuous_features(train_data.iloc[:,0:-1])
    
    categorical_features = [train_data.columns.get_loc(c) for c in categorical_features_]
    
    continuous_features = [train_data.columns.get_loc(c) for c in continuous_features_]
    
    ##### Encoding categorical values to labels #####
    le = preprocessing.LabelEncoder()
    all_df = pd.concat([train_data,test_data,dev_data])
    for feature in categorical_features_:
        le.fit(all_df[feature])
        train_data[feature] = le.transform(train_data[feature])
        test_data[feature] = le.transform(test_data[feature])
        dev_data[feature] = le.transform(dev_data[feature])
    
    featuresUniqueValues = [train_data[col].unique() for col in col_names]
    
    ##### Convert pandas dataframe to numpy array #####
    x = train_data.iloc[:,0:train_data.shape[1]-1].values
    y = (train_data.values)[:,-1]
    
    x_test = test_data.iloc[:,0:test_data.shape[1]-1].values
    y_test = (test_data.values)[:,-1]
    
    x_dev = dev_data.iloc[:,0:dev_data.shape[1]-1].values
    y_dev = (dev_data.values)[:,-1]
    
    ##### Getting the splits i.e. if continuous variable is 1,2,3,4
    ##### Then splits are 1.5, 2.5, 3.5 (This was mentioned in class)
    splits = {}
    for feature in continuous_features:
        uniqueValues = np.unique(x[:,feature])
        uniqueValues.sort()
        splits[feature] = uniqueValues[0:-1] + (uniqueValues[1:] - uniqueValues[0:-1]) / 2
    
    
    print ("\nTraining please wait ..... (takes 60 seconds)")
    import time
    start_time = time.time()
    node_ = id3_train(x,y)
    print("--- %s seconds ---" % (time.time() - start_time))

    ##### For storing the model #####
#    with open("id3tree_v4", "wb") as f:
#        pickle.dump(node_, f)
#    
#    with open("id3tree_v4", "rb") as f:
#        node_ = pickle.load(f)
    
    
    yPred_train = predict(node_, x, y)
    print ("\nTraining Accuracy: "+str( round(100*yPred_train.sum()/x.shape[0],2))+"%")
    
    yPred_dev = predict(node_, x_dev, y_dev)
    print ("\nDev Accuracy: "+str( round(100*yPred_dev.sum()/x_dev.shape[0],2))+"%")
                  
    yPred_test = predict(node_, x_test, y_test)
    print ("\nTesting Accuracy: "+str( round(100*yPred_test.sum()/x_test.shape[0],2))+"%")
    
    print ("\n\nPruning.....")   
    
    yPred, node_pruned, prune = pruning(copy.deepcopy(node_), x_dev, y_dev)
        
    yPred_train_pruned = predict(node_pruned, x, y)
    print ("\nTraining Accuracy: "+str( round(100*yPred_train_pruned.sum()/x.shape[0],2))+"%")    
    
    yPred_dev_pruned = predict(node_pruned, x_dev, y_dev)
    print ("\nDev Accuracy: "+str( round(100*yPred_dev_pruned.sum()/x_dev.shape[0],2))+"%")
    
    yPred_test_pruned = predict(node_pruned, x_test, y_test)
    print ("\nTesting Accuracy: "+str( round(100*yPred_test_pruned.sum()/x_test.shape[0],2))+"%")
    
    
    print ("\n\nComparing with Scikit implementation.....")
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x, y)

    print ("\nTraining Accuracy: "+str( round(100*np.sum(y == clf.predict(x))/x.shape[0],2))+"%")
    print ("\nDev Accuracy: "+str( round(100*np.sum(y_dev == clf.predict(x_dev))/x_dev.shape[0],2))+"%")
    print ("\nTesting Accuracy: "+str( round(100*np.sum(y_test == clf.predict(x_test))/x_test.shape[0],2))+"%")