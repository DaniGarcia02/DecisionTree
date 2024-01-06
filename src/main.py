import numpy as np
import pandas as pd
import math
import copy
import time

from ucimlrepo import fetch_ucirepo 

 
class Node():
    def __init__(self, attribute, prevDecision, nSamples, children=None):
        self.attribute = attribute
        self.prevDecision = prevDecision
        self.nSamples = nSamples
        
        if children is None:
            children = []
        self.children = children
        

def VisualizeTree(node, file, i=0):
    if i == 0:
        string = ""
    else:
        string = "  |" + (i-1) * "              |"
        string += "-- " + str(node.prevDecision) + " --> "
    
    if node.children != []:
        file.write(string + "{" + node.attribute + "} " + str(node.nSamples) + " \n")    
        for child in node.children:
            VisualizeTree(child, file, i+1)
    else:
        file.write(string + "{" + node.attribute + "} " + str(node.nSamples) + " \n")
    
        
def GetValueRepetitions(column):
    values = {}
    
    for value in column:
        if value not in values:
            values[value] = 1
        else:
            values[value] += 1
            
    return values


# Always give 'A16' column
def Entropy(column):
    values = GetValueRepetitions(column)
    
    entropy = 0
    for key in values:
        entropy += values[key] / len(column) * math.log(values[key] / len(column), 2)
            
    return -entropy


def EntropySA(df, attribute, values):
    entropySA = 0
    
    for key in values:
        entropySA += values[key] / len(df)* Entropy(df.loc[df[attribute] == key]['A16'].tolist())
    
    return entropySA


def SplitInfo(column, values):
    splitInfo = math.pow(1, -5)
    
    for key in values:
        splitInfo -= values[key] / len(column) * math.log(values[key] / len(column), 2)
    
    return splitInfo    


def Gini(df, attribute, values):    
    giniIndex = 0
    
    for key in values:
        gini = 0
        v = GetValueRepetitions(df.loc[df[attribute] == key]['A16'].tolist())
        
        if len(v.keys()) == 2:
            gini = 1 - (math.pow(v['+'] / values[key], 2) + math.pow(v['-'] / values[key], 2))
        
        giniIndex += values[key] / len(df) * gini
        
    return giniIndex


def MajorityVote(column):
    values = GetValueRepetitions(column)
        
    if values['+'] >= values['-']:
        return '+'
    else:
        return '-'

    
def RemoveRows(df, attribute, value):
    indexes = df[df[attribute] != value].index
    df.drop(indexes, inplace=True)


def readDataset():
    # fetch dataset 
    credit_approval = fetch_ucirepo(id=27)
    
    # data (as pandas dataframes) 
    X = credit_approval.data.features 
    y = credit_approval.data.targets 
    
    credit_approval_data = pd.concat([X, y], axis=1)
    credit_approval_data.dropna(inplace=True)
    
    candidates = X.columns.tolist()
    
    return credit_approval_data, candidates
    

def allEqual(l, symbol):
    for i in l:
        if i == symbol:
            continue
        else:
            return False
    return True


def ID3(df, candidates, method=0, decision="Root"):   
    
    if allEqual(df['A16'], '+'):
        return Node('+', decision, len(df))
            
    if allEqual(df['A16'], '-'):
        return Node('-', decision, len(df))
    
    if not candidates:
        return Node(MajorityVote(df['A16']), decision, len(df))
    
    classifier = None

    # Gain (ID3) or Gain-Ratio (C4.5)
    if method != 2:
        bestGain = 0
        entropy = Entropy(df['A16'])
        for attribute in candidates:
            values = GetValueRepetitions(df[attribute])
            
            entropySA = EntropySA(df, attribute, values)
            gain = entropy - entropySA
            
            # Gain-Ratio (C4.5)
            if method == 1:
                splitInfo = SplitInfo(df[attribute], values)
                gain = gain / splitInfo
                
                
            if gain > bestGain:
                bestGain = gain
                classifier = attribute
    
    # Gini
    if method == 2:
        bestGini = 1
        for attribute in candidates:
            values = GetValueRepetitions(df[attribute])
            
            gini = Gini(df, attribute, values)        
            if gini < bestGini:
                bestGini = gini
                classifier = attribute
   
    if classifier == None:
        return Node(MajorityVote(df['A16']), decision, len(df))
    
    node = Node(classifier, decision, len(df))

    values = GetValueRepetitions(df[classifier])

    for key in values:
        candidates_copy = copy.deepcopy(candidates)
        candidates_copy.remove(classifier)
        df_copy = pd.DataFrame(copy.deepcopy(df.to_dict()))
        RemoveRows(df_copy, classifier, key)
        node.children.append(ID3(df_copy, candidates_copy, method, key))
    
    return node 
    

def PredictRec(node, row):
    if node.children == []:
        return node.attribute
    
    for child in node.children:
        if row[node.attribute] == child.prevDecision:
            return PredictRec(child, row)
    
    return None


def Predict(root, test):
    
    confusionMatrix = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    unableToPredict = 0
    
    for i, row in test.iterrows():
        solution = PredictRec(root, row)
        
        if solution == '+' and solution == row['A16']:
            confusionMatrix['TP'] += 1
        elif solution == '+' and solution != row['A16']:
            confusionMatrix['FP'] += 1
        elif solution == '-' and solution == row['A16']:
            confusionMatrix['TN'] += 1
        elif solution == '-' and solution != row['A16']:
            confusionMatrix['FN'] += 1
        else:  
            unableToPredict += 1    
            
    return confusionMatrix, unableToPredict


def KFold(df, candidates, k, method=0):
    
    folds = np.array_split(df, k)
    bestAccuracy = 0
    total = 0
    bestRoot = None
    
    for i, df_i in enumerate(folds):
        test = df_i
        emptyTrain = True
        for j, df_j in enumerate(folds):
            if i == j:
                continue            
            if emptyTrain:
                train = df_j
                emptyTrain = False
            else:
                train = pd.concat([train, df_j])
                
        print("===== Building Tree", i , "=====")        
                
        st = time.time()
        root = ID3(train, candidates, method)
        et = time.time()
        
        print("Time:", et - st,)
        
        print("===== Making Predicition ", i, "=====")  
        
        st = time.time()
        confusionMatrix, unableToPredict = Predict(root, test)
        et = time.time()
        
        print('-- Confusion Matrix --')
        print(confusionMatrix)
        
        print('Test Length:', len(test))
        print('Unable to Predict:', unableToPredict)
        
        accuracy = (confusionMatrix['TP'] + confusionMatrix['TN']) / len(test)
        print("Accuracy:",  accuracy * 100, "%")
        
        
        if confusionMatrix['TP'] + confusionMatrix['FP'] != 0:
            precision = confusionMatrix['TP'] / (confusionMatrix['TP'] + confusionMatrix['FP'])
        else:
            precision = 0
        print("Precision:", precision * 100, "%")
            
        if confusionMatrix['TP'] + confusionMatrix['FN'] != 0:
            sensitivity = confusionMatrix['TP'] / (confusionMatrix['TP'] + confusionMatrix['FN'])
        else:
            sensitivity = 0
        print("Sensitivity:", sensitivity * 100, "%")
        
        if confusionMatrix['FP'] != 0 or confusionMatrix['TN']:
            specificity = confusionMatrix['TN'] / (confusionMatrix['FP'] + confusionMatrix['TN'])
        else:
            specificity = 0
        print("Specificity:", specificity * 100, "%")
        
        print("Time:", et - st, '\n')
        
        total += accuracy
        
        if bestAccuracy < accuracy:
            bestAccuracy = accuracy
            bestRoot = root
    
    meanAccuracy = total / k
    
    if bestRoot == None:
        bestRoot = root
        bestAccuracy = accuracy
     
    return bestRoot, meanAccuracy


def LeaveOneOut(df, candidates, method=0):
    
    bestAccuracy = 0
    total = 0
    bestRoot = None
    
    i = 0
    while i != df.index[-1]:
        while i not in df.index:
            i += 1
        
        test = df.loc[[i]]
        train = copy.deepcopy(df)
        train.drop(i, inplace=True)
        
        print("===== Building Tree", i , "=====")        
                
        root = ID3(train, candidates, method)
        
        confusionMatrix, unableToPredict = Predict(root, test)
        
        accuracy = (confusionMatrix['TP'] + confusionMatrix['TN']) / len(test)
        total += accuracy
        
        if bestAccuracy < accuracy:
            bestAccuracy = accuracy
            bestRoot = root
            
        i += 1
    
    meanAccuracy = total / len(df)
    
    if bestRoot == None:
        bestRoot = root
        bestAccuracy = accuracy
         
    return bestRoot, meanAccuracy


def main():
    df, candidates = readDataset()
    
    method = 1
    
    if method == 2:
        filename = '../TreeRepresentation/Gini_Tree.txt'
        string = 'Gini'
    elif method == 1:
        filename = '../TreeRepresentation/C4.5_Tree.txt'
        string = 'Gain Ratio (C4.5)'
    else:
        filename = '../TreeRepresentation/ID3_Tree.txt'
        string = 'Gain (ID3)'
    
    print("Method selected:", string, "\n")
    
    #root, meanAccuracy = KFold(df, candidates, 10, method)
    root, meanAccuracy = LeaveOneOut(df, candidates, method)
    
    print("Saving tree with best accuracy...")
    with open(filename, 'w') as file:
        VisualizeTree(root, file)
        
    print("Tree saved in:  " + filename + "\n") 
        
    print("Mean of the accuracy of all trees: ", meanAccuracy * 100, "%")
    
    
if __name__=="__main__":
    main()