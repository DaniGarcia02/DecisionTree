import numpy as np
import pandas as pd
import math
import copy
import time

from ucimlrepo import fetch_ucirepo 

 
class Node():
    def __init__(self, attribute, prevDecision, children=None):
        self.attribute = attribute
        self.prevDecision = prevDecision
        
        if children is None:
            children = []
        self.children = children
        

def VisualizeTree(node, i, file):
    if i == 0:
        string = ""
    else:
        string = "  |" + (i-1) * "              |"
        string += "-- " + str(node.prevDecision) + " -->"
    
    if node.children != []:
        file.write(string + "{" + node.attribute + "}" + "\n")    
        for child in node.children:
            VisualizeTree(child, i+1, file)
    else:
        file.write(string + "{" + node.attribute + "}" + "\n")
    
        
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


def EntropySA(df, attribute):
    values = GetValueRepetitions(df[attribute])
    entropySA = 0
    
    for key in values:
        entropySA += values[key] / len(df)* Entropy(df.loc[df[attribute] == key]['A16'].tolist())
    
    return entropySA


def SplitInfo(column):
    values = GetValueRepetitions(column)
    splitInfo = 0
    
    for key in values:
        splitInfo += values[key] / len(column) * math.log(values[key] / len(column), 2)
    
    return -splitInfo    


def Gini(df, attribute):    
    values = GetValueRepetitions(df[attribute])
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
        return 'M+'
    else:
        return 'M-'

    
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
        return Node('+', decision)
            
    if allEqual(df['A16'], '-'):
        return Node('-', decision)
    
    if not candidates:
        return Node(MajorityVote(df['A16']), decision)
    
    classifier = None

    # Gain or Gain-Ratio
    if method != 2:
        bestGain = 0
        entropy = Entropy(df['A16'])
        for attribute in candidates:
            entropySA = EntropySA(df, attribute)
            gain = entropy - entropySA
            
            # Gain-Ratio
            if method == 1:
                splitInfo = SplitInfo(df[attribute])
                gain = gain / splitInfo
                
            if gain > bestGain:
                bestGain = gain
                classifier = attribute
    
    # Gini
    if method == 2:
        bestGini = 1
        for attribute in candidates:
            gini = Gini(df, attribute)        
            if gini < bestGini:
                bestGini = gini
                classifier = attribute
   
    if classifier == None:
        return Node(MajorityVote(df['A16']), decision)
    
    node = Node(classifier, decision)
    
    values = GetValueRepetitions(df[classifier])

    for key in values:
        candidates_copy = copy.deepcopy(candidates)
        candidates_copy.remove(classifier)
        df_copy = pd.DataFrame(copy.deepcopy(df.to_dict()))
        RemoveRows(df_copy, classifier, key)
        node.children.append(ID3(df_copy, candidates_copy, method, key))
    
    return node 
    

def main():
    df, candidates = readDataset()
    
    print(df)
    
    method = 0
    
    st = time.time()
    root = ID3(df, candidates, method)
    et = time.time()
    
    #print(root)

    if method == 2:
        filename = 'GiniTree.txt'
    elif method == 1:
        filename = 'GainRatioTree.txt'
    else:
        filename = 'GainTree.txt' 
    
    with open(filename, 'w') as file:
        VisualizeTree(root, 0, file)
    
    print("Time:", et - st)
    
    
    
if __name__=="__main__":
    main()