#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np

#filename='all_data/train_c300_d100.csv'
#filename='B.csv'
#data=np.loadtxt(filename, delimiter=',', dtype=bool)

#rows,cols=data.shape
#clscol=cols-1
#clsvals=set(data[:,clscol])
#%timeit np.where(df['B'] == 0, 0, df['A'] / df['B'])       # 1.17 ms
#%timeit (df['A'] / df['B']).replace([np.inf, -np.inf], 0)  # 1.96 ms

def CalcEntropy(arr):
    """To calculate and return the entropy of the current node"""

    #To remove log 0 error from code
    if 0 in arr:
        return 0
    #To remove log 0 error from code
    
    arr=np.array(arr)
    total=[sum(arr)]
    return -sum(arr*np.log2(arr/total)/total)

def CalcEntropy1(data):
    """To calculate and return the entropy of the current node"""

    #To remove log 0 error from code
    if np.size(data) == 0:
        return 0
    #To remove log 0 error from code
    
    arr = np.array(data)
    a,b = np.unique(arr[:,-1] , return_counts = True)
    total = [np.size(arr,0)]
    return -sum(b*np.log2(b/total)/total)

def CalcEntropyFull(data, avail_cols):
    """To calculate the entropy for all the features in one go"""
    arr=np.array(data)
    X=arr[:,:-1]
    Y=arr[:,-1]
    xy=((((~X).T*(~Y).T).T).sum(0)).astype(float)
    xy[xy==0]=0.1
    xY=((((~X).T*Y.T).T).sum(0)).astype(float)
    xY[xY==0]=0.1
    Xy=(((X.T*(~Y).T).T).sum(0)).astype(float)
    Xy[Xy==0]=0.1
    XY=(((X.T*Y.T).T).sum(0)).astype(float)
    XY[XY==0]=0.1
    
    #print(xy*np.log2(xy/(xy+xY))+xY*np.log2(xY/(xy+xY))+Xy*np.log2(Xy/(Xy+XY))+XY*np.log2(XY/(Xy+XY)))
    #return np.argmax(xy*np.log2(xy/(xy+xY))+xY*np.log2(xY/(xy+xY))+Xy*np.log2(Xy/(Xy+XY))+XY*np.log2(XY/(Xy+XY)))
    return avail_cols[np.argmax((xy*np.log2(xy/(xy+xY))+xY*np.log2(xY/(xy+xY))+Xy*np.log2(Xy/(Xy+XY))+XY*np.log2(XY/(Xy+XY)))[avail_cols])]

def CalcVariance(data):
    """To Calculate and return the variance of the current node"""
    
    arr = np.array(data)
    a,b = np.unique(arr[:,-1], return_counts = True)
    return np.where(len(b) == 2, np.prod(b)/(np.size(arr,0)**2), 0)

def CalcVarianceFull(data, avail_cols):
    """To calculate the entropy for all the features in one go"""
    arr=np.array(data)
    X=arr[:,:-1]
    Y=arr[:,-1]
    xy=((((~X).T*(~Y).T).T).sum(0)).astype(float)
    #xy[xy==0]=0.1
    xY=((((~X).T*Y.T).T).sum(0)).astype(float)
    #xY[xY==0]=0.1
    Xy=(((X.T*(~Y).T).T).sum(0)).astype(float)
    #Xy[Xy==0]=0.1
    XY=(((X.T*Y.T).T).sum(0)).astype(float)
    #XY[XY==0]=0.1
    
    #result = (xy*xY/(xy+xY)+Xy*XY/(Xy+XY))[avail_cols]
    #print(avail_cols)
    
    #print(xy*xY/(xy+xY)+Xy*XY/(Xy+XY))
    return avail_cols[np.nanargmin((xy*xY/(xy+xY)+Xy*XY/(Xy+XY))[avail_cols])]

def GetBestFeatureToSplit(data, avail_cols, parentropy, rows, heuristic=None):
    """To calculate and return the best feature to split on and the entropy of the child-node"""
    
    if heuristic=="Entropy":
        selected_feature = CalcEntropyFull(data, avail_cols)
    elif heuristic=="Variance":
        selected_feature = CalcVarianceFull(data, avail_cols)
    leftdata = data[data[:, selected_feature] == False]
    rightdata = data[data[:, selected_feature] == True]
    
    return [selected_feature, leftdata, rightdata]

class Node:
    """To represnet a node of a tree"""
    
    def __init__(self, data, parent = None, heuristicValue = None, rows = None, cols = None, path = "", depth = 0):
        
        if data is None or len(data) == 0:
            print(parent)
            np.savetxt('Trace.csv', parent.data, fmt="%d", delimiter=',')
            raise NotFoundError("Data not found")
        else:
            self.data = data
            self.parent = parent
            self.leftchild = None
            self.rightchild = None
            self.path = path
            self.depth = depth
            self.split_on = None
            self.classifier = None
            self.heuristicValue = heuristicValue
            self.rows, self.cols = self.data.shape
            if parent:
                self.avail_cols = np.delete(parent.avail_cols, parent.avail_cols.index(parent.split_on)).tolist() 
            else:
                self.avail_cols = [i for i in range(self.cols - 1)]
                
    def __repr__(self):
        return repr("Rows:"+ str(self.rows) + "; Cols:"+ str(self.cols)+ "; Depth:"+ str(self.depth) + "; Split:"+ str(self.split_on) + " HeuristicValue:"+ str(self.heuristicValue) + " Class:"+ str(set(self.data[:,-1])))
    
    def setHeuristicValue(self, heuristic):
        if heuristic == "Entropy":
            self.heuristicValue = CalcEntropy1(self.data)
        elif heuristic == "Variance":
            self.heuristicValue = CalcVariance(self.data)
        else:
            raise NotFoundError("Heuristic not found")
            
def Branch(node, heuristic = "Entropy", d_max = 100):
    """To partition a node recursively until it is pure or until it reaches max depth"""
    
    if node.heuristicValue is None:
        node.setHeuristicValue(heuristic)
        
    if node.heuristicValue > 0:
        node.classifier = bool(np.argmax(np.unique(node.data[:,-1], return_counts = True)[1]))
        node.split_on, leftdata, rightdata = GetBestFeatureToSplit(node.data, node.avail_cols, node.heuristicValue, node.rows, heuristic)
        #print(node.split_on)
        if node.depth < d_max:
            if np.size(leftdata, 0) > 0:
                node.leftchild = Node(leftdata, node, None, None, None, node.path+"~"+str(node.split_on), node.depth+1)
                Branch(node.leftchild, heuristic, d_max)
                
            if np.size(rightdata, 0) > 0:
                node.rightchild = Node(rightdata, node, None, None, None, node.path+str(node.split_on), node.depth+1)
                Branch(node.rightchild, heuristic, d_max)
        return
    
    else:
        node.classifier = np.unique(node.data[:,-1])[0]
        return

def Test(node, test_data):
    """To test the accuracy of the traned decision tree.
    Returns the data with predicted column added at the end and the accuracy percentage"""
    
    if node:
        if node.leftchild or node.rightchild:
            if node.depth == 0:
                return (Test(node.leftchild, test_data[test_data[:,node.split_on] == False]) + Test(node.rightchild, test_data[test_data[:,node.split_on] == True]))*100/node.rows
            else:
                return Test(node.leftchild, test_data[test_data[:,node.split_on] == False]) + Test(node.rightchild, test_data[test_data[:,node.split_on] == True])
        else:
            return sum(test_data[:,-1] == node.classifier)
    return 0

def Prune(node, validation_data, algo = "REP", max_depth = 100):
    """Prune the tree to increase accuracy based on validation data"""
    if algo == "DBP":
        if node:
            if node.leftchild or node.rightchild:
                if node.depth < max_depth:
                    Prune(node.leftchild, None, "DBP", max_depth)
                    Prune(node.rightchild, None, "DBP", max_depth)
                else:
                    node.leftchild = None
                    node.rightchild = None
    else:
        if node:
            if node.leftchild or node.rightchild:
                children_accuracy = Prune(node.leftchild, validation_data[validation_data[:,node.split_on] == False], "REP", None) + Prune(node.rightchild, validation_data[validation_data[:,node.split_on] == True], "REP", None)
                self_accuracy = sum(validation_data[:,-1] == node.classifier)

                if children_accuracy >= self_accuracy:
                    return children_accuracy
                else:
                    node.leftchild = None
                    node.rightchild = None
                    return self_accuracy

            else:
                return sum(validation_data[:,-1] == node.classifier)
        return 0
    

if __name__ == '__main__':
    import time
    import sys
    t_1=time.time()
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
    if sys.argv[1] == "-f":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        file_mid = "c1800_d5000"
        data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        rfc = RandomForestClassifier(n_estimators=10)
        rfc.fit(data[:,0:-1],data[:,-1])
        print("Accuracy of Random Forest", str(accuracy_score(test_data[:,-1], rfc.predict(test_data[:,0:-1]))*100))
    
        file_mid = "c1800_d5000"
        heuristic = "Entropy"
        data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        heuristic = "Variance"
        #data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        #test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        #valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        file_mid = "c1800_d1000"
        heuristic = "Entropy"
        data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        heuristic = "Variance"
        #data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        #test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        #valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        file_mid = "c1800_d100"
        heuristic = "Entropy"
        data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        heuristic = "Variance"
        #data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        #test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        #valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        file_mid = "c1500_d5000"
        heuristic = "Entropy"
        data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        heuristic = "Variance"
        #data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        #test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        #valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        file_mid = "c1500_d1000"
        heuristic = "Entropy"
        data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        heuristic = "Variance"
        #data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        #test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        #valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        file_mid = "c1500_d100"
        heuristic = "Entropy"
        data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        heuristic = "Variance"
        #data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        #test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        #valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        file_mid = "c1000_d5000"
        heuristic = "Entropy"
        data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        heuristic = "Variance"
        #data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        #test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        #valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        file_mid = "c1000_d1000"
        heuristic = "Entropy"
        data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        heuristic = "Variance"
        #data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        #test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        #valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        file_mid = "c1000_d100"
        heuristic = "Entropy"
        data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        heuristic = "Variance"
        #data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        #test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        #valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        file_mid = "c500_d5000"
        heuristic = "Entropy"
        data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        heuristic = "Variance"
        #data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        #test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        #valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        file_mid = "c500_d1000"
        heuristic = "Entropy"
        data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        heuristic = "Variance"
        #data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        #test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        #valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        file_mid = "c500_d100"
        heuristic = "Entropy"
        data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        heuristic = "Variance"
        #data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        #test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        #valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        file_mid = "c300_d5000"
        heuristic = "Entropy"
        data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        heuristic = "Variance"
        #data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        #test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        #valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        file_mid = "c300_d1000"
        heuristic = "Entropy"
        data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        heuristic = "Variance"
        #data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        #test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        #valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        file_mid = "c300_d100"
        heuristic = "Entropy"
        data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        
        heuristic = "Variance"
        #data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        #test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy before pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        #valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        Prune(root, valid_data, "REP", None)
        #print("Accuracy after REP pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        root=Node(data)
        Branch(root, heuristic)
        Prune(root, None, "DBP", 15)
        #print("Accuracy after DBP15 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 10)
        #print("Accuracy after DBP10 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        Prune(root, None, "DBP", 5)
        #print("Accuracy after DBP5 pruning", str(Test(root, test_data)))
        print("", str(Test(root, test_data)))
        #root=Node(data)
        #Branch(root, heuristic, depth)
        #test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        #print("Accuracy after pruning", str(Test(root, test_data)))

    elif sys.argv[1] == "-D":
        print("Building Decision Tree..")
        file_mid = sys.argv[2]
        print("Training File:", 'train_'+ file_mid +'.csv')
        print("Validation File:", 'valid_'+ file_mid +'.csv')
        print("Test File:", 'test_'+ file_mid +'.csv')
        if sys.argv[3] == "-v":
            heuristic = "Variance"
        else:
            heuristic = "Entropy"
        print("Heuristic:", heuristic)
        if sys.argv[4] == "-n":
            pruning = "No Pruning"
        elif sys.argv[4] == "-d":
            depth = int(sys.argv[5])
            pruning = "Depth Based Pruning"
        elif sys.argv[4] == "-r":
            pruning = "Reduced Error Pruning"
        else:
            pruning = "No Pruning"
        print("Pruning:", pruning, "with depth="+str(depth) if sys.argv[4] == "-d" else "")

        data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        root=Node(data)
        Branch(root, heuristic)
        test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        print("Accuracy before pruning", str(Test(root, test_data)))
        if pruning == "Reduced Error Pruning":
            valid_data=np.loadtxt('all_data/valid_'+ file_mid +'.csv', delimiter=',', dtype=bool)
            Prune(root, valid_data, "REP", None)
            print("Accuracy after pruning", str(Test(root, test_data)))
        elif pruning == "Depth Based Pruning":
            Prune(root, None, "DBP", depth)
            print("Accuracy after pruning", str(Test(root, test_data)))

    elif sys.argv[1] == "-R":
        print("Building Random Forest..")
        file_mid = sys.argv[2]
        print("Training File:", 'train_'+ file_mid +'.csv')
        print("Test File:", 'test_'+ file_mid +'.csv')
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        data=np.loadtxt('all_data/train_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        test_data=np.loadtxt('all_data/test_'+ file_mid +'.csv', delimiter=',', dtype=bool)
        rfc = RandomForestClassifier(n_estimators=10)
        rfc.fit(data[:,0:-1],data[:,-1])
        print("Accuracy of Random Forest", str(accuracy_score(test_data[:,-1], rfc.predict(test_data[:,0:-1]))*100))
    t0=time.time()
    print("Completed in", str(t0-t_1)+"s")


# In[15]:


t0=time.time()
for i in range(1):
    pass
    #np.unique(root.data[:,-1])
t1=time.time()
for i in range(1):
    pass
    #set(root.data[:,-1])
t2=time.time()

def Prune(node, validation_data):
    """Prune the tree to increase accuracy based on validation data"""
    
    if node:
        if node.leftchild or node.rightchild:
            children_accuracy = Prune(node.leftchild, validation_data[validation_data[:,node.split_on] == False]) + Prune(node.rightchild, validation_data[validation_data[:,node.split_on] == True])
            self_accuracy = sum(validation_data[:,-1] == node.classifier)
            
            if children_accuracy >= self_accuracy:
                return children_accuracy
            else:
                node.leftchild = None
                node.rightchild = None
                return self_accuracy
            
        else:
            return sum(validation_data[:,-1] == node.classifier)
    return 0
#print(t1-t0)
#print(t2-t1)
#(base) H:\Assignments\HW due on 16 sept 19>python runscript.py -D c1800_d5000 -e -d 12


# In[ ]:




