import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

feature=['age','bp','sg','al','su','rbc',	'pc','pcc','ba','bgr','bu','sc','sod',
         'pot','hemo'	,'pcv','wc','rc','htn','dm','cad','appet','pe','ane','class']
key_list=['normal','abnormal','present','notpresent','yes','no','poor','good','ckd',
          'notckd','ckd\t','\tno','yes','\tyes',' yes']
key_val=[0.0,1.0,1.0,0.0,1.0,0.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0,1.0,1.0]#[0,1,0,1,0,1,0,1,1,0,1,1,0,0]
#Import and prepare data
data=pd.read_csv("chronic_kidney_disease.arff",sep=',',skiprows=29,na_values=['?','\t','?\t','\t?'],names=feature)#data
data_copy=data.copy() #data_copy
data_polish=data_copy.replace(key_list,key_val)
data_polish=data_polish.astype(float)

X=data_polish.dropna()


for val in range(0,5):
    # All patients having val Nan are selected
    # All patients having no Nan are dropped
    x_one_nan=data_polish.dropna(thresh=(24-val))
    index_row_full_feature=(x_one_nan.dropna()).index
    x_one_nan=x_one_nan.drop(index_row_full_feature)

    # Indexing position of Nan in a
    # Left column ---> Number of patient
    # Right column ---> Feature missing
    idx,idy=np.where(pd.isnull(x_one_nan))
    a=np.column_stack([x_one_nan.index[idx],x_one_nan.columns[idy]])

    # Grouping all missing features for the same patient
    a1=[]
    a2=[]
    for i in range(len(a)):
        if i==len(a)-1:
            a1.append(a[i,0])
            break
        if(a[i+1,0]!=a[i,0]):
            a1.append(a[i,0])
    v=[]
    for i in range(len(a)):
        if i==len(a)-1:
            v.append(a[i,1])
            a2.append(v)
            break
        v.append(a[i,1])
        if(a[i+1,0]!=a[i,0]):
            a2.append(v)
            v=[]

    # Grouping all the patients having a given missing feature
    a_group=((pd.DataFrame(a)).groupby(a[:,1])).groups

    # For each missing feature, all the patients are taken
    for col,row in a_group.items():
        name_col=col #Name of the feature missing
        pat_row=list(a[row,0]) #List of patients having the name_col feature missing
        list_row=[] # List of indexes of the patients in a1 list
        for i in pat_row:
            list_row.append(a1.index(i))
        index_col=X.columns.get_loc(name_col)

        x_one_nan1=x_one_nan.copy()
        x_one_nan_np=(x_one_nan1.values).astype(float)

        # Each patient having the missing feature is taken
        for i in pat_row:
            patient=pd.DataFrame(x_one_nan1.loc[i]).T
            miss_val_col=(patient).columns[patient.isnull().any()]
            patient_np=(patient.values).astype(float)
            # Indexes of the patient's missing feature
            ind_miss_val_col=[]
            j=0
            for k in miss_val_col:
                c=X.columns.get_loc(k)
                ind_miss_val_col.append(c)
                j+=1

            # Standardize data and apply Ridge Regression
            X_np=(X.values).astype(float)
            (row,column)=X_np.shape
            Np,Nf=row,column
            mean_X=np.mean(X_np,0).reshape(1,Nf)
            std_X=np.std(X_np,0).reshape(1,Nf)
            norm_X=(X_np-mean_X)/std_X
            norm_patient_np=(patient_np-mean_X)/std_X
            norm_patient_np=np.delete(norm_patient_np,ind_miss_val_col,1)

            mean=mean_X[0,index_col]
            std=std_X[0,index_col]

            y_train=norm_X[:,index_col].reshape(Np,1)
            ind0=np.array(index_col,dtype=int)
            ind1=np.array(ind_miss_val_col,dtype=int)
            ind=np.unique(np.sort(np.append(ind0,ind1)))
            X_train=np.delete(norm_X,ind,1)

            I = np.eye(Nf-len(ind))
            w = np.dot(np.dot(np.linalg.inv((np.dot(np.transpose(X_train), X_train) + 10*I)), np.transpose(X_train)), y_train)
            y_hat=(np.dot(norm_patient_np,w)*std)+mean
            # Estimated value is saved on the dataset
            data_polish.loc[i][col]=y_hat
            # Approximation of the data depending on the feature column
            if name_col=='age':
                data_polish[name_col]=data_polish[name_col].round(0)
            elif name_col=='bp':
                data_polish[name_col]=data_polish[name_col].round(0)
            elif name_col=='sg':
                if data_polish.loc[i][name_col]>=1.005 and data_polish.loc[i][name_col]<1.0075:
                    data_polish.loc[i][name_col]=1.005
                if data_polish.loc[i][name_col]>=1.0075 and data_polish.loc[i][name_col]<1.0125:
                    data_polish.loc[i][name_col]=1.010
                if data_polish.loc[i][name_col]>=1.0125 and data_polish.loc[i][name_col]<1.0175:
                    data_polish.loc[i][name_col]=1.015
                if data_polish.loc[i][name_col]>=1.0175 and data_polish.loc[i][name_col]<1.0225:
                    data_polish.loc[i][name_col]=1.020
                if data_polish.loc[i][name_col]>=1.0225 and data_polish.loc[i][name_col]<1.0275:
                    data_polish.loc[i][name_col]=1.025
            elif name_col=='al':
                data_polish[name_col]=data_polish[name_col].round(0)
            elif name_col=='su':
                data_polish[name_col]=data_polish[name_col].round(0)
            elif name_col=='rbc':
                data_polish[name_col]=data_polish[name_col].round(0)
            elif name_col=='pc':
                data_polish[name_col]=data_polish[name_col].round(0)
            elif name_col=='ba':
                data_polish[name_col]=data_polish[name_col].round(0)
            elif name_col=='pcc':
                data_polish[name_col]=data_polish[name_col].round(0)
            elif name_col=='bgr':
                data_polish[name_col]=data_polish[name_col].round(0)
            elif name_col=='bu':
                data_polish[name_col]=data_polish[name_col].round(0)
            elif name_col=='bgr':
                data_polish[name_col]=data_polish[name_col].round(0)
            elif name_col=='bu':
                data_polish[name_col]=data_polish[name_col].round(0)
            elif name_col=='sc':
                data_polish[name_col]=data_polish[name_col].round(1)
            elif name_col=='sod':
                data_polish[name_col]=data_polish[name_col].round(1)
            elif name_col=='pot':
                data_polish[name_col]=data_polish[name_col].round(1)
            elif name_col=='hemo':
                data_polish[name_col]=data_polish[name_col].round(1)
            elif name_col=='pcv':
                data_polish[name_col]=data_polish[name_col].round(0)
            elif name_col=='wc':
                data_polish[name_col]=data_polish[name_col].round(0)
            elif name_col=='rc':
                data_polish[name_col]=data_polish[name_col].round(3)
            elif name_col=='htn':
                data_polish[name_col]=data_polish[name_col].round(0)
            elif name_col=='dm':
                data_polish[name_col]=data_polish[name_col].round(0)
            elif name_col=='cad':
                data_polish[name_col]=data_polish[name_col].round(0)
            elif name_col=='appet':
                data_polish[name_col]=data_polish[name_col].round(0)
            elif name_col=='pe':
                data_polish[name_col]=data_polish[name_col].round(0)
            elif name_col=='ane':
                data_polish[name_col]=data_polish[name_col].round(0)
            elif name_col=='class':
                data_polish[name_col]=data_polish[name_col].round(0)


data_polish_complete=data_polish.dropna(thresh=20)

target=data_polish_complete['class']
data_polish_complete_2 = data_polish_complete.drop('class', 1)

clf = tree.DecisionTreeClassifier("entropy")
clf = clf.fit(data_polish_complete_2, target)

feature=feature.remove('class')

dot_data= tree.export_graphviz(clf,out_file="Tree.dot",feature_names=['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane'],class_names=['ckd','notckd'],filled=True,rounded=True,special_characters=True)
