import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from minimization import *


np.random.seed(15) #Setting the random seed
x=pd.read_csv("parkinsons_updrs.data") #Load the data
x.info() #Showing the information about the dataset
data=x.values
(a,b)=data.shape #Finding the dimensions of the dataset

np.random.shuffle(data) #Shuffle the lines of the dataset
data1=data[:,4:21] #Delete the columns from 0 to 3
(row,column)=data1.shape
Np,Nf=row,column
# Divide the dataset in training, validation and test set
data_train=data1[0:int(row/2),:]
data_validation=data1[int(row*0.5):int(row*0.75),:]
data_test=data1[int(row*0.75):row,:]

print(data_train.shape)
print(data_validation.shape)
print(data_test.shape)

#Evaluate the mean and standard deviation of the feature column and
# reshape the matrix and standardize the data
mean_train=(np.mean(data_train,0)).reshape(1,Nf)
std_train=np.std(data_train,0).reshape(1,Nf)
norm_data_train=(data_train-mean_train)/std_train
norm_data_validation=(data_validation-mean_train)/std_train
norm_data_test=(data_test-mean_train)/std_train

print("Mean Training Set columns:\n")
print(np.mean(norm_data_train,0))
print("Standard deviation Training Set columns:\n")
print(np.var(norm_data_train,0))

# Delete the feature column to regress from the matrixes
F0=1 #1=Total_UPDRS 7=Shimmer
mean=mean_train[0,F0]
std=std_train[0,F0]
y_train=norm_data_train[:,F0]
X_train=np.delete(norm_data_train,F0,1)
y_val=norm_data_validation[:,F0]
X_val=np.delete(norm_data_validation,F0,1)
y_test=norm_data_test[:,F0]
X_test=np.delete(norm_data_test,F0,1)

y_train=y_train.reshape(y_train.shape[0],1)
y_val=y_val.reshape(y_val.shape[0],1)
y_test=y_test.reshape(y_val.shape[0],1)

logx=0
logy=0
Nit=10000
gamma=1e-7

#Performing regression with algorithms defined in minimization.py
m=SolveLLS(X_train,X_val,X_test,y_train,y_val,y_test,mean,std)
m.run()
m.print_result('Total UPDRS Linear Least Square Weight w')
m.plot_w('Total UPDRS Weight w LLS')
m.plot_yhat_y_train("Total UPDRS yhat_train vs y_train LLS",mean,std)
m.plot_yhat_y_test("Total UPDRS yhat_test vs y_test LLS",mean,std)
m.plot_hist_train("Total UPDRS Histogram y_train-yhat_train LLS")
m.plot_hist_test("Total UPDRS Histogram y_test-yhat_test LLS")

m1=SolveGrad(X_train,X_val,X_test,y_train,y_val,y_test,mean,std)
m1.run(gamma,Nit)
m1.print_result("Total UPDRS Gradient Algorithm Weight w")
m1.plot_w("Total UPDRS Weight w Gradient Algorithm")
m1.plot_err("Total UPDRS MSE Gradient Algorithm")
m1.plot_yhat_y_train("Total UPDRS yhat_train vs y_train Gradient Algorithm",mean,std)
m1.plot_yhat_y_test("Total UPDRS yhat_test vs y_test Gradient Algorithm",mean,std)
m1.plot_hist_train("Total UPDRS Histogram y_train-yhat_train Gradient Algorithm")
m1.plot_hist_test("Total UPDRS Histogram y_test-yhat_test Gradient Algorithm")

m2=SolveSteep(X_train,X_val,X_test,y_train,y_val,y_test,mean,std)
m2.run(Nit)
m2.print_result("Total UPDRS Steepest Descent Weight w")
m2.plot_w("Total UPDRS Weight w Steepest Descent Algorithm")
m2.plot_err("Total UPDRS MSE Steepest Descent Algorithm")
m2.plot_yhat_y_train("Total UPDRS yhat_train vs y_train Steepest Descent",mean,std)
m2.plot_yhat_y_test("Total UPDRS yhat_test vs y_test Steepest Descent",mean,std)
m2.plot_hist_train("Total UPDRS Histogram y_train-yhat_train Steepest Descent")
m2.plot_hist_test("Total UPDRS Histogram y_test-yhat_test Steepest Descent")

m3=SolveStochastic(X_train,X_val,X_test,y_train,y_val,y_test,mean,std)
m3.run(gamma,Nit)
m3.print_result("Total UPDRS Stochastic Gradient Weight w")
m3.plot_w("Total UPDRS Weight w Stochastic Gradient Algorithm")
m3.plot_err("Total UPDRS MSE Stochastic Gradient Algorithm")
m3.plot_yhat_y_train("Total UPDRS yhat_train vs y_train Stochastic Gradient",mean,std)
m3.plot_yhat_y_test("Total UPDRS yhat_test vs y_test Stochastic Gradient",mean,std)
m3.plot_hist_train("Total UPDRS Histogram y_train-yhat_train Stochastic Gradient")
m3.plot_hist_test("Total UPDRS Histogram y_test-yhat_test Stochastic Gradient")

m4=SolveRidge(X_train,X_val,X_test,y_train,y_val,y_test,mean,std)
m4.run()
m4.print_result("Total UPDRS Ridge Algorithm Weight w")
m4.plot_w("Total UPDRS Weight w Ridge Algorithm")
m4.plot_err_ridge("Total UPDRS MSE Ridge Algorithm on Lambda variation")
m4.plot_yhat_y_train("Total UPDRS yhat_train vs y_train Ridge Algorithm",mean,std)
m4.plot_yhat_y_test("Total UPDRS yhat_test vs y_test Ridge Algorithm",mean,std)
m4.plot_hist_train("Total UPDRS Histogram y_train-yhat_train Ridge Algorithm")
m4.plot_hist_test("Total UPDRS Histogram y_test-yhat_test Ridge Algorithm")

m5=SolveConjugate(X_train,X_val,X_test,y_train,y_val,y_test,mean,std)
m5.run()
m5.print_result("Total UPDRS Conjugate Algorithm Weight w")
m5.plot_w("Total UPDRS Weight w Conjugate Algorithm")
m5.plot_err_conjugate("Total UPDRS MSE Conjugate Algorithm")
m5.plot_yhat_y_train("Total UPDRS yhat_train vs y_train Conjugate Algorithm",mean,std)
m5.plot_yhat_y_test("Total UPDRS yhat_test vs y_test Conjugate Algorithm",mean,std)
m5.plot_hist_train("Total UPDRS Histogram y_train-yhat_train Conjugate Algorithm")
m5.plot_hist_test("Total UPDRS Histogram y_test-yhat_test Conjugate Algorithm")


