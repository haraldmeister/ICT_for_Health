import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class SolveMinProbl:
    """
    Parameters
    ----------
    y : vector of floats
        column vector of feature F0
    A : matrix Np*Nf of floats
        matrix containing all features except F0
    y_val : vector y used for validation
    B :  X matrix used for validation
    C :  X matrix used for test
    y_test : vector used for test
    mean :  Mean value of the feature column
    std :   Standard deviation of the feature column.
    """
    def __init__(self,A,B,C,y,y_val,y_test,mean,std):
        self.matr=A
        self.matr2=B
        self.matr3=C
        self.Np_train=A.shape[0] # Number of patients
        self.Nf_train=A.shape[1] # Number of features
        self.Np_val=B.shape[0]
        self.Nf_val=B.shape[1]
        self.Np_test=C.shape[0]
        self.Nf_test=C.shape[1]
        self.vect=y
        self.sol=np.zeros((self.Nf_train,1),dtype=float) #Array of the solution
        self.ytest=y_test
        self.yval=y_val
        self.yhat_test=np.zeros((y_test.shape[0])) #Array of the estimation yhat_test
        self.yhat_train=np.zeros((y.shape[0])) #Array of the estimation yhat_train
        self.err=np.zeros((1,3),dtype=float)
        self.mean=mean
        self.std=std
        return
    def plot_w(self,title="Solution"):
        """It plots the coefficients w .
        """
        w=self.sol
        n=np.arange(self.Nf_train)
        plt.figure()
        plt.stem(n,w)
        plt.xlabel("Feature")
        plt.ylabel("Weight w")
        plt.title(title)
        plt.grid()
        str=title+'.pdf'
        plt.savefig(str, bbox_inches='tight')
        plt.close()
        return
    def plot_yhat_y_train(self,title="Solution",mean_value=0,std_value=0):
        """Plot the yhat_train vs y_train graph"""
        w=self.sol
        self.yhat_train=np.dot(self.matr,w).reshape(self.Np_train,1)
        yhat_train_denorm=(self.yhat_train*std_value)+mean_value
        y_train_denorm=(self.vect*std_value)+mean_value

        plt.figure()
        plt.scatter(y_train_denorm,yhat_train_denorm,s=3)
        plt.plot(y_train_denorm,y_train_denorm,color='black')

        plt.ylabel("yhat Train")
        plt.xlabel("y Train")
        plt.title(title)
        plt.grid()

        str=title+'.pdf'
        plt.savefig(str, bbox_inches='tight')
        plt.close()
        return
    def plot_yhat_y_test(self,title="Solution",mean_value=0.0,std_value=0.0):
        """Plot the yhat_test vs y_test graph"""
        w=self.sol
        self.yhat_test=np.dot(self.matr3,w).reshape(self.Np_test,1)
        y_test_denorm=(self.ytest*std_value)+mean_value
        yhat_test_denorm=(self.yhat_test*std_value)+mean_value

        plt.figure()
        plt.scatter(y_test_denorm,yhat_test_denorm,s=3)
        plt.plot(y_test_denorm,y_test_denorm,color='black')

        plt.ylabel("yhat Test")
        plt.xlabel("y Test")
        plt.title(title)
        plt.grid()

        str=title+'.pdf'
        plt.savefig(str, bbox_inches='tight')
        plt.close()
        return
    def print_result(self,title):
        """It prints the w vector and the minimum Mean Square Error on screen.
        """
        print(title,' :')
        print("The optimal solution is:"," ",self.sol)
        print("The minimum error is:",self.min)
        return
    def plot_err(self, title='Square error',logy=0,logx=0):
        """It plots the MSE in log or linear scales of training,
        validation and test set."""
        err=self.err
        plt.figure()
        if(logy==0) & (logx==0):
            plt.plot(err[:,0],err[:,1])
            plt.plot(err[:,0],err[:,2])
            plt.plot(err[:,0],err[:,3])
        if(logy==1)& (logx==0):
            plt.semilogy(err[:,0],err[:,1])
            plt.semilogy(err[:,0],err[:,2])
            plt.semilogy(err[:,0],err[:,3])
        if(logy==0)&(logx==1):
            plt.semilogx(err[:,0],err[:,1])
            plt.semilogx(err[:,0],err[:,2])
            plt.semilogx(err[:,0],err[:,3])
        if(logy==1)&(logx==1):
            plt.loglog(err[:,0],err[:,1])
            plt.loglog(err[:,0],err[:,2])
            plt.loglog(err[:,0],err[:,3])
        plt.xlabel('Iteration n')
        plt.ylabel('Mean Square Error e(n)')
        plt.title(title)
        plt.legend(('Training Set','Validation Set','Test Set'),loc='upper left')
        plt.margins(0.01,0.1)
        plt.xlim(0,300)
        plt.grid()
        str=title+'.pdf'
        plt.savefig(str, bbox_inches='tight')
        plt.close()
        return
    def plot_err_conjugate(self, title='Square error',logy=0,logx=0):
        """Plot the MSE for Conjugate Algorithm"""
        err=self.err
        plt.figure()
        if(logy==0) & (logx==0):
            plt.plot(err[:,0],err[:,1])
            plt.plot(err[:,0],err[:,2])
            plt.plot(err[:,0],err[:,3])
        if(logy==1)& (logx==0):
            plt.semilogy(err[:,0],err[:,1])
            plt.semilogy(err[:,0],err[:,2])
            plt.semilogy(err[:,0],err[:,3])
        if(logy==0)&(logx==1):
            plt.semilogx(err[:,0],err[:,1])
            plt.semilogx(err[:,0],err[:,2])
            plt.semilogx(err[:,0],err[:,3])
        if(logy==1)&(logx==1):
            plt.loglog(err[:,0],err[:,1])
            plt.loglog(err[:,0],err[:,2])
            plt.loglog(err[:,0],err[:,3])
        plt.xlabel('Iteration n')
        plt.ylabel('Mean Square Error e(n)')
        plt.title(title)
        plt.legend(('Training Set','Validation Set','Test Set'),loc='upper left')
        plt.margins(0.01,0.1)
        plt.grid()
        str=title+'.pdf'
        plt.savefig(str, bbox_inches='tight')
        plt.close()
        return
    def plot_err_ridge(self, title='Square error',logy=0,logx=0):
        """Plot MSE for Ridge Algorithm"""
        err=self.err
        plt.figure()
        if(logy==0) & (logx==0):
            plt.plot(err[:,0],err[:,1])
            plt.plot(err[:,0],err[:,2])
            plt.plot(err[:,0],err[:,3])
        if(logy==1)& (logx==0):
            plt.semilogy(err[:,0],err[:,1])
            plt.semilogy(err[:,0],err[:,2])
            plt.semilogy(err[:,0],err[:,3])
        if(logy==0)&(logx==1):
            plt.semilogx(err[:,0],err[:,1])
            plt.semilogx(err[:,0],err[:,2])
            plt.semilogx(err[:,0],err[:,3])
        if(logy==1)&(logx==1):
            plt.loglog(err[:,0],err[:,1])
            plt.loglog(err[:,0],err[:,2])
            plt.loglog(err[:,0],err[:,3])
        plt.xlabel('Lambda')
        plt.ylabel('Mean Square Error e($\lambda$)')
        plt.legend(('Training Set','Validation Set','Test Set'),loc='upper left')
        plt.title(title)
        plt.margins(0.01,0.1)
        plt.grid()
        str=title+'.pdf'
        plt.savefig(str, bbox_inches='tight')
        plt.close()
        return
    def plot_hist_train(self,title):
        """Plot the histogram for y_train-y_hat_train"""
        plt.hist(((self.vect*self.std)+self.mean)-((self.yhat_train*self.std)+self.mean),50)
        plt.title(title)
        plt.grid()
        plt.xlabel("y_train-yhat_train intervals")
        plt.ylabel("Number of occurrences")
        plt.ylim(ymin=0,ymax=400)
        plt.xlim(xmin=-15,xmax=15)
        str=title+'.pdf'
        plt.savefig(str, bbox_inches='tight')
        plt.close()
    def plot_hist_test(self,title):
        """Plot the histogram for y_train-y_hat_train"""
        plt.hist(((self.ytest*self.std)+self.mean)-((self.yhat_test*self.std)+self.mean),50)
        plt.title(title)
        plt.grid()
        plt.xlabel("y_test-yhat_test Intervals")
        plt.ylabel("Number of occurrences")
        plt.ylim(ymin=0,ymax=400)
        plt.xlim(xmin=-15,xmax=15)
        str=title+'.pdf'
        plt.savefig(str, bbox_inches='tight')
        plt.close()
"""Here are applied the algorithms applied for regression"""
class SolveLLS(SolveMinProbl):
    def run(self):
        A=self.matr
        B=self.matr2
        C=self.matr3
        ytest=self.ytest
        yval=self.yval
        y=self.vect
        mean=self.mean
        std=self.std
        self.err=np.zeros((1,4),dtype=float)

        w=np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),y)
        self.sol=w
        self.err[0,1]=np.linalg.norm((np.dot(A,w)*std+mean)-(y*std+mean))**2/self.Np_train
        self.err[0,2]=np.linalg.norm((np.dot(B,w)*std+mean)-(yval*std+mean))**2/self.Np_val
        self.err[0,3]=np.linalg.norm((np.dot(C,w)*std+mean)-(ytest*std+mean))**2/self.Np_test
        self.min=[min(self.err[:,1]),min(self.err[:,2]),min(self.err[:,3])]

class SolveGrad(SolveMinProbl):
    def run(self, gamma,Nit):
        self.err=np.zeros((Nit,4),dtype=float)
        A=self.matr
        B=self.matr2
        C=self.matr3
        ytest=self.ytest
        yval=self.yval
        y=self.vect
        mean=self.mean
        std=self.std
        w=np.random.rand(self.Nf_train,1)
        for it in range(Nit):
            grad=2*np.dot(A.T,(np.dot(A,w)-y))
            w=w-gamma*grad
            self.err[it,0]=it
            self.err[it,1]=np.linalg.norm((np.dot(A,w)*std+mean)-(y*std+mean))**2/self.Np_train
            self.err[it,2]=np.linalg.norm((np.dot(B,w)*std+mean)-(yval*std+mean))**2/self.Np_val
            self.err[it,3]=np.linalg.norm((np.dot(C,w)*std+mean)-(ytest*std+mean))**2/self.Np_test
            self.min=[min(self.err[:,1]),min(self.err[:,2]),min(self.err[:,3])]

            if self.err[-1,1] <= self.min[0]:
                self.sol = w
        #self.sol=w
        #self.min=self.err[it,1]

class SolveSteep(SolveMinProbl):
    def run(self,Nit):
        self.err=np.zeros((Nit,4),dtype=float)
        A=self.matr
        B=self.matr2
        C=self.matr3
        ytest=self.ytest
        yval=self.yval
        y=self.vect
        mean=self.mean
        std=self.std
        w=np.random.rand(self.Nf_train,1)
        for it in range(Nit):
            H=2*np.dot(A.T,A)#4*
            grad=2*np.dot(A.T,(np.dot(A,w)-y))
            w=w-grad*((np.linalg.norm(grad))**2/(np.dot(np.dot(grad.T,H),grad)))
            self.err[it,0]=it
            self.err[it,1]=np.linalg.norm((np.dot(A,w)*std+mean)-(y*std+mean))**2/self.Np_train
            self.err[it,2]=np.linalg.norm((np.dot(B,w)*std+mean)-(yval*std+mean))**2/self.Np_val
            self.err[it,3]=np.linalg.norm((np.dot(C,w)*std+mean)-(ytest*std+mean))**2/self.Np_test
            self.min=[min(self.err[:,1]),min(self.err[:,2]),min(self.err[:,3])]

            if self.err[-1,1] <= self.min[0]:
                self.sol = w

class SolveStochastic(SolveMinProbl):
    def run(self,gamma,Nit):
        self.err=np.zeros((Nit,4),dtype=float)
        A=self.matr
        B=self.matr2
        C=self.matr3
        ytest=self.ytest
        yval=self.yval
        std=self.std
        mean=self.mean
        y=self.vect
        w=np.random.rand(self.Nf_train,1)
        for it in range(Nit):
            for i in range(self.Np_train):
                grad_i=2*(np.dot(A[i,:],w)-y[i])*(A[i,:].reshape(len(A[i,:]),1))
                w = w - gamma * grad_i
            self.err[it,0]=it
            self.err[it,1]=np.linalg.norm((np.dot(A,w)*std+mean)-(y*std+mean))**2/self.Np_train
            self.err[it,2]=np.linalg.norm((np.dot(B,w)*std+mean)-(yval*std+mean))**2/self.Np_val
            self.err[it,3]=np.linalg.norm((np.dot(C,w)*std+mean)-(ytest*std+mean))**2/self.Np_test
            self.min=[min(self.err[:,1]),min(self.err[:,2]),min(self.err[:,3])]

            if self.err[-1,1] <= self.min[0]:
                self.sol = w

class SolveMinibatches(SolveMinProbl):
    def run(self,gamma,Nit,number_batches=8):
        self.err=np.zeros((Nit,4),dtype=float)
        A=self.matr
        B=self.matr2
        C=self.matr3
        ytest=self.ytest
        yval=self.yval
        y=self.vect
        w=np.random.rand(self.Nf_train,1)
        K1=np.shape(A)[0]
        K=int(K1/number_batches)
        grad_tot=0
        for it in range(Nit):
            k=0
            for minibatch in range(0,K):
                if(k+K>=np.shape(A)[0]):
                    X=A[range(k,np.shape(A)[0]),:]
                    yj=y[range(k,np.shape(A)[0]),:]

                else:
                    X=A[range(k,k+K),:]
                    yj=y[range(k,k+K),:]

                grad_i=2*(np.dot(np.dot(X.T,X),w)-np.dot(X.T,yj))
                w=w-gamma*grad_i
                k=k+K

            self.err[it,0]=it
            self.err[it,1]=(np.linalg.norm(np.dot(A,w)-y))**2/self.Np_train
            self.err[it,2]=(np.linalg.norm(np.dot(B,w)-yval))**2/self.Np_val
            self.err[it,3]=(np.linalg.norm(np.dot(C,w)-ytest))**2/self.Np_test
            self.min=min(self.err[:,1])

            if self.err[-1,1] <= self.min:
                self.sol = w

class SolveRidge(SolveMinProbl):
    def run(self):
        self.err=np.zeros((200,4),dtype=float)
        A=self.matr
        B=self.matr2
        C=self.matr3
        ytest=self.ytest
        yval=self.yval
        y=self.vect
        std=self.std
        mean=self.mean
        I = np.eye(self.Nf_train)

        i=0
        for lam in range(0,200):
            w = np.dot(np.dot(np.linalg.inv((np.dot(A.T, A) + lam*I)), A.T), y)
            self.err[i,0]=i
            self.err[i,1]=np.linalg.norm((np.dot(A,w)*std+mean)-(y*std+mean))**2/self.Np_train
            self.err[i,2]=np.linalg.norm((np.dot(B,w)*std+mean)-(yval*std+mean))**2/self.Np_val
            self.err[i,3]=np.linalg.norm((np.dot(C,w)*std+mean)-(ytest*std+mean))**2/self.Np_test
            self.min=[min(self.err[:,1]),min(self.err[:,2]),min(self.err[:,3])]

            if self.err[-1,2] <= self.min[2]:
                self.sol = w
            i+=1
        print("Lambda min: %d"  %np.argmin(self.err[:,2]))

class SolveConjugate(SolveMinProbl):
    def run(self):
        A=self.matr
        B=self.matr2
        C=self.matr3
        ytest=self.ytest
        yval=self.yval
        y=self.vect
        mean=self.mean
        std=self.std
        w=np.zeros((self.Nf_train,1),dtype=float)
        self.err=np.zeros((self.Nf_train,4),dtype=float)
        b=2*np.dot(A.T,y)
        d=b
        g=-b
        Q=2*np.dot(A.T,A)
        for it in range(self.Nf_train):
            alpha=-(np.dot(d.T,g)/np.dot(np.dot(d.T,Q),d))
            w=w+alpha*d
            g=g+alpha*(np.dot(Q,d))
            beta=np.dot(np.dot(g.T,Q),d)/np.dot(np.dot(d.T,Q),d)
            d=-g+beta*d
            self.err[it,0]=it
            self.err[it,1]=np.linalg.norm((np.dot(A,w)*std+mean)-(y*std+mean))**2/self.Np_train
            self.err[it,2]=np.linalg.norm((np.dot(B,w)*std+mean)-(yval*std+mean))**2/self.Np_val
            self.err[it,3]=np.linalg.norm((np.dot(C,w)*std+mean)-(ytest*std+mean))**2/self.Np_test
            self.min=[min(self.err[:,1]),min(self.err[:,2]),min(self.err[:,3])]

            if self.err[-1,1] <= self.min[0]:
                self.sol = w