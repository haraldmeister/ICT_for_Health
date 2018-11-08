import numpy as np
import matplotlib.pyplot as plt

class SolveMinProbl:
    def __init__(self,A=np.eye(3),y=np.ones((3,1))):
        self.matr=A
        self.Np=y.shape[0]
        self.Nf=A.shape[1]
        self.vect=y
        self.sol=np.zeros((self.Nf,1),dtype=float)
        return
    def plot_w(self,title="Solution"):
        w=self.sol
        n=np.arange(self.Nf)
        plt.figure()
        plt.plot(n,w)
        plt.xlabel("n")
        plt.ylabel("w")
        plt.title(title)
        plt.grid()
        str=title+'.pdf'
        plt.savefig(str, bbox_inches='tight')
        plt.close()
        return
    def print_result(self,title):
        print(title,' :')
        print("The optimal solution is:"," ",self.sol)
        return
    def plot_err(self, title='Square error',logy=0,logx=0):
        err=self.err
        plt.figure()
        if(logy==0) & (logx==0):
            plt.plot(err[:,0],err[:,1])
        if(logy==1)& (logx==0):
            plt.semilogy(err[:,0],err[:,1])
        if(logy==0)&(logx==1):
            plt.semilogx(err[:,0],err[:,1])
        if(logy==1)&(logx==1):
            plt.loglog(err[:,0],err[:,1])
        plt.xlabel('n')
        plt.ylabel('e(n)')
        plt.title(title)
        plt.margins(0.01,0.1)
        plt.grid()
        str=title+'.pdf'
        plt.savefig(str, bbox_inches='tight')
        plt.close()
        return

class SolveLLS(SolveMinProbl):
    def run(self):
        A=self.matr
        y=self.vect
        w=np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),y)
        self.sol=w
        self.min=np.linalg.norm(np.dot(A,w)-y)

class SolveGrad(SolveMinProbl):
    def run(self, gamma=1e-3,Nit=100):
        self.err=np.zeros((Nit,2),dtype=float)
        A=self.matr
        y=self.vect
        w=np.random.rand(self.Nf,1)
        for it in range(Nit):
            grad=2*np.dot(A.T,(np.dot(A,w)-y))
            w=w-gamma*grad
            self.err[it,0]=it
            self.err[it,1]=np.linalg.norm(np.dot(A,w)-y)
        self.sol=w
        self.min=self.err[it,1]

class SolveSteep(SolveMinProbl):
    def run(self,Nit=100):
        self.err=np.zeros((Nit,2),dtype=float)
        A=self.matr
        y=self.vect
        w=np.random.rand(self.Nf,1)
        for it in range(Nit):
            H=2*np.dot(A.T,A)#4*
            grad=2*np.dot(A.T,(np.dot(A,w)-y))
            w=w-grad*((np.linalg.norm(grad))**2/(np.dot(np.dot(grad.T,H),grad)))
            self.err[it,0]=it
            self.err[it,1]=np.linalg.norm(np.dot(A,w)-y)
        self.sol=w
        self.min=self.err[it,1]

class SolveStochastic(SolveMinProbl):
    def run(self,gamma=1e-3,Nit=500):
        self.err=np.zeros((Nit,2),dtype=float)
        A=self.matr
        y=self.vect
        w=np.random.rand(self.Nf,1)
        for it in range(Nit):
            for i in range(Np):
                grad_i=2*(np.dot(A[i,:],w)-y[i])*(A[i,:].reshape(len(A[i,:]),1))
                w = w - gamma * grad_i
            self.err[it,0]=it
            self.err[it,1]=np.linalg.norm(np.dot(A,w)-y)
        self.sol=w
        self.min=self.err[it,1]

class SolveMinibatches(SolveMinProbl):
    def run(self,gamma=1e-3,Nit=500,number_batches=8):
        self.err=np.zeros((Nit,2),dtype=float)
        A=self.matr
        y=self.vect
        w=np.random.rand(self.Nf,1)
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
            self.err[it,1]=np.linalg.norm(np.dot(A,w)-y)
        self.sol=w
        self.min=self.err[it,1]



plt.close('all')
Np=80
Nf=4
gamma=1e-3
Nit= 500
logx=0
logy=1

# A=np.random.randn(Np,Nf)
# print(A)
# print("\n")
#
# w_id=np.random.randn(Nf)
# print(w_id)
# print("\n")
#
# y=np.dot(A,w_id)
# print(y)
# print("\n")
#
# # AAt=np.dot(A.T,A)
# # print(AAt)
# # print("\n")
# # AAtinv=np.linalg.inv(AAt)
# # print(AAtinv)
# # print("\n")
# # AAtinvAT=np.dot(AAtinv,A.T)
# # print(AAtinvAT)
# # print("\n")
# # w=np.dot(AATinvAt,y)
# # print(w)
# # print("\n")
#
# w=np.dot(np.linalg.pinv(A),y)
# print(w)

np.random.seed(7)
A=np.random.randn(Np,Nf)
y=np.random.randn(Np,1)
m=SolveLLS(A,y)
m.run()
m.print_result('Linear Least Square: MSE')
m.plot_w('Linear Least Square')

g=SolveGrad(A,y)
g.run(gamma,Nit)
g.print_result('Gradient Algorithm:')
g.plot_w('Gradient Algorithm MSE')
g.plot_err('Gradient algorithm square error',logy,logx)

h=SolveSteep(A,y)
h.run(Nit)
h.print_result('Steepest Descent:')
h.plot_w('Steepest Descent MSE')
h.plot_err('Steepest Descent square error',logy,logx)

h2=SolveStochastic(A,y)
h2.run()
h2.print_result('Stochastic gradient')
h2.plot_w('Stochastic Gradient MSE')
h2.plot_err('Stochastic Gradient square error',logy,logx)

h3=SolveMinibatches(A,y)
h3.run(gamma,Nit,8)
h3.print_result('Gradient Algorithm Minibatches')
h3.plot_w('Gradient Algorithm Minibatches MSE')
h3.plot_err('Gradient Algorithm Minibatches square error',logy,logx)
