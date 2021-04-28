import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class EM_Gaussian:

    def __init__(self, k=4, dim=2, tol=0.0005):
        self.dim = dim    # Dimension of samples 
        self.X = None     # Samples
        self.k = k        # Number of Gaussian models 
        self.N = 0        # Number of samples
        self.mu = None    # Average of each dimension
        self.gamma = None # Gamma of each model
        self.tol = tol

    def load_data(self, X_csv_path): 
        print("-"*20, "LOADING","-"*20)
        data = pd.read_csv(X_csv_path)
        self.X = np.mat(data.values.tolist())[:, 1:3]
        self.y = np.mat(data.values.tolist())[:, 3]
        self.N = self.X.shape[0]
        print("Data shape:", self.X.shape)

        # Random initialization
        self.alpha=[1/self.k for _ in range(self.k)]    # Mixture coefficient initialization
        self.gamma = np.zeros((self.N, self.k))     # Expectation that i-th data belongs to j-th model
        # self.gamma = np.random.random((self.N, self.k)) 
        self.mu = np.random.random((self.k, self.dim))
        # self.mu = np.ones((self.k, self.dim))
        # self.mu = self.X.mean(axis=0).repeat(self.k, axis=0)
        self.sigma = [np.mat(np.identity(self.dim)) for _ in range(self.k)]

    def generate_data(self, 
        sigma = [np.mat([[30, 0], [0, 30]]) for _ in range(4)], 
        N = 500,  # Number of sample
        mu1=[5,35], mu2=[30,40], mu3=[20,20], mu4=[45,15],
        alpha=[0.1,0.2,0.3,0.4]): # Mixture coefficient

        self.sigma = sigma
        self.X = np.mat(np.zeros((N, 2)))      # 初始化X，2行N列。2维数据，N个样本
        self.mu = np.mat(np.random.random((self.k,2)))
        self.N = N
        self.gamma = np.zeros((N,self.k))     # Expectation of i-th data belongs to j-th model
        self.alpha=[0.25,0.25,0.25,0.25]    # Mixture coefficient initialization
        for i in range(N):
            if np.random.random(1) < 0.1:  # 生成0-1之间随机数
                self.X[i,:]  = np.random.multivariate_normal(mu1, sigma, 1)     #用第一个高斯模型生成2维数据
            elif 0.1 <= np.random.random(1) < 0.3:
                self.X[i,:] = np.random.multivariate_normal(mu2, sigma, 1)      #用第二个高斯模型生成2维数据
            elif 0.3 <= np.random.random(1) < 0.6:
                self.X[i,:] = np.random.multivariate_normal(mu3, sigma, 1)      #用第三个高斯模型生成2维数据
            else:
                self.X[i,:] = np.random.multivariate_normal(mu4, sigma, 1)      #用第四个高斯模型生成2维数据
        # print("可观测数据：\n",self.X)       #输出可观测样本
        # print("初始化的mu1，mu2，mu3，mu4：",mu)      #输出初始化的mu

    def e_step(self, sigma, k, N):
        for i in range(N):
            denom = 0
            exp = lambda i,j: np.exp(- 0.5 * (self.X[i,:]-self.mu[j,:]) * sigma[j].I * \
                    (self.X[i,:]-self.mu[j,:]).T)
            for j in range(k):
                denom += self.alpha[j] * exp(i, j) / np.sqrt(np.linalg.det(sigma[j]))      
            for j in range(k):
                numer = exp(i, j) / np.sqrt(np.linalg.det(sigma[j]))       
                self.gamma[i,j] = self.alpha[j] * numer / denom     

    def m_step(self, k, N):
        for j in range(k):
            gamma, gamma_y, gamma_y_mu = 0, 0, 0   
            for i in range(N):
                gamma_y += self.gamma[i,j] * self.X[i, :]
                gamma_y_mu += self.gamma[i,j] * (self.X[i,:] - self.mu[j,:]).T * (self.X[i,:] - self.mu[j,:])
                gamma += self.gamma[i,j]
            self.mu[j,:] = gamma_y / gamma    
            self.alpha[j] = gamma / N    
            self.sigma[j] = gamma_y_mu / gamma   

    def fit(self, iter_num=1000):
        print("-"*20, "FITTING","-"*20)
        for i in range(iter_num):
            err, err_alpha = 0, 0    
            Old_mu = self.mu.copy()
            Old_alpha = self.alpha.copy()
            self.e_step(self.sigma, self.k, self.N)    
            self.m_step(self.k, self.N)          
            for z in range(self.k):
                err += (abs(Old_mu[z,0]-self.mu[z,0]) + abs(Old_mu[z,1]-self.mu[z,1]))     
                err_alpha += abs(Old_alpha[z] - self.alpha[z])
            print("Iteration:{}, err_mean:{:.5f}, err_alpha:{:.5f}".format(i+1, err, err_alpha))
            if (err<=self.tol) and (err_alpha<self.tol):  break
                       
    def visualization(self, show_img=False, save_img=False, show_3D=True):
        print("-"*20, "VISUALIZATION","-"*20)
        probability = np.zeros(self.N)  

        # Original data, c: scatter color，s: scatter size，alpha: transparent，marker: scatter shape
        plt.subplot(221)
        plt.scatter(self.X[:,0].tolist(), self.X[:,1].tolist(), c='b', s=25, alpha=0.4, marker='o')    
        plt.title('Random generated data')

        # Classified data
        plt.subplot(222)
        plt.title('Classified data through EM')
        order = np.argmax(self.gamma, axis=1) # which model X[i,:] belongs to
        color=['b','r','k','y']
        for i in range(self.N):  
            plt.scatter(self.X[i, 0].tolist(), self.X[i, 1].tolist(), c=color[int(order[i])], s=25, alpha=0.4, marker='o') 
        
        # Ground truth
        plt.subplot(223)
        plt.title('Ground truth')
        for i in range(self.N):
            plt.scatter(self.X[i, 0].tolist(), self.X[i, 1].tolist(), c=color[int(self.y[i,0])], s=25, alpha=0.4, marker='o')
        
        # 3D image
        if show_3D:
            ax = plt.subplot(224, projection='3d')
            plt.title('3d view')
            for i in range(self.N):
                for j in range(self.k):
                    probability[i] += self.alpha[int(order[i])] * np.exp(-(self.X[i,:]-self.mu[j,:]) * self.sigma[j].I \
                        * np.transpose(self.X[i,:]-self.mu[j,:])) / (np.sqrt(np.linalg.det(self.sigma[j])) * 2 * np.pi) 
            for i in range(self.N):
                ax.scatter(self.X[i, 0].tolist(), self.X[i, 1].tolist(), probability[i], c=color[int(order[i])])
        if save_img: plt.savefig("result.png")
        if show_img: plt.show()
        

if __name__ == '__main__':

    path = "data\GMM_EM_data_for_clustering.csv"

    model = EM_Gaussian(4)
    model.load_data(path)
    model.fit(iter_num=1000)
    model.visualization(save_img=True)