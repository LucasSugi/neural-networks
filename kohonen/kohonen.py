'''
Author: Lucas Yudi Sugi - 9293251
Reference: http://www.computacaointeligente.com.br/algoritmos/mapas-auto-organizaveis-som/
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class KOHONEN():

    def __init__(self,data,grid,alpha=0.1,sigma=None):
        self.data = np.array(data)
        self.grid = grid
        self.alpha0 = alpha
        if(sigma == None):
            self.sigma0 = max(grid) / 2.0
        else:
            self.sigma0 = sigma
        self.w = np.random.uniform(-0.5,0.5,[grid[0],grid[1],self.data.shape[1]])

    def train(self,iteration,verbose=False):
        #Time constant
        time = iteration/np.log(self.sigma0)
        
        #Run for each iteration
        for epoch in range(iteration):
            if(verbose and epoch % 10 == 0):
                print('Epoch:',epoch)

            #Computes alpha and sigma
            alpha = self.alpha0 * np.exp(-epoch/time)
            sigma = self.sigma0 * np.exp(-epoch/time)

            #Run for each sample
            for p in range(self.data.shape[0]):
                #Matrix of all distances
                dist = self.distance(self.data[p,:],self.w)

                #Get the winner's position
                winPos = self.winnerPos(dist)
                
                #Update weights for other nodes
                for i in range(self.w.shape[0]):
                    for j in range(self.w.shape[1]):
                        #Compute distance between winner and some node
                        dNode = self.distanceNode([i,j],winPos)

                        #Winner influence
                        h = np.exp(-dNode/(2*np.power(sigma,2)))

                        #Update weights
                        self.w[i,j,:] += (alpha * h * (self.data[p,:] - self.w[i,j,:]))

    def distance(self,a,b):
        return np.sqrt(np.sum(np.power(a-b,2),2,keepdims=True))

    def winnerPos(self,dist):
        pos = dist.argmin()
        return pos//dist.shape[1], pos%dist.shape[1]

    def distanceNode(self,a,b):
        a = np.array(a)
        b = np.array(b)
        return np.sum(np.power(a-b,2))

    def save(self,path):
        np.save(path,self.w)

    def load(self,path):
        self.w = np.load(path)

    def show(self,target):
        target = np.array(target)
        img = np.ones([self.grid[0],self.grid[1]])

        #Crete image to plot
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                dist = np.sqrt(np.sum(np.power(self.w[i,j]-self.data,2),axis=1))
                img[i,j] = target[dist.argmin()]
        im = plt.imshow(img)
        
        #Get colors from imshow
        label = np.unique(target)
        colors = im.cmap(im.norm(label))
        
        #Create legend
        patches = [ mpatches.Patch(color=colors[i], label="Class {l}".format(l=label[i])) for i in range(len(colors)) ]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

        #Plot image
        plt.show()
