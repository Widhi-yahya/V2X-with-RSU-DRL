#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:34:54 2021

@author: widhi
"""
import random
from typing import List
import numpy as np
import math
from  Propagation import Propagation
from sklearn import preprocessing

huge = 10e5

class Environment:
    def __init__(self):
        # Basic environtment setup
        self.steps_left = 10
        self.total_reward = 0.0
        self.done = False
        
        # topology settings
        self.BVR = 100000
        self.BRV = 100000
        self.BRR = 100000
        self.BRA = 100000
        self.BAR = 100000
        self.rNum = 9
        self.gNum = 3
        self.vv = (30, 40)
        random.seed(0)
        self.vNum = [[random.randint(self.vv[0], self.vv[1]) for i in range(self.rNum)] for j in range(self.gNum)]
        # self.vNum = [[random.randint(40,50) for i in range(self.rNum)] for j in range(self.gNum)]
        self.lam = [[random.randint(20,30) for i in range(self.rNum)] for j in range(self.gNum)]
        self.propagationAN = Propagation("anmec.csv", self.gNum).propagation_delay

        # Arrival traffic rates
        np.random.seed(0)
        self.lamda = [[0 for i in range(self.rNum)]for j in range(self.gNum)]
        # self.lamda = np.zeros(self.rNum)
        for i in range(self.gNum):
            for j in range(self.rNum):
                self.lamda[i][j] = self.vNum[i][j] * self.lam[i][j]
        # print(self.lamda)
        
        # Array to collect latency (initial will 0) updated during calculation (1 means local exe)
        self.latency = [np.zeros((self.rNum,self.rNum+self.gNum)) for i in range(self.gNum)]
 
        self.R = [[0 for i in range(self.rNum)] for j in range(self.gNum)]
        self.A = [0 for i in range(self.gNum)]
    
    def InitmiuA(self):
        return [60000 for i in range(self.gNum)]
    
    def InitmiuR(self):
        return [[5000 for i in range(self.rNum)] for j in range(self.gNum)]    
    
    def get_observation(self) :
        latency = self.latency
        lamda = self.lamda
        BVR = self.BVR
        BRV = self.BRV
        BRR = self.BRR
        BRA = self.BRA
        BAR = self.BAR
        
        # Calculate residual capacity afterdeducted by arrival traffic at those site
        
                
        miuR = self.InitmiuR()
        for m in range(len(miuR)):
            for n in range(len(miuR[m])):
                miuR[m][n] -= self.R[m][n]
       
        miuA = self.InitmiuA()
        for k in range(len(miuA)):
            miuA[k] -= self.A[k]
            
       
        scaler = preprocessing.RobustScaler()
        ob = [latency, lamda, BVR, BRV, BRR, BRA, BAR, miuR, miuA]
        l = scaler.fit_transform(np.array(latency).flatten().reshape(-1,1))
        a = scaler.fit_transform(np.array(lamda).flatten().reshape(-1,1))
        b = scaler.fit_transform(np.array([BVR,BRV]).reshape(-1,1))
        c = scaler.fit_transform(np.array(BRR).flatten().reshape(-1,1))
        d = scaler.fit_transform(np.array([BRA,BAR]).reshape(-1,1))
        e = scaler.fit_transform(np.array(miuR).flatten().reshape(-1,1))
        f = scaler.fit_transform(np.array(miuA).flatten().reshape(-1,1))
       
        observe = np.concatenate((l,a,b,c,d,e,f)).flatten()
        return ob,l,observe 
        
    # def get_actions(self):
    #     b=[]
    #     # lamdax=[]
    #     for n in range(self.gNum):
    #         matrix = np.random.rand(self.rNum,self.rNum+self.gNum)
    #         b.append(matrix)
    #     # print("len b=", b)
    #     # xc
    #     array = np.array(b)
      
    #     return b, array.flatten()
    
    def get_actions(self):

        miuR = self.InitmiuR()
        for i in range(len(miuR)):
            for j in range(len(miuR[i])):
                miuR[i][j]-= self.R[i][j]
                
        miuAN = self.InitmiuA()
        for m in range(len(miuAN)):
            miuAN[m] -= self.A[m]

        act = [np.zeros((self.rNum, self.gNum+self.rNum)) for i in range(self.gNum)]
        for i in range(self.gNum):
            for j in range(self.rNum):
                tmp = np.concatenate((np.array(miuR[i]).flatten(), np.array(miuAN).flatten()))
                for k in range(len(tmp)):
                    act[i][j][k] = tmp[k]/sum(tmp)                    

        return act, np.array(act).flatten()
    
    # def get_actions(self):
        
    #     self.InitmiuA = [0000 for i in range(self.gNum)]
    #     self.InitmiuR = [[0000 for i in range(self.rNum)] for j in range(self.gNum)]
    #     self.InitmiuV = [[0000 for i in range(self.vNum)] for j in range(self.rNum)]
        
    #     miuV = self.InitmiuV
    #     for i in range(len(miuV)):
    #         for j in range(len(miuV[i])):
    #             miuV[i][j]-= self.V[i][j] 
                
    #     miuR = self.InitmiuR
    #     for m in range(len(miuR)):
    #         for n in range(leng(miuR[n]))
    #             miuR[m][n -= self.R[m][n]
       
    #     miuA = self.InitmiuA
    #         for o in range(len(miuA)):
    #             miuA[o] -= self.A[o]

    #     act = [np.zeros((self.vNum, self.rNum+self.vNum+self.gNum)) for i in range(self.rNum)]
    #     #act = [[np.zeros((self.rNum, self.gNum+self.rNum)) for i in range(self.gNum)] for j in range(self.rNum)]
    #     # print("act=1 ", act)
    #     # act2 = miuV
    #     for i in range(self.rNum):
    #         for j in range(self.vNum):
                
    #                 tmp = np.concatenate((np.array(miuV[i]).flatten(), np.array(miuR).flatten()))
    #                     for k in range(len(tmp)):
    #                         act[i][j][k] = tmp[k]/sum(tmp)                    
    #     # print("act= ", np.array(act).flatten())
       

    #     return act, np.array(act).flatten()
    
    def traffic_alloc(self, b):
        # self.lamda = arrival traffic rate
        # b = offloading ratios
        
        lamdax=[]
        for n in range(self.gNum):
            # 1 for local process on a vehicle
            lamdax.append(np.random.rand(self.rNum,self.rNum+self.gNum))   
        # print("=====================")         
        # print("len lamda=", len(lamdax[0]))
        # print(self.lamda)
        # print("len b=", len(b[0]))
        # xc
        for i in range(len(lamdax)):
            for j in range(len(lamdax[i])):
                for k in range(len(lamdax[i][j])):
                    lamdax[i][j][k] = math.ceil(self.lamda[i][j]*round(b[i][j][k],2))
        x=[]            
        for m in range(len(lamdax)):
            x.append(lamdax[m].sum(axis=1))       
        
        for i in range(len(x)):
            for j in range(len(x[i])):
                if x[i][j] < self.lamda[i][j]:
                    lamdax[i][j][random.randint(0,len(lamdax[i][j])-1)] += self.lamda[i][j] - x[i][j]
                elif x[i][j] > self.lamda[i][j]:
                    lamdax[i][j][random.randint(0,len(lamdax[i][j])-1)] -= x[i][j] - self.lamda[i][j]
                else:
                    continue
        return lamdax
                
    def get_latency(self, miu, lamda):
        if miu <= lamda:
            lat = huge
        else:
            lat = 1/(miu-lamda)
        return lat

       
    def get_latency_R(self, BVR, BRV, miuR, lamdaR, lamda):
        if miuR <= lamdaR:
            latency = huge
        else:    
            latency = 1/(BVR-(lamda)) + 1/(miuR-lamdaR) + 1/(BRV - (lamda))
        return latency
    
    def get_latency_neighbour_R(self, BVR, BRV, BRR, miuR, lamdaR, lamda):
        if miuR <= lamdaR:
            latency = huge
        else:
            latency = 1/(BVR-(lamda)) + 1/(BRR-(lamdaR)) + 1/(miuR-lamdaR) + 1/(BRR-lamdaR) + 1/(BRV-(lamda))
        return latency
    
    def get_latency_A(self, BVR, BRV, BRA, BAR, miuA, lamdaA, lamda):
        if miuA <= lamdaA:
            latency = huge
        else:
            latency = 1/(BVR-(lamda)) + 1/(BRA-lamdaA) + 1/(miuA-lamdaA) + 1/(BAR-lamdaA) + 1/(BRV-(lamda))
        return latency
    
    
    def get_latency_neighbour_AN(self, BVR, BRV, BRA, BAR, miuA, lamdaA, lamda, DAA ):
        if miuA <= lamdaA:
            latency = huge
        else:
            latency = 1/(BVR-(lamda)) + 1/(BRA-lamdaA) + 1/(miuA-lamdaA) + 1/(BAR-lamdaA) + 1/(BRV-(lamda)) + 2*DAA
        return latency
        

    def get_lamda_R(self, i,k, actions):
        lamdaR = 0
        for j in range(self.rNum):
            lamdaR += actions[i][j][k]
        # for i in range(self.gNum):
        #     for j in range(self.rNum):
        #         for k in range(self.rNum+self.gNum):
        #             lamdaR += actions[i][j][k]       
        return lamdaR

    def get_lamda_A(self, k, actions):
        # print(actions)
        lamdaA = 0
        for i in range(self.gNum):
            for j in range(self.rNum):
                lamdaA += actions[i][j][k]
        return lamdaA
    
    
            
    def is_done(self, step):
        if step == 0:
            self.done = True
        else:
            self.done = False
        return self.done
    
        
    # Calculate reward for given actions
    def action(self, obs, actions):
        latency = obs[0]
        lamda = obs[1]
        BRV = obs[2]
        BVR = obs[3]
        BRR = obs[4]
        BRA = obs[5]
        BAR = obs[6]
        miuR = obs[7]
        miuA = obs[8]
        b = actions

        actions = self.traffic_alloc(b)
        
        for i in range(len(latency)):
            for j in range(len(latency[i])):
                for k in range(len(latency[i][j])):
                    # lamdaV = self.get_lamda_V(i, j, k, actions)
                    lamdaR = self.get_lamda_R(i,k, actions)
                    lamdaA = self.get_lamda_A(k, actions)
                    # print("lamdaR=",lamdaR )
                    # print("lamdaA=",lamdaA )
                   
                    ## Latency that is served at R-MEC
                    if j==k and k<=self.rNum:
                        latency[i][j][k] =self.get_latency_R(BVR, BRV, miuR[i][j], lamdaR, lamda[i][j]) 
                        
                    ## Latency that is served at R-MEC Neighbor
                    elif k<self.rNum and k!=j:
                        latency[i][j][k] = self.get_latency_neighbour_R(BVR, BRV, BRR, miuR[i][j], lamdaR, lamda[i][j])
                        # print(latency[i][j][k])
                        # print(lamdaR)
                        # print(miuR[i][j]-lamdaR)
                    ## Latency that is served at AN-MEC (gNB)
                    elif self.rNum-1 < k < (self.rNum + self.gNum) and k == self.rNum+i:
                        latency[i][j][k] = self.get_latency_A(BVR, BRV, BRA, BAR, miuA[i], lamdaA, lamda[i][j])                       
                        # print(latency[i][j][k])
                    elif self.rNum-1 < k < (self.rNum + self.gNum) and k != self.rNum+i:
                        
                        DAA = self.propagationAN()[i][k-self.rNum]
                        # print ("DCC : ", DCC)
                        latency[i][j][k] = self.get_latency_neighbour_AN(BVR, BRV, BRA, BAR, miuA[i], lamdaA, lamda[i][j], DAA)
                    
        
        total_latency = 0
        for m in range(len(latency)):
            for n in range(len(latency[m])):
                for o in range(len(latency[m][n])):
                    total_latency += (actions[m][n][o]*latency[m][n][o])
                 
        total_traffic = sum(sum(map(sum, actions)))
        
        avg_delay = total_latency/total_traffic
        
        reward = (1/avg_delay)/1000
        if self.done == True:
            self.reset()
            self.R = [[0 for i in range(self.rNum)] for j in range(self.gNum)]
            self.A = [0 for i in range(self.gNum)]
            pass
            # raise Exception("An episode is done")
            
        # self.steps_left -= 1
        return reward
    
    
    def ratios(self, x):        
        for i in range(len(x)):
            # print("sm=========== ",sum(x[i]))
            for j in range(len(x[i])):
                sm = sum(x[i][j])
                for k in range(len(x[i][j])):
                    x[i][j][k] = x[i][j][k]/sm
                
        return x
    def step(self, act):
        curret_obs, obs2, obs = self.get_observation()
        actions =self.ratios(act)
        reward = self.action(curret_obs, actions)
        self.total_reward += reward
        actions = self.traffic_alloc(actions)
        # print ("############################")
        
        A = [] 
        for i in actions:
            A.append([sum(x) for x in zip(*i)])
       
        B = [sum(x) for x in zip(*A)]
        
        for i in range(len(self.A)):
            self.A[i] += B[self.rNum+i]
            for j in range(len(self.R[i])):
                self.R[i][j] += A[i][j]
        
        
        curret_obs, obs2, obs = self.get_observation()
        self.steps_left -= 1
        done = self.is_done(self.steps_left)
        return obs, reward, done
    
    
    def reset(self):
        self.steps_left = 10
        self.rNum = 9
        self.gNum = 3
        self.vNum = [[random.randint(self.vv[0], self.vv[1]) for i in range(self.rNum)] for j in range(self.gNum)]
        self.lam = [[random.randint(20,30) for i in range(self.rNum)] for j in range(self.gNum)]
        # Arrival traffic rates
        # np.random.seed(0)
        self.lamda = [[0 for i in range(self.rNum)]for j in range(self.gNum)]
        # self.lamda = np.zeros(self.rNum)
        for i in range(self.gNum):
            for j in range(self.rNum):
                self.lamda[i][j] = self.vNum[i][j] * self.lam[i][j]
        
        # Array to collect latency (initial will 0) updated during calculation (1 means local exe)
        self.latency = [np.zeros((self.rNum,self.rNum+self.gNum)) for i in range(self.gNum)]

        self.R = [[0 for i in range(self.rNum)] for j in range(self.gNum)]
        self.A = [0 for i in range(self.gNum)]
        # env.is_done = False
        
if __name__ == "__main__":
    
    ############################ Main Step #####################################################
    env = Environment()
    import timeit  
    done = False
    rew = 0
    while not done:
        # print("Action : ", env.get_actions())
        start = timeit.default_timer()
        actions, array=env.get_actions()
        print("actions= ",actions)
        stop = timeit.default_timer()
        execution_time = stop - start
        obs, obs2, obs3= env.get_observation()
        obs, reward, done = env.step(actions)
        rew += reward
        print("reward: ", reward)
        print("R after : ", env.R)
        print("A after : ", env.A)
        total_traffic = (sum(env.R[0])+ sum(env.A))
        print("total_traffic=", total_traffic)
        # print("##################################################")
        if done == True:
            env.reset()
            obs, obs2, obs3 = env.get_observation()
            print("Total reward got : ", rew/10)
    
    ########################################################################################

    # print(env.get_actions())
    # print("AN before : ", env.AN)
    # print("CN before : ", env.CN)
    # print("======================")
    # print("miuV====>", env.InitmiuV)
    # print("miuCn====>", env.InitmiuCn)
    # print("reward=", reward)