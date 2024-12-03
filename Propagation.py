#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 17:56:18 2021

@author: widhi
"""
import numpy as np
from numpy import genfromtxt
import math
import pandas as pd
import delay

class Propagation:
    def __init__(self, file, site_num):
        self.my_data = genfromtxt(file, delimiter=',')
        self.long = []
        self.lat =[]
        self.i = 1
        self.ANNum = 20
        while self.i < len(self.my_data):
            self.long.append(self.my_data[self.i][0])
            self.lat.append(self.my_data[self.i][1])
            self.i +=1
        self.column = [i for i in range(len(self.long))]
        self.index = self.column
        self.dist_df = np.zeros((self.ANNum,self.ANNum))
        self.prop_df = np.zeros((self.ANNum,self.ANNum))
        self.neighbour = [[] for self.i in range(len(self.long))]

    def distance(self, long1, long2, lati1, lati2):
        lon1 = math.radians(long1)
        lon2 = math.radians(long2)
        lat1 = math.radians(lati1)
        lat2 = math.radians(lati2)
        R = 6373.0
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance
    
    def propagation_delay(self):
    #Measure propagation delay
        for i in self.index:
            for j in self.column:
                self.dist_df[i][j] = self.distance(self.long[i], self.long[j], self.lat[i], self.lat[j])
                self.prop_df[i][j] = self.dist_df[i][j]/(2*10**5)
        return self.prop_df

    # Find neighbour
    def neighbour(self):
        for m in range(len(self,self.neighbour)):
            neightmp = np.where(self.dist_df < 1, 0, self.dist_df)
            indeks = 0
            for n in neightmp[m]:
                print(n)
                if n <= 2:
                    self.neighbour[m].append(indeks)
                indeks += 1
                
        return self.neighbour

propa = Propagation('anmec.csv',20)
print (propa.propagation_delay())
 