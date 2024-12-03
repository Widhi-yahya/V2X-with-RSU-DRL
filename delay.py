# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 19:39:50 2021

@author: widhi
"""

import math

# =============================================================================
# Computation offloading
# =============================================================================

def comp_local(uE, uRU, uRD, rh, Po):
    delay =  (1/(uRU- rh)) + (1/(uE-(1-Po)*rh)) + (1/(uRD-rh))
    # print (1/(uRU- rh))
    # print (1/(uE-(1-Po)*rh))
    # print (1/(uRD-rh))
    return delay

def comp_near(uE, uRU, uRD, rh, Po, CN, ANNum):
    Pn = Po*(1/(ANNum-1))
    delay =  (1/(uRU- rh)) + 2*CN + (1/(uE-rn-Pn*rh)) + (1/(uRD-rh))
    # print (1/(uRU- rh))
    # print (2*CN)
    # print (1/(uE-rn-Pn*rh))
    # print (1/(uRD-rh))
    return delay

def comp_nearE(uE, uEh, uRU, uRD, rh, rn, Po, CN, ANNum):
    Pn = Po*(1/(ANNum-1))
    alpha1 = (Pn*rh)/(rn+Pn*rh)
    alpha2 = rn/(rn+Pn*rh)
    EX = alpha1/uEh + alpha2/uE
    EX2 = 2 * (alpha1/uEh**2 + alpha2/uE**2)
    miu = 1/EX
    r = rn+math.ceil(Po*rh)
    # print ("r :", r)
    rho = r/miu
    ETe = 1/miu + (rho*EX2/(2*(1-rho)*EX))     
    # print ("ete '", EX)
    delay =  (1/(uRU- rh)) + 2*CN + ETe + (1/(uRD-rh))
    return delay

# =============================================================================
# Communication offloading
# =============================================================================
def comm_local(uE, uRU, uRD, rh, Po):
    delay = 1/(uRU - (1-Po)*rh) + 1/(uE-rh) + 1/(uRD-(1-Po)*rh)
    return delay

def comm_nearE(uE, uRU, uRUh, uRD, uRDh, rh, rn, Po, CN, ANNum):
    Pn = Po*(1/(ANNum-1))
    alpha1 = (Pn*rh)/(rn+Pn*rh)
    alpha2 = rn/(rn+Pn*rh)
    r = rn+math.ceil(Po*rh)
    # print ("r :", r)
    EXRU = alpha1/uRUh + alpha2/uRU
    EXRU2 = 2 * (alpha1/uRUh**2 + alpha2/uRU**2)
    miuRU = 1/EXRU
    rhoRU = r/miuRU
    ETRU =  1/miuRU + (rhoRU*EXRU2/(2*(1-rhoRU)*EXRU))
    # print ("uru :", uRU)
    EXRD = alpha1/uRDh + alpha2/uRD
    EXRD2 = 2 * (alpha1/uRDh**2 + alpha2/uRD**2)
    miuRD = 1/EXRD
    rhoRD = r/miuRD
    ETRD = 1/miuRD + (rhoRD*EXRD2/(2*(1-rhoRD)*EXRD))
    
    delay =  ETRU + 2*CN + (1/(uE-rh)) + ETRD

    return delay


# =============================================================================
# Communication and computation offloading
# =============================================================================

def comm_comp_local(uE, uRU, uRD, rh, Po):
    delay = 1/(uRU - (1-Po)*rh) + 1/(uE-(1-Po)*rh) + 1/(uRD-(1-Po)*rh)
    # print("delay")
    return delay

def comm_comp_nearBS(uE, uRU, uRUh, uRD, uRDh, rh, rn, Po, Pe, CN, ANNum):
    X = 6
    Pn = Po*1/X
    delay = 1/(uRU-rn-Pn*rh) + 1/(uE-(1-Pe)*Pn*rh-rn) + 1/(uRD-rn-Pn*rh)
    return delay

def comm_comp_nearBSEmp(uE, uRU, uRUh, uRD, uRDh, rh, rn, Po, Pe, CN, ANNum):
    X = 6
    Pn = Po*1/X
    alpha1 = (Pn*rh)/(rn+Pn*rh)
    alpha2 = rn/(rn+Pn*rh)
    EX = alpha1/uEh + alpha2/uE
    EX2 = 2 * (alpha1/uEh**2 + alpha2/uE**2)
    miu = 1/EX
    r = rn+math.ceil(Po*rh)
    # print ("r :", r)
    rho = r/miu
    ETe = 1/miu + (rho*EX2/(2*(1-rho)*EX))
    
    delay =  1/(uRU-rn-Pn*rh) + ETe + 1/(uRD-rn-Pn*rh)
    return delay

def comm_comp_nearBSEmm(uE, uRU, uRUh, uRD, uRDh, rh, rn, Po, Pe, CN, ANNum):
    X = 6
    Pn = Po*1/X
    alpha1 = (Pn*rh)/(rn+Pn*rh)
    alpha2 = rn/(rn+Pn*rh)
    r = rn+math.ceil(Po*rh)
    # print ("r :", r)
    EXRU = alpha1/uRUh + alpha2/uRU
    EXRU2 = 2 * (alpha1/uRUh**2 + alpha2/uRU**2)
    miuRU = 1/EXRU
    rhoRU = r/miuRU
    ETRU =  1/miuRU + (rhoRU*EXRU2/(2*(1-rhoRU)*EXRU))
    # print ("uru :", uRU)
    EXRD = alpha1/uRDh + alpha2/uRD
    EXRD2 = 2 * (alpha1/uRDh**2 + alpha2/uRD**2)
    miuRD = 1/EXRD
    rhoRD = r/miuRD
    ETRD = 1/miuRD + (rhoRD*EXRD2/(2*(1-rhoRD)*EXRD))
    
    delay = ETRU + 1/(uE-(1-Pe)*Pn*rh-rn +ETRD)
    return delay

def comm_comp_nearBSEall(uE, uRU, uRUh, uRD, uRDh, rh, rn, Po, Pe, CN, ANNum):
    X = 6
    Pn = Po*1/X
    alpha1 = (Pn*rh)/(rn+Pn*rh)
    alpha2 = rn/(rn+Pn*rh)
    r = rn+math.ceil(Po*rh)
    
    # print ("r :", r)
    EXRU = alpha1/uRUh + alpha2/uRU
    EXRU2 = 2 * (alpha1/uRUh**2 + alpha2/uRU**2)
    miuRU = 1/EXRU
    rhoRU = r/miuRU
    ETRU =  1/miuRU + (rhoRU*EXRU2/(2*(1-rhoRU)*EXRU))
    # print ("uru :", uRU)
    EXRD = alpha1/uRDh + alpha2/uRD
    EXRD2 = 2 * (alpha1/uRDh**2 + alpha2/uRD**2)
    miuRD = 1/EXRD
    rhoRD = r/miuRD
    ETRD = 1/miuRD + (rhoRD*EXRD2/(2*(1-rhoRD)*EXRD))
    
    EX = alpha1/uEh + alpha2/uE
    EX2 = 2 * (alpha1/uEh**2 + alpha2/uE**2)
    miu = 1/EX
    r = rn+math.ceil(Po*rh)
    # print ("r :", r)
    rho = r/miu
    ETe = 1/miu + (rho*EX2/(2*(1-rho)*EX))
    
    delay = ETRU + ETe + ETRD
    
    return delay

# =============================================================================
# Communication (neighbour) & computation (non-neighbour)
# =============================================================================

def comp_non_near(uE, uRU, uRD, rh, rn, Po, Pe, CN):
    X = 6
    Pn = Po*1/X
    delay = 1/(uRU-rn-Pn*rh)+ 2*CN +1/(uE-Pe*Pn*rh-rn) +1/(uRD-rn-Pn*rh)
    return delay

def comp_non_near_Emp(uE,uEh, uRU, uRD, rh, rn, Po, Pe, CN):
    X = 6
    Pn = Po*1/X
    alpha1 = (Pn*rh)/(rn+Pn*rh)
    alpha2 = rn/(rn+Pn*rh)
    EX = alpha1/uEh + alpha2/uE
    EX2 = 2 * (alpha1/uEh**2 + alpha2/uE**2)
    miu = 1/EX
    r = rn+math.ceil(Po*rh)
    # print ("r :", r)
    rho = r/miu
    ETe = 1/miu + (rho*EX2/(2*(1-rho)*EX))
    delay = 1/(uRU-rn-Pn*rh)+ 2*CN + ETe +1/(uRD-rn-Pn*rh)
    
    return delay

def comp_non_near_Emm(uE, uRU, uRUh, uRD, uRDh, rh, rn, Po, Pe, CN, ANNum):
    X = 6
    Pn = Po*1/X
    alpha1 = (Pn*rh)/(rn+Pn*rh)
    alpha2 = rn/(rn+Pn*rh)
    r = rn+math.ceil(Po*rh)
    # print ("r :", r)
    EXRU = alpha1/uRUh + alpha2/uRU
    EXRU2 = 2 * (alpha1/uRUh**2 + alpha2/uRU**2)
    miuRU = 1/EXRU
    rhoRU = r/miuRU
    ETRU =  1/miuRU + (rhoRU*EXRU2/(2*(1-rhoRU)*EXRU))
    # print ("uru :", uRU)
    EXRD = alpha1/uRDh + alpha2/uRD
    EXRD2 = 2 * (alpha1/uRDh**2 + alpha2/uRD**2)
    miuRD = 1/EXRD
    rhoRD = r/miuRD
    ETRD = 1/miuRD + (rhoRD*EXRD2/(2*(1-rhoRD)*EXRD))
    
    delay = ETRU + 2*CN +1/(uE-Pe*Pn*rh-rn) +ETRD
    return delay

def comp_non_near_Eall(uE, uRU, uRUh, uRD, uRDh, rh, rn, Po, Pe, CN, ANNum):
    X = 6
    Pn = Po*1/X
    alpha1 = (Pn*rh)/(rn+Pn*rh)
    alpha2 = rn/(rn+Pn*rh)
    r = rn+math.ceil(Po*rh)
    
    # print ("r :", r)
    EXRU = alpha1/uRUh + alpha2/uRU
    EXRU2 = 2 * (alpha1/uRUh**2 + alpha2/uRU**2)
    miuRU = 1/EXRU
    rhoRU = r/miuRU
    ETRU =  1/miuRU + (rhoRU*EXRU2/(2*(1-rhoRU)*EXRU))
    # print ("uru :", uRU)
    EXRD = alpha1/uRDh + alpha2/uRD
    EXRD2 = 2 * (alpha1/uRDh**2 + alpha2/uRD**2)
    miuRD = 1/EXRD
    rhoRD = r/miuRD
    ETRD = 1/miuRD + (rhoRD*EXRD2/(2*(1-rhoRD)*EXRD))
    
    EX = alpha1/uEh + alpha2/uE
    EX2 = 2 * (alpha1/uEh**2 + alpha2/uE**2)
    miu = 1/EX
    r = rn+math.ceil(Po*rh)
    # print ("r :", r)
    rho = r/miu
    ETe = 1/miu + (rho*EX2/(2*(1-rho)*EX))
    
    delay = ETRU + 2*CN + ETe + ETRD
    
    return delay

# =============================================================================
# ANNum = 10              # Unit
# rn = 100       # Traffic rate
# rh = 500       # Traffic rate HYPER
# rnMn = 128 * 10**3    # Byte (Wan-Chi Chang) communication demand (request size)
# rnM = 256 * 10**3   # communication demand (request size) HYPER
# rnC = 12 * 10**6    # Instructions/sec comptation demand HYPER (https://www.d.umn.edu/~gshute/arch/performance-equation.xhtml#example-solution)
# rnCn = 6 * 10**6    # computation demand normal
# RC = 300 * 10**9        # Computation capacity Instructions/sec
# RMRU = 1.25 * 10**9     # Communication capacity Byte/sec
# RMRD =  2.5 * 10**9     # Communication capacity 
# RCN = 7.5 * 10**9       # Communication capacity between AN
# Po = 0.3
# CN = rnM/RCN           # service rate CN
# Pe = 0.3
# uRU = RMRU/rnMn        # RAN Upload service rate 
# uRD = RMRU/(rnMn*10)   # RAN download service rate 
# uRUh = RMRU/rnM        # RAN upload HYPER service rate 
# uRDh = RMRU/(rnM*10)   # RAN download HYPER service rate 
# uE = RC/rnCn           # Edge service rate
# uEh = RC/rnC           # Edge HYPER service rate
# 
# 
# print ("==============COMPUTATION OFFLOADING===================")
# 
# comp_local = comp_local(uE, uRU, uRD, rh, Po)
# print ("comp_local: ",comp_local)
# 
# comp_near = comp_near(uE, uRU, uRD, rh, Po, CN, ANNum)
# print ("comp_near: ",comp_near)
# 
# comp_nearE = comp_nearE(uE, uEh, uRU, uRD, rh, rn, Po, CN, ANNum)
# print ("comp_nearE: ",comp_nearE)
# print()
# print()
# print ("==============COMMUNICATION OFFLOADING===================")
# 
# comm_local = comm_local(uE, uRU, uRD, rh, Po)
# print ("comm_local: ",comm_local)
# 
# comm_nearE = comm_nearE(uE, uRU, uRUh, uRD, uRDh, rh, rn, Po, CN, ANNum)
# print ("comm_nearE: ",comm_nearE)
# print()
# print()
# print ("==============COMMUNICATION & COMPUTATION OFFLOADING===================")
# 
# comm_comp_local = comm_comp_local(uE, uRU, uRD, rh, Po)
# print ("comm_comp_local : ",comm_comp_local)
# 
# comm_comp_nearBS = comm_comp_nearBS(uE, uRU, uRUh, uRD, uRDh, rh, rn, Po, Pe, CN, ANNum)
# print ("comm_comp_nearBS: ",comm_comp_nearBS)
# 
# comm_comp_nearBSEmp = comm_comp_nearBSEmp(uE, uRU, uRUh, uRD, uRDh, rh, rn, Po, Pe, CN, ANNum)
# print ("comm_comp_nearBSEmp: ",comm_comp_nearBSEmp)
# 
# comm_comp_nearBSEmm = comm_comp_nearBSEmm(uE, uRU, uRUh, uRD, uRDh, rh, rn, Po, Pe, CN, ANNum)
# print ("comm_comp_nearBSEmm: ",comm_comp_nearBSEmm)
# 
# comm_comp_nearBSEall = comm_comp_nearBSEall(uE, uRU, uRUh, uRD, uRDh, rh, rn, Po, Pe, CN, ANNum)
# print ("comm_comp_nearBSEall: ",comm_comp_nearBSEall)
# 
# 
# print()
# print()
# print ("==============COMMUNICATION & COMPUTATION NON-NEIGHBOUR===================")
# comp_non_near = comp_non_near(uE, uRU, uRD, rh, rn, Po, Pe, CN)
# print ("comp_non_near: ",comp_non_near)
# comp_non_near_Emp = comp_non_near_Emp(uE,uEh, uRU, uRD, rh, rn, Po, Pe, CN)
# print ("comp_non_near_Emp: ",comp_non_near_Emp)
# comp_non_near_Emm= comp_non_near_Emm(uE, uRU, uRUh, uRD, uRDh, rh, rn, Po, Pe, CN, ANNum)
# print ("comp_non_near_Emm: ",comp_non_near_Emm)
# comp_non_near_Eall = comp_non_near_Eall(uE, uRU, uRUh, uRD, uRDh, rh, rn, Po, Pe, CN, ANNum)
# print ("comp_non_near_Eall: ",comp_non_near_Eall)
# =============================================================================
