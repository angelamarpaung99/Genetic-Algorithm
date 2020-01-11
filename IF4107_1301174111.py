#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import random
from array import *
import copy

def getIndividu(nChrom):
    individu = [0 for x in range(nChrom)] 
    for i in range(6):
        individu[i] = np.random.randint(low=0, high=9)
    return(individu)

def decodeChromosome(chrom):
    x1 = -3 + 6 / (9 * pow(10,-1) + 9 * pow(10,-2) + 9 * pow(10,-3)) * (chrom[0]*pow(10,-1) + chrom[1]*pow(10,-2)+ chrom[2]*pow(10,-3))
    x2 = -2 + 4 / (9 * pow(10,-1) + 9 * pow(10,-2) + 9 * pow(10,-3)) * (chrom[3]*pow(10,-1) + chrom[4]*pow(10,-2)+ chrom[5]*pow(10,-3))
    return [x1,x2]

def countFitness(individu):
    x = decodeChromosome(individu)
    h = (4 - 2.1*pow(x[0],2) + (pow(x[0],4)/3))*pow(x[0],2) + x[0]*x[1] + (-4 + 4*pow(x[1],2))*pow(x[1],2)
    f = -h
    return f    

def tournamentSelection(populasi, size, nIndividu):
    best = []
    for i in range(0,size):
        indv = populasi[random.randint(0,nIndividu-1)]
        if ((best == []) or (countFitness(indv) > countFitness(best))):
            best = indv
    return best

def crossover(pc, parent1, parent2):
    rand = random.random()
    titik = np.random.randint(low=0, high=6)
    if (pc < rand):
        y = titik+1
        for x in range(5-titik):
            temp = parent1[y]
            parent1[y] = parent2[y]
            parent2[y] = temp
            y += 1
    return [parent1,parent2]

def mutation(pm, newPop, nChrom):
    for i in range(nIndividu-2):
        rand = random.random()
        if (pm < rand):
            mutasi = np.random.randint(low=0, high=nChrom-1)
            newPop[i][mutasi] = np.random.randint(low=0, high=9)
    return newPop

def elitism(populasi):
    max = []
    for i in range(nIndividu):
        indv = populasi[i]
        if ((max == []) or (countFitness(indv) > countFitness(max))):
            max = indv    
    return max

def generationalReplacement(elitismResult,newPop,nIndividu):
    newPop[nIndividu-2] = elitismResult
    newPop[nIndividu-1] = elitismResult
    return newPop

maxGenerasi = 300
nIndividu = 15 #banyak individu dalam satu populasi
populasi = [0 for y in range(nIndividu)]
nChrom = 6 #banyak chromosome dalam satu individu

for i in range(0,nIndividu):
    populasi[i] = getIndividu(nChrom)
    
for i in range(maxGenerasi):  
    print("\nGenerasi ke-", i+1)
    nGen = 3 #banyak gen dalam kromosom
    pc = 0.7 #probability crossover
    size = int(nIndividu/2) #tournament size
    pm = 1/nGen #probabilty mutation
    
    elitismResult = [0 for x in range(nChrom)]
    elitismResult = elitism(populasi)
    
    print("Best Fitness : ", countFitness(elitismResult))
    
    num_parents_mating = nIndividu - 2
    newPop = [0 for x in range(nIndividu)]
    oldPop = [0 for x in range(nIndividu)]
    populasi1  = copy.deepcopy(populasi) #melakukan copy terhadap array populasi 

    i = 0
    maks = []
    while (i < num_parents_mating):
        parent1 = tournamentSelection(populasi,size,nIndividu)
        parent2 = tournamentSelection(populasi,size,nIndividu)
        while parent1 == parent2:
            parent2 = tournamentSelection(populasi,size,nIndividu)
        newPop = copy.deepcopy(oldPop)
        parent = crossover(pc, parent1, parent2)
        newPop[i] = parent[0]
        newPop[i+1] = parent[1]
        oldPop = newPop
        i += 2
    
    newPop = mutation(pm, newPop, nChrom)
    populasi = generationalReplacement(elitismResult,newPop, nIndividu)
print("Kromosom terbaik : ", elitism(populasi1))
print("Hasil dekode kromosom terbaik : ", decodeChromosome(elitism(populasi1)))

