#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math
import numpy as np
import random
from array import *
import copy

column_train = ['suhu', 'waktu', 'langit', 'kelembapan', 'terbang']
data_train = pd.read_csv('Data Train.txt', delimiter='\t', header=None, names=column_train, dtype=str).apply(lambda x: x.astype(str).str.lower())

column_test = ['suhu', 'waktu', 'langit', 'kelembapan']
data_test = pd.read_csv('Data Test.txt', delimiter='\t', header=None, names=column_test, dtype=str).apply(lambda x: x.astype(str).str.lower())

def getIndividu(nChrom):
    individu = [0 for x in range(nChrom)] 
    for i in range(nChrom):
        individu[i] = np.random.randint(low=0, high=2)
    return(individu)

def encodeChromosome(data):
    chromosome = []
    if data[0] == "tinggi":
        chromosome.extend([1,0,0])
    elif data[0] == "normal":
        chromosome.extend([0,1,0])
    elif data[0] == "rendah":
        chromosome.extend([0,0,1])
    if data[1] == "pagi":
        chromosome.extend([1,0,0,0])
    elif data[1] == "siang":
        chromosome.extend([0,1,0,0])
    elif data[1] == "sore":
        chromosome.extend([0,0,1,0])
    elif data[1] == 'malam':
        chromosome.extend([0,0,0,1])
    if data[2] == "cerah":
        chromosome.extend([1,0,0,0])
    elif data[2] == "berawan":
        chromosome.extend([0,1,0,0])
    elif data[2] == "rintik":
        chromosome.extend([0,0,1,0])
    elif data[2] == 'hujan':
        chromosome.extend([0,0,0,1])
    if data[3] == "tinggi":
        chromosome.extend([1,0,0])
    elif data[3] == "normal":
        chromosome.extend([0,1,0])
    elif data[3] == "rendah":
        chromosome.extend([0,0,1])
    if data[4] == 'ya':
        chromosome.extend([1])
    elif data[4] == 'tidak':
        chromosome.extend([0])
    return chromosome

def checkSimilarity(data,encode):
    suhu = data[0] == encode[0] ==1 or data[1] == encode[1] ==1 or data[2] == encode[2] ==1
    waktu = data[3] == encode[3] ==1 or data[4] == encode[4] ==1 or data[5] == encode[5] ==1 or data[6] == encode[6] ==1
    langit = data[7] == encode[7] ==1 or data[8] == encode[8] ==1 or data[9] == encode[9] ==1 or data[10] == encode[10] ==1
    kelembapan = data[11] == encode[11] ==1 or data[12] == encode[12] ==1 or data[13] == encode[13] ==1
    if (suhu and waktu and langit and kelembapan):
        return (data[14] == encode[14])
    else:
        return (data[14] != encode[14])

def countFitness(individu,encoded):
    true = 0
    rules = []
    rules = [individu[i:i + 15] for i in range(0, len(individu), 15)]
  
    for i in range(len(encoded)):
        nYes = 0
        nNo = 0
        for j in range(len(rules)):
            check = checkSimilarity(rules[j], encoded[i])
            if (check == True):
                nYes  += 1
            else:
                nNo += 1
        if (nYes > nNo):
            true += 1 
    return (true/80)

def countFitnessAll(populasi, encoded):
    fitness = [0 for x in range(len(populasi))]
    for i in range(len(populasi)):
        fitness[i] = countFitness(populasi[i], encoded)
    return fitness

def rouletteWheelSelection(populasi, encoded):
    total = 0
    for i in range(len(populasi)):
        total += countFitness(populasi[i], encoded)

    r = np.random.uniform(0,1)
    indv = 0
    while (r>0):
        r -= (countFitness(populasi[indv], encoded))/total
        indv +=1
    return (indv-1)

def crossover(pc, parent1, parent2):
    rand = np.random.uniform(0,1)
    if (pc < rand):
        if (len(parent1) > len(parent2)):
            temp = parent1
            parent1 = parent2
            parent2 = temp

        titik1 = np.random.randint(low=0, high=len(parent1))
        titik2 = np.random.randint(low=0, high=len(parent1))

        while (titik1 == titik2):
            titik2 = np.random.randint(low=0, high=len(parent1))

        if(titik1 > titik2):
            temp = titik1
            titik1 = titik2
            titik2 = temp
            
        parent1_titik = [titik1,titik2]
        possible_crossover = []
        p1 = titik2-titik1
        gap = int(p1%15)
        possible_crossover.append([titik1, titik1+p1])
        possible_crossover.append([titik1, titik1+gap])
        possible_crossover.append([titik2-p1, titik2])
        possible_crossover.append([titik2-gap, titik2])
        idx = np.random.randint(low=0, high=4)

        parent2_titik = possible_crossover[idx]
        temp = copy.deepcopy(parent1)
        parent1[parent1_titik[0] : parent1_titik[1]] = parent2[parent2_titik[0] : parent2_titik[1]]
        parent2[parent2_titik[0] : parent2_titik[1]] = temp[parent1_titik[0] : parent1_titik[1]]
    return [parent1,parent2]

def mutation(pm,individu):
    rand = np.random.uniform(0,1)
    if (rand > pm):
        array = np.random.randint(0,2,len(individu))
        for j in range(len(individu)):
            if (array[j] == 1):
                if (individu[j] == 0):
                    individu[j] = 1
                else:
                    individu[j] = 0
    return individu

def elitism(populasi, fitness):
    return fitness.index(max(fitness))

def generationalReplacement(individu,newPop,nIndividu):
    newPop[0] = individu
    newPop[1] = individu
    return newPop
    
maxGenerasi = 200
nIndividu = 50 #banyak individu dalam satu populasi
populasi = [0 for y in range(nIndividu)]
nChrom = 15
pc = 0.7
pm = 0.4

populasi = [0 for x in range(nIndividu)]
for i in range(0,nIndividu):
    random = np.random.randint(low=1, high=9)
    populasi[i] = getIndividu(nChrom*random)

encoded = [[0 for x in range(15)] for y in range(80)]
for i in range(len(data_train)):
    encoded[i] = encodeChromosome(data_train.iloc[i])

for i in range(maxGenerasi):
    populasi1 = copy.deepcopy(populasi)
    print("Generasi ke -", i+1)
    fitness = countFitnessAll(populasi,encoded)

    i = 0
    newPop = [0 for x in range(nIndividu)]
    oldPop = [0 for x in range(nIndividu)]
    while (i < len(populasi)):
        parent1 = rouletteWheelSelection(populasi,encoded)
        parent2 = rouletteWheelSelection(populasi,encoded)
        while (parent1 == parent2):
            parent2 = rouletteWheelSelection(populasi,encoded)
        newPop = copy.deepcopy(oldPop)
        parent = crossover(pc, populasi[parent1], populasi[parent2])
        newPop[i] = mutation(pm,parent[0])
        newPop[i+1] = mutation(pm,parent[1])
        oldPop = newPop
        i += 2
    elitisme = elitism(populasi1, fitness)
    best_individu = populasi1[elitisme]
    print("Fitness terbaik = ",countFitness(best_individu, encoded))
    populasi = generationalReplacement(best_individu,newPop,nIndividu) 
    print()
best_chromosome = populasi[elitisme]
print("Kromosom terbaik : ",best_chromosome)


# In[5]:


def encodeDataTest(data):
    chromosome = []
    if data[0] == "tinggi":
        chromosome.extend([1,0,0])
    elif data[0] == "normal":
        chromosome.extend([0,1,0])
    elif data[0] == "rendah":
        chromosome.extend([0,0,1])
    if data[1] == "pagi":
        chromosome.extend([1,0,0,0])
    elif data[1] == "siang":
        chromosome.extend([0,1,0,0])
    elif data[1] == "sore":
        chromosome.extend([0,0,1,0])
    elif data[1] == 'malam':
        chromosome.extend([0,0,0,1])
    if data[2] == "cerah":
        chromosome.extend([1,0,0,0])
    elif data[2] == "berawan":
        chromosome.extend([0,1,0,0])
    elif data[2] == "rintik":
        chromosome.extend([0,0,1,0])
    elif data[2] == 'hujan':
        chromosome.extend([0,0,0,1])
    if data[3] == "tinggi":
        chromosome.extend([1,0,0])
    elif data[3] == "normal":
        chromosome.extend([0,1,0])
    elif data[3] == "rendah":
        chromosome.extend([0,0,1])
    return chromosome

def decodeDataTest(data):
    if (data ==0):
        return "tidak"
    else:
        return "ya"

encodeTest = [[0 for x in range(14)] for y in range(20)]
for j in range(len(data_test)):
    encodeTest[j] = encodeDataTest(data_test.iloc[j])

rules = []
rules = [best_chromosome[i:i + 15] for i in range(0, len(best_chromosome), 15)]

hasil = [0 for x in range(20)]
for i in range(len(data_test)):
    nYes = 0
    nNo = 0
    for j in range(len(rules)):
        check = checkSimilarity(rules[j], encoded[i])
        if (check == True):
            nYes  += 1
        else:
            nNo += 1
    if (nYes > nNo):
        hasil[i] = 1
    else:
        hasil[i] = 0
            
final_output = []
for i in range(len(hasil)):
    final_output.append(decodeDataTest(hasil[i]))
print(hasil)
print(final_output)

with open('target_latih.txt', 'w') as f:
    for item in final_output:
        f.write("%s\n" % item)



