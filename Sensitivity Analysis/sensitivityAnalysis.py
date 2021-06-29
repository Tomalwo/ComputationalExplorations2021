# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 19:20:50 2021

@author: liors
"""

from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import os
import re
import matplotlib.pyplot as plt

filePath = "C:\\Users\\liors\\Desktop\\test sens\\"

def parseObjs(filePath):
    '''Parse all objective from all logs in the file path'''
    allObjectives = []
    
    #Get text files 

    for file in os.listdir(filePath):
        if file.endswith("SensetivityAnalysis.txt"):
            textFiles = file
            break
    
    #Read in all objectives
    with open(filePath + textFiles, "r") as file:
        objectivesList = []
        
        for j,line in enumerate(file):
            #End if maximum evaluations reached (for logs that are too long)
            if j>0:
                words = line.split(" ")
                objectiveStrs = words[5].split(",")   
                objectives = list(map(lambda obj: float(obj), objectiveStrs))
                objectivesList.append(objectives)
    
    mins = []
    maxs = []
    for i in range(len(objectivesList[0])):
        mins.append(min([objectivesList[j][i] for j in range(len(objectivesList))]))
        maxs.append(max([objectivesList[j][i] for j in range(len(objectivesList))]))
    
    
    for i in range(len(objectivesList)):
        for j in range(len(mins)):
            if mins[j]!=maxs[j]:
                objectivesList[i][j] = (((objectivesList[i][j] - mins[j]) * (1 - 0)) / (maxs[j] - mins[j])) + 0
    
    objectives = []
    for i in range(len(objectivesList[0])):
        objectives.append(np.array([objectivesList[j][i] for j in range(len(objectivesList))]))
        
    return objectives

def parseProblem(filePath):
    '''Parse problem'''
    allObjectives = []
    
    #Get text files 

    for file in os.listdir(filePath):
        if file.endswith("SensetivityAnalysisAtt.txt"):
            textFiles = file
            break
    
    #Read in all objectives
    txt = ''
    with open(filePath + textFiles, "r") as file:
        for j,line in enumerate(file):
            txt += line
    
    parameters =  re.split(r";", txt)[:-1]
    samples = int(re.split(r";", txt)[-1])
    
    names = []
    bounds = []
    for param in parameters:
        att = re.split(r",",param)
        names.append(att[0])
        #minimum = float(att[1])
        #maximum = float(att[2])
        bounds.append([0.0,1.0])

    problem = {
        'num_vars': len(names),
        'names': names,
        'bounds': bounds
    }
    
    return problem,samples
    
def CreatePlot(index,values,title = "",yLabel = ""):
    
    x = np.arange(len(index))
    width = 0.7
    
    fig, ax = plt.subplots()
    rects = []
    labels = []
    for i in range(len(values)):
        
        rects.append(ax.bar(x + (width/len(values))*i,values[i], width/len(values), label='Obj'+str(i)))
        labels.append('Obj'+str(i))
    
    ax.set_ylabel(yLabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(index,rotation='82.5')
    
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(filePath+title+'.png',dpi=300, bbox_inches = "tight")
    plt.show()
        



problem,samples = parseProblem(filePath)
outputs = parseObjs(filePath)



allFirst = []
allSecond = []
allTotal = []
namesFirst = []
namesSecond = []
namesTotal = []


for i,o in enumerate(outputs):
    Si = sobol.analyze(problem, o,seed=0)
    total_Si, first_Si, second_Si = Si.to_df()
    
    allFirst.append(first_Si["S1"].values)
    namesFirst = first_Si.index
    
    allSecond.append(second_Si["S2"].values)
    namesSecond = [second_Si.index[i][0]+second_Si.index[i][1] for i in range(len(second_Si.index))]
    
    allTotal.append(total_Si["ST"].values)
    namesTotal = total_Si.index


CreatePlot(namesFirst,allFirst,title = "First Order",yLabel = "")
CreatePlot(namesSecond,allSecond,title = "Second Order",yLabel = "")
CreatePlot(namesTotal,allTotal,title = "Total",yLabel = "")




        
