# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 11:29:48 2021

@author: liors
"""

import sys
from subprocess import PIPE,Popen
import re
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np


parameters =  re.split(r";", sys.argv[1])[:-1]
samples = int(re.split(r";", sys.argv[1])[-1])
names = []
bounds = []
for param in parameters:
    att = re.split(r",",param)
    names.append(att[0])
    minimum = float(att[1])
    maximum = float(att[2])
    bounds.append([0.0,1.0])

problem = {
    'num_vars': len(names),
    'names': names,
    'bounds': bounds
}

param_values = saltelli.sample(problem, samples).tolist()
print (param_values)
