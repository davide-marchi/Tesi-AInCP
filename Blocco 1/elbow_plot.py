import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib as jl

#per finestre da 300: 3 cluster
#per finestre da 600: non chiaro
#per finestre da 900: 4 cluster


root_dir = 'Blocco 1/60_patients/KMeans'

window = '900'
spec = 'dba'

x = []
y = []
os.chdir(root_dir)
for dirname in os.listdir("."):
    os.chdir(dirname)
    if window in dirname:
        if spec in dirname:
            model = jl.load("trained_model")
            y.append(model.inertia_)
            x.append((model.cluster_centers_).shape[0])
    os.chdir("..")
        
print(x)
print(y)    
        
plt.scatter(x,y)
plt.show()

