import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib as jl
import re

#per finestre da 300: 3 cluster
#per finestre da 600: non chiaro
#per finestre da 900: 4 cluster


root_dir = 'Blocco 1/concat_version/60_patients/KMeans'

window = '900'
spec = 'euclidean'

x = []
y = []
os.chdir(root_dir)
for dirname in os.listdir("."):
    os.chdir(dirname)
    if window in dirname:
        if spec in dirname:
            match = re.search(r"_K(\d+)", dirname)
            if match:
                model = jl.load("trained_model")
                y.append(model.inertia_)
                x.append(int(match.group(1)))            
            
    os.chdir("..")
        
print(x)
print(y)    

plt.scatter(x,y)
plt.xlabel("numero di clusters")
plt.ylabel("inertia")
plt.show()

