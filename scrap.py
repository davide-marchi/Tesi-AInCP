import numpy as np

y_AHA = [1,2,3,4,5]
y_pred = [1,1,0,0,0]

hemi_cluster = 1 if np.mean([x for x, y in zip(y_AHA, y_pred) if y == 0]) > np.mean([x for x, y in zip(y_AHA, y_pred) if y == 1]) else 0


print(hemi_cluster)