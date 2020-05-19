import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pickle
import time

data2018 = pd.read_csv('data2018_ready.csv')
data2018.pop('id')
data2018.pop('OP_CARRIER_FL_NUM')
data2018.pop('CANCELLED')
data2018.pop('DIVERTED')
data2018.pop('ARR_TIME')

data2018 = data2018.values

data2018[:, 0] /= 10.0
data2018[:, 0] += np.random.normal(0, 4, len(data2018))
data2018[:, 1] /= 10.0
data2018[:, 1] += np.random.normal(0, 4, len(data2018))
data2018[:, 2] /= 100.0
data2018[:, 2] += np.random.randn(len(data2018))
data2018[:, 3] += np.random.randn(len(data2018))
data2018[:, 4] += np.random.randn(len(data2018))
data2018[:, 5] += np.random.randn(len(data2018))
data2018[:, 6] /= 100.0
data2018[:, 6] += np.random.randn(len(data2018))
data2018[:, 7] += np.random.randn(len(data2018))
data2018[:, 8] /= 10.0
data2018[:, 8] += np.random.randn(len(data2018))
data2018[:, 9] /= 10.0
data2018[:, 9] += np.random.randn(len(data2018))
data2018[:, 10] /= 10.0
data2018[:, 10] += np.random.randn(len(data2018))

print(data2018.shape)
print(data2018.mean(axis=0))

model = MiniBatchKMeans(n_clusters=3, init='k-means++', n_init=1, batch_size=150000, random_state=69)
#model = KMeans(n_clusters=3, init='k-means++', n_init=1, random_state=69)
time1 = time.time()
res = model.fit_predict(data2018)
print(time.time() - time1)

#f = open('res4.txt', 'wb')
#pickle.dump(res, f)
#f.close()

#print('done')

print(calinski_harabasz_score(data2018, res))
print(davies_bouldin_score(data2018, res))