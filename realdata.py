import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import pickle

data2018 = pd.read_csv('../airline-delay-and-cancellation-data-2009-2018/2018.csv').drop(['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'Unnamed: 27', 'OP_CARRIER', 'DEP_TIME', 'WHEELS_OFF', 'WHEELS_ON', 'AIR_TIME', 'CANCELLATION_CODE'], axis=1)
data2018 = data2018.dropna()
date = data2018.pop('FL_DATE')
print(data2018.shape)
origin = data2018.pop("ORIGIN").values
dest = data2018.pop("DEST").values
cols = data2018.columns
data2018 = data2018.values

airports = np.unique(np.append(origin, dest))
o = np.zeros(len(origin))
d = np.zeros(len(dest))
for i in enumerate(airports):
    o += (origin == i[1]) * (i[0] + 1)
    d += (dest == i[1]) * (i[0] + 1)
o = o.reshape(-1, 1)
d = d.reshape(-1, 1)
data2018 = np.hstack([o, d, data2018])
cols_tmp = np.array(['ORIGIN', 'DEST'])
cols = np.insert(cols, 0, cols_tmp)

data2018_ready = pd.DataFrame(data2018, columns=cols)
data2018_ready.to_csv('data2018_ready.csv')

f = open('airports.txt', 'wb')
pickle.dump(airports, f)
f.close()

print('airports done')
print(data2018.shape)

#day, month = np.array([]), np.array([])
#for i in date:
#    d, m = int(i[8:]), int(i[5:7])
#    day = np.append(day, d)
#    month = np.append(month, m)

#day = day.reshape(-1, 1)
#month = month.reshape(-1, 1)
#data2018 = np.hstack([day, month, data2018])
#cols_tmp = np.array(['day', 'month'])
#cols = np.insert(cols, 0, cols_tmp)

model = MiniBatchKMeans(n_clusters=3, init='k-means++', n_init=4, batch_size=1000000, random_state=69)
res = model.fit_predict(data2018)

f = open('res1.txt', 'wb')
pickle.dump(res, f)
f.close()

print(silhouette_score(data2018, res))