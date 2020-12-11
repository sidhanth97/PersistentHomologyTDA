import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from persistence_features import PersistantHomology

filename = sys.argv[1]
print('Fetching 1000 data instances.')
df = pd.read_csv(filename,nrows=1000)
ph_feature_extractor = PersistantHomology()
ph_features = []
dist_times = []
bar_times = []
feat_times = []
total_times = []
window_sizes = []
stats = []
print('Analysing time taken to compute multiple window sizes.')
for i in range(20):
  total_time = time.time()
  window_size = (i+1)*20
  data_arr = df.values[:window_size,:]
  ph_feature,dist_time,bar_time,feat_time = ph_feature_extractor.generate_topological_feature(data_arr)
  total_time = time.time() - total_time
  ph_features.append(ph_feature)
  dist_times.append(dist_time)
  bar_times.append(bar_time)
  feat_times.append(feat_time)
  window_sizes.append(window_size)
  total_times.append(total_time)
  stats.append({"window_size":(window_size,df.shape[1]),"processing_time":total_time})

plt.plot(window_sizes,dist_times,label='Distance Matrix')
plt.plot(window_sizes,bar_times,label='Barcode')
plt.plot(window_sizes,feat_times,label='Vectorised Barcode')
#plt.plot(window_sizes,_times,label='Distance Matrix')
plt.legend()
plt.show()

print('Done.')
