from pyspark import SparkContext, SparkConf
from sklearn.cluster import KMeans
import numpy as np
import os, sys
import time
from operator import add

timeStart = time.time()

INPUT = sys.argv[1]
N_CLUSTER = int(sys.argv[2])
OUTPUT = sys.argv[3]

# Data Process: Load Data
sc = SparkContext('local[*]', 'BFR')
rawData = sc.textFile(INPUT, None, False)
rawData = rawData.map(lambda x: x.decode().split(','))
allData = rawData.map(lambda x: list(map(eval,x[2:]))).collect()
intermediate_res = list()

def getRS(arr):
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        if v == 1:
            result[k] = v
    return result


# -------- Initiate --------
# Step 1. Load 20% of the data randomly.
n_data = len(allData)
percentage = 0.02
big_cluster = N_CLUSTER * 10
print(int(n_data * percentage))
init_data = np.array(allData[:int(n_data * percentage)])
init_data_inx = np.arange(1, int(n_data * percentage)+1)

# Step 2. Run K-Means (e.g., from sklearn) with a large K (e.g., 10 times of the given cluster numbers) on the data in memory using the Euclidean distance as the similarity measurement.
s2Start = time.time()
filKmeans = KMeans(n_clusters=big_cluster * 8, random_state=0).fit(init_data)
print("outliner number: " + str(len(getRS(filKmeans.labels_))))
s2End = time.time()
print("percentage: "+str(percentage)+", cluster: "+str(big_cluster*8))
print("step2: %f sec" % (s2End - s2Start))

# Step 3. In the K-Means result from Step 2, move all the clusters with only one point to RS (outliers).
s3Start = time.time()
init_key = np.unique(filKmeans.labels_)
retained_set = list()
retained_set_inx = list()
for k in init_key:
    mask = (filKmeans.labels_ == k)
    arr_temp = filKmeans.labels_[mask]
    v = arr_temp.size
    if v == 1:
        inx = np.argwhere(filKmeans.labels_ == k)[0][0]
        retained_set.append(init_data[inx])
        retained_set_inx.append(inx)
init_data = np.delete(init_data, retained_set_inx, axis=0)
init_data_inx = np.delete(init_data_inx, retained_set_inx)
retained_set = np.array(retained_set)
retained_set_inx = np.array([i+1 for i in retained_set_inx])
s3End = time.time()
print("step3: %f sec" % (s3End - s3Start))

# Step 4. Run K-Means again to cluster the rest of the data point with K = the number of input clusters.
s4Start = time.time()
initKmeans = KMeans(n_clusters = N_CLUSTER, random_state=0).fit(init_data)
s4End = time.time()
print("step4: %f sec" % (s4End - s4Start))

# Step 5. Use the K-Means result from Step 4 to generate the DS clusters (i.e., discard their points and generate statistics).
s5Start = time.time()
discard_set = -np.ones(n_data)
ds_N = list()
ds_SUMSQ = list()
ds_count = 0
init_key = np.unique(initKmeans.labels_)
for k in init_key:
    inx = np.argwhere(initKmeans.labels_ == k).flatten()
    ds_count += inx.size
    ds_N.append([inx.size])
    ds_SUMSQ.append(np.sum(np.power(init_data[inx], 2), axis=0))
    ground_inx = init_data_inx[inx] - 1
    discard_set[ground_inx] = k
ds_CEN = initKmeans.cluster_centers_
ds_SUM = ds_CEN * ds_N
ds_SUMSQ = np.array(ds_SUMSQ)
ds_SV = np.power((ds_SUMSQ / ds_N) - np.power(ds_CEN,2), 0.5) * 2
s5End = time.time()
print("step5: %f sec" % (s5End - s5Start))

# Step 6. Run K-Means on the points in the RS with a large K to generate CS (clusters with more than one points) and RS (clusters with only one point).
s6Start = time.time()
temp_cluster = int(retained_set_inx.size * 0.7)
initKmeans = KMeans(n_clusters=temp_cluster, random_state=0).fit(retained_set)
print("outliner number: " + str(len(getRS(initKmeans.labels_))))

init_key = np.unique(initKmeans.labels_)
compression_set_inx = list()
inxList = list()
cs_N = list()
cs_SUMSQ = list()
cs_CEN = list()
cs_count = 0
for k in init_key:
    inx = np.argwhere(initKmeans.labels_ == k).flatten().tolist()
    if len(inx) > 1:
        cs_count += len(inx)
        compression_set_inx.append(retained_set_inx[inx].tolist())
        cs_N.append([len(inx)])
        cs_SUMSQ.append(np.sum(np.power(retained_set[inx], 2), axis=0))
        cs_CEN.append(initKmeans.cluster_centers_[k])
        inxList += inx
cs_CEN = np.array(cs_CEN)
cs_SUM = cs_CEN * cs_N
cs_SUMSQ = np.array(cs_SUMSQ)
cs_SV = np.power((cs_SUMSQ / cs_N) - np.power(cs_CEN, 2), 0.5) * 2
retained_set = np.delete(retained_set, inxList, axis=0)
retained_set_inx = np.delete(retained_set_inx, inxList)
s6End = time.time()
print("step6: %f sec" % (s6End - s6Start))

intermediate_res.append((ds_count, len(compression_set_inx), cs_count, retained_set_inx.size))
print (intermediate_res)

# -------- Computation Loop --------
# Step 7. Load another 20% of the data randomly.
start = int(n_data * percentage)
end = start + int(n_data * percentage)
while start < n_data:
    # Step 8. For the new points, compare them to each of the DS using the Mahalanobis Distance and assign them to the nearest DS clusters if the distance is < 2√𝑑.
    # Step 9. For the new points that are not assigned to DS clusters, using the Mahalanobis Distance and assign the points to the nearest CS clusters if the distance is < 2√𝑑
    # Step 10. For the new points that are not assigned to a DS cluster or a CS cluster, assign them to RS.
    # Step 11. Run K-Means on the RS with a large K to generate CS (clusters with more than one points) and RS (clusters with only one point).
    # Step 12. Merge CS clusters that have a Mahalanobis Distance < 2√𝑑.
    start = end
    end = start + int(n_data * percentage)
    # if end >= n_data:
    # assign CS to DS

# Test
# testData = rawData.map(lambda x: (int(x[0]), int(x[1])))

# Print
# fileOfOutput = open(OUTPUT, 'w')
# outputStr = ""
# fileOfOutput.close()

timeEnd = time.time()
print("Duration: %f sec" % (timeEnd - timeStart))

# bin/spark-submit \
# --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:conf/log4j.xml" \
# --conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=file:conf/log4j.xml" \
# ../inf553-hw5/jingyue_fu_bfr.py ../inf553-hw5/input/hw5_clustering.txt 10 ../inf553-hw5/output/jingyue_fu_bfr.txt
