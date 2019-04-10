from pyspark import SparkContext, SparkConf
from sklearn.cluster import KMeans
import numpy as np
import os, sys
import time
from operator import add
from collections import Counter
from sklearn.metrics.cluster import normalized_mutual_info_score

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
percentage = 0.2
big_cluster = N_CLUSTER * 10
print(int(n_data * percentage))
init_data = np.array(allData[:int(n_data * percentage)])
init_data_inx = np.arange(int(n_data * percentage))

# Step 2. Run K-Means (e.g., from sklearn) with a large K (e.g., 10 times of the given cluster numbers) on the data in memory using the Euclidean distance as the similarity measurement.
s2Start = time.time()
filKmeans = KMeans(n_clusters=big_cluster, random_state=0).fit(init_data)
print("outliner number: " + str(len(getRS(filKmeans.labels_))))
s2End = time.time()
print("percentage: "+str(percentage)+", cluster: "+str(big_cluster))
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
retained_set_inx = np.array([i for i in retained_set_inx])
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
ds_N = np.array(ds_N)
ds_CEN = initKmeans.cluster_centers_
ds_SUM = ds_CEN * ds_N
ds_SUMSQ = np.array(ds_SUMSQ)
ds_SV = np.sqrt((ds_SUMSQ / ds_N) - np.power(ds_CEN,2))
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
cs_N = np.array(cs_N)
cs_CEN = np.array(cs_CEN)
cs_SUM = cs_CEN * cs_N
cs_SUMSQ = np.array(cs_SUMSQ)
cs_SV = np.sqrt((cs_SUMSQ / cs_N) - np.power(cs_CEN, 2))
retained_set = np.delete(retained_set, inxList, axis=0)
retained_set_inx = np.delete(retained_set_inx, inxList)
s6End = time.time()
print("step6: %f sec" % (s6End - s6Start))

intermediate_res.append((ds_count, len(compression_set_inx), cs_count, retained_set_inx.size))


# -------- Computation Loop --------
# Step 7. Load another 20% of the data randomly.
percentage = 0.05
start = int(n_data * percentage)
end = start + int(n_data * percentage)
while start < n_data:
    if end > n_data:
        end = n_data
    data = np.array(allData[start:end])
    data_len = end - start
    # Step 8. For the new points, compare them to each of the DS using the Mahalanobis Distance and assign them to the nearest DS clusters if the distance is < 2√𝑑.
    ds_temp = [[] for i in range(10)]
    cs_temp = [[] for i in range(len(compression_set_inx))]
    for i in range(data_len):
        ds_times = np.abs(ds_CEN - data[i]) / ds_SV
        ds_min_line = np.argmax(np.bincount(ds_times.argmin(axis=0)))
        ds_times = ds_times[ds_min_line]
        if np.argwhere(ds_times > 2.5).size == 0:
            discard_set[start + i] = ds_min_line
            ds_temp[ds_min_line].append(data[i])
            ds_count += 1
        else:
            # Step 9. For the new points that are not assigned to DS clusters, using the Mahalanobis Distance and assign the points to the nearest CS clusters if the distance is < 2√𝑑
            if np.argwhere(cs_SV == 0).size >0:
                print (cs_SV)
            cs_times = np.abs(cs_CEN - data[i]) / cs_SV
            cs_min_line = np.argmax(np.bincount(cs_times.argmin(axis=0)))
            cs_times = cs_times[cs_min_line]
            # print (len(cs_temp), cs_min_line)
            if np.argwhere(cs_times > 2).size == 0:
                cs_temp[cs_min_line].append(data[i])
                compression_set_inx[cs_min_line].append(start + i)
                cs_count += 1
            else:
                # Step 10. For the new points that are not assigned to a DS cluster or a CS cluster, assign them to RS.
                retained_set = np.r_[retained_set,np.array([data[i]])]
                retained_set_inx = np.append(retained_set_inx, start + i)
    # update ds & cs N, SUM, SUMSQ
    for i in range(10):
        temp = np.array(ds_temp[i])
        ds_SUM[i] += np.sum(temp, axis=0)
        ds_SUMSQ[i] += np.sum(np.power(temp, 2), axis=0)
    ds_N_temp = np.array([[len(i)] for i in ds_temp])
    ds_N += ds_N_temp
    ds_CEN = ds_SUM / ds_N
    ds_SV = np.sqrt((ds_SUMSQ / ds_N) - np.power(ds_CEN, 2))
    for i in range(len(compression_set_inx)):
        temp = np.array(cs_temp[i])
        cs_SUM[i] += np.sum(temp, axis=0)
        cs_SUMSQ[i] += np.sum(np.power(temp, 2), axis=0)
    cs_N_temp = np.array([[len(i)] for i in cs_temp])
    cs_N += cs_N_temp
    cs_CEN = cs_SUM / cs_N
    cs_SV = np.sqrt((cs_SUMSQ / cs_N) - np.power(cs_CEN, 2))

    # Step 11. Run K-Means on the RS with a large K to generate CS (clusters with more than one points) and RS (clusters with only one point).
    if retained_set_inx.size != 0:
        temp_cluster = min(int(retained_set_inx.size * 0.7), N_CLUSTER * 10)
        retainedKmeans = KMeans(n_clusters=temp_cluster, random_state=0).fit(retained_set)
        init_key = np.unique(retainedKmeans.labels_)
        inxList = list()
        for k in init_key:
            inx = np.argwhere(retainedKmeans.labels_ == k).flatten().tolist()
            if len(inx) > 1:
                cs_count += len(inx)
                compression_set_inx.append(retained_set_inx[inx].tolist())
                cs_N = np.r_[cs_N, np.array([[len(inx)]])]
                sq_temp = np.array([np.sum(np.power(retained_set[inx], 2), axis=0)])
                # cen_temp = np.array([retainedKmeans.cluster_centers_[k]])
                sum_temp = np.array([np.sum(retained_set[inx], axis=0)])
                cen_temp = np.array(sum_temp/len(inx))
                sv_temp = (sq_temp / len(inx)) - np.power(cen_temp, 2)
                if np.argwhere(sv_temp < 0).size > 0:
                    print (np.sum(retained_set[inx], axis=0)/len(inx))
                    print (cen_temp)
                cs_SUMSQ = np.r_[cs_SUMSQ, sq_temp]
                cs_CEN = np.r_[cs_CEN, cen_temp]
                cs_SUM = np.r_[cs_SUM, sum_temp]
                cs_SV = np.r_[cs_SV, np.sqrt(sv_temp)]

                inxList += inx
        retained_set = np.delete(retained_set, inxList, axis=0)
        retained_set_inx = np.delete(retained_set_inx, inxList)

    # Step 12. Merge CS clusters that have a Mahalanobis Distance < 2√𝑑.
    merge_list = list()
    # for i in range(len(compression_set_inx)):
    #     for j in range(i+1, len(compression_set_inx)):
    #         dis_times_1 = np.abs(cs_CEN[i] - cs_CEN[j]) / cs_SV[i]
    #         dis_times_2 = np.abs(cs_CEN[i] - cs_CEN[j]) / cs_SV[j]
    #         if np.argwhere(dis_times_1 > 2).size == 0 and np.argwhere(dis_times_2 > 2).size == 0:
    #             merge_list.append([i,j])
    for i in range(len(compression_set_inx)-1):
        for j in range(i+1, len(compression_set_inx)):
            N_temp = cs_N[i] + cs_N[j]
            SUM_temp = cs_SUM[i] + cs_SUM[j]
            SUMSQ_temp = cs_SUMSQ[i] + cs_SUMSQ[j]
            SV_temp = np.sqrt((SUMSQ_temp / N_temp) - np.power((SUM_temp / N_temp), 2))
            if np.argwhere(SV_temp > 5).size == 0:
                merge_list.append([i,j])
    merge_len = len(merge_list)
    if merge_len != 0:
        for i in range(merge_len):
            for j in range(merge_len):
                x = list(set(merge_list[i]+merge_list[j]))
                y = len(merge_list[j])+len(merge_list[i])
                if i == j:
                    break
                elif len(x) < y:
                    merge_list[i] = x
                    merge_list[j] = [-1]
        merge_list = [i for i in merge_list if i != [-1]]
        for m in merge_list:
            inx_temp = list()
            for i in m:
                inx_temp += compression_set_inx[i]
            compression_set_inx.append(inx_temp)
            cs_N = np.r_[cs_N, np.array([[len(inx_temp)]])]
            sq_temp = np.array([np.sum(cs_SUMSQ[m], axis=0)])
            sum_temp = np.array([np.sum(cs_SUM[m], axis=0)])
            # print (np.array([[len(inx_temp)]]).shape, sum_temp.shape, sq_temp.shape)
            # print (cs_N.shape, cs_SUM.shape, cs_SUMSQ.shape)
            cs_SUMSQ = np.r_[cs_SUMSQ, sq_temp]
            cs_SUM = np.r_[cs_SUM, sum_temp]
            cs_CEN = np.r_[cs_CEN, sum_temp / len(inx_temp)]
            cs_SV = np.r_[cs_SV, np.sqrt((sq_temp / len(inx_temp)) - np.power(sum_temp / len(inx_temp), 2))]
        delete_line = sorted(sum(merge_list, []),key=int,reverse = True)
        for l in delete_line:
            del compression_set_inx[l]
        cs_N = np.delete(cs_N, delete_line, axis=0)
        cs_SUM = np.delete(cs_SUM, delete_line, axis=0)
        cs_SUMSQ = np.delete(cs_SUMSQ, delete_line, axis=0)
        cs_CEN = np.delete(cs_CEN, delete_line, axis=0)
        cs_SV = np.delete(cs_SV, delete_line, axis=0)

    # If this is the last run (after the last chunk of data), merge CS clusters with DS clusters that have a Mahalanobis Distance < 2√𝑑.
    if end == n_data:
        delete_line = list()
        for i in range(len(compression_set_inx)):
            ds_times = np.abs(ds_CEN - cs_CEN[i]) / ds_SV
            ds_min_line = np.argmax(np.bincount(ds_times.argmin(axis=0)))
            ds_times = ds_times[ds_min_line]
            if np.argwhere(ds_times > 2).size == 0:
                inx = compression_set_inx[i]
                discard_set[inx] = ds_min_line
                ds_count += len(compression_set_inx[i])
                cs_count -= len(compression_set_inx[i])
                delete_line.append(i)
        delete_line = sorted(delete_line, key = int, reverse = True)
        for l in delete_line:
            del compression_set_inx[l]

    start = end
    end = start + int(n_data * percentage)
    metadata = (ds_count, len(compression_set_inx), cs_count, retained_set_inx.size)
    print (metadata)
    intermediate_res.append(metadata)
# Test
testData = rawData.map(lambda x: int(x[1])).collect()
score = normalized_mutual_info_score(testData, discard_set)
print (score)

# Print
fileOfOutput = open(OUTPUT, 'w')
outputStr = "The intermediate results:\n"
for i in range(len(intermediate_res)):
    r = intermediate_res[i]
    outputStr = outputStr + "Round " + str(i+1) + ": " + str(r[0]) + "," + str(r[1]) + "," + str(r[2]) + "," + str(r[3]) + "\n"
outputStr += "\nThe clustering results:\n"
for i in range(len(discard_set)):
    r = str(i) + "," + str(int(discard_set[i])) + "\n"
    outputStr += r
fileOfOutput.write(outputStr)
fileOfOutput.close()

timeEnd = time.time()
print("Duration: %f sec" % (timeEnd - timeStart))

# bin/spark-submit \
# --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:conf/log4j.xml" \
# --conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=file:conf/log4j.xml" \
# ../inf553-hw5/jingyue_fu_bfr.py ../inf553-hw5/input/hw5_clustering.txt 10 ../inf553-hw5/output/jingyue_fu_bfr.txt
