from pyspark import SparkContext, SparkConf
import os, sys
import time
from operator import add

timeStart = time.time()

INPUT = sys.argv[1]
N_CLUSTER = int(sys.argv[2])
OUTPUT = sys.argv[3]

# Data Process: Creat baskets
sc = SparkContext('local[*]', 'CD_GF')
rawData = sc.textFile(INPUT, None, False)
rawData = rawData.map(lambda x: x.split(',')).map(lambda x: (int(x[1]), list(map(eval,x[2:])))).sortByKey()
print rawData.first()


# Print
# fileOfOutput = open(OUTPUT, 'w')
# outputStr = ""
# fileOfOutput.close()

timeEnd = time.time()
print "Duration: %f sec" % (timeEnd - timeStart)

# bin/spark-submit \
# --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:conf/log4j.xml" \
# --conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=file:conf/log4j.xml" \
# ../inf553-hw5/jingyue_fu_bfr.py ../inf553-hw5/input/hw5_clustering.txt 10 ../inf553-hw5/output/jingyue_fu_bfr.txt
