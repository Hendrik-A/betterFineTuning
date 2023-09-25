import os
import argparse
import math
import pyspark
from pyspark.sql import functions as F
from pyspark.sql import types as spark_types

def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_root", type=str, help="")

	args, unknown = parser.parse_known_args()
	return args, unknown

def main():

	conf = pyspark.SparkConf()
	sc = pyspark.SparkContext(conf=conf)
	spark = pyspark.sql.SparkSession(sc)

	args, unknown = read_args()

	train_data = os.path.join(args.data_root, 'train.txt')
	val_data = os.path.join(args.data_root, 'val.txt')
	test_data = os.path.join(args.data_root, 'test.txt')

	data_prefixes = ['train', 'val', 'test']
	data_paths = [train_data, val_data, test_data]
	
	task_output_dir = os.path.join(args.data_root, "processed/filtered")
	if not os.path.exists(task_output_dir):
		os.makedirs(task_output_dir)

	for data_path, prefix in zip(data_paths, data_prefixes):

		df = spark.read.json(data_path) \
					.repartition(500, "article_id")
	
		df = df.drop("labels", "section_names", "sections", "LEDtokenizer", "PXtokenizer")
	
		df = df.withColumn("abstract", F.concat_ws(" ", F.col("abstract_text"))).withColumn("abstract", F.regexp_replace("abstract", "<\/?S>", "")).drop("abstract_text").withColumn("article_text", F.concat_ws(" ", F.col("article_text")))

		df.write.json(
			path=os.path.join(task_output_dir, prefix),
			mode="overwrite")

if __name__ == "__main__":
	main()
