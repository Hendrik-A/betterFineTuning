import os
import argparse
import math
import pyspark
from pyspark.sql import functions as F
from pyspark.sql import types as spark_types

from rouge_score import rouge_scorer

def sorter(text, abstract):
	sen_scores = [0] * len(text)
	metrics = ['rouge1', 'rouge2', 'rougeL']
	rg_scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
	for i, sentence in enumerate(text):
		results = rg_scorer.score(abstract, sentence)
		sen_scores[i] = results['rouge1'][2] + 2*results['rouge2'][2] + results['rougeL'][2]
	sorted_ids = [i[0] for i in sorted(enumerate(sen_scores), key=lambda k: k[1], reverse=True)]
	sorted_text = [text[i] for i in sorted_ids]
	return sorted_text
	
def topX(text, percentile):
	if percentile == 1:
		return text[:math.ceil(len(text)/4)]
	elif percentile == 2:
		return text[:math.ceil(len(text)/2)]
	elif percentile == 3:
		return text[:math.ceil(len(text)/4*3)]
	else:
		raise Exception("invalid percentile")

def getOne():
	return 1
def getTwo():
	return 2
def getThree():
	return 3

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
	
	data_path = os.path.join(args.data_root, 'train.txt')
	task_output_dir = os.path.join(args.data_root, "processed/betterFineTuning")
	if not os.path.exists(task_output_dir):
		os.makedirs(task_output_dir)
	
	sorter_udf = F.udf(sorter, spark_types.ArrayType(spark_types.StringType()))
	getOne_udf = F.udf(getOne, spark_types.IntegerType())
	getTwo_udf = F.udf(getTwo, spark_types.IntegerType())
	getThree_udf = F.udf(getThree, spark_types.IntegerType())
	topX_udf = F.udf(topX, spark_types.ArrayType(spark_types.StringType()))

	for data_path, prefix in zip(data_paths, data_prefixes):

		df = spark.read.json(data_path) \
					.repartition(500, "article_id")
	
		df = df.drop("labels", "section_names", "sections")
	
		df = df.withColumn("abstract", F.concat_ws(" ", F.col("abstract_text"))).withColumn("abstract", F.regexp_replace("abstract", "<\/?S>", "")).drop("abstract_text")

		df = df.withColumn("sorted_text", sorter_udf(F.col('article_text'), F.col('abstract'))) \
			.withColumn("article_text", F.concat_ws(" ", F.col("article_text"))) \
			.withColumn("top25", getOne_udf()).withColumn("top50", getTwo_udf()).withColumn("top75", getThree_udf()) \
			.withColumn("top25", topX_udf(F.col('sorted_text'), F.col('top25'))).withColumn("top50", topX_udf(F.col('sorted_text'), F.col('top50'))).withColumn("top75", topX_udf(F.col('sorted_text'), F.col('top75'))) \
			.withColumn("sorted_text", F.concat_ws(" ", F.col("sorted_text"))) \
			.withColumn("top25", F.concat_ws(" ", F.col("top25"))).withColumn("top50", F.concat_ws(" ", F.col("top50"))).withColumn("top75", F.concat_ws(" ", F.col("top75")))

		df.write.json(
			path=os.path.join(task_output_dir, prefix),
			mode="overwrite")

if __name__ == "__main__":
	main()

