from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_json, struct
from pyspark.sql.types import StructType, StructField, DoubleType
import os

HADOOP_HOME = r"C:\hadoop"
HADOOP_BIN = r"C:\hadoop\bin"

os.environ["HADOOP_HOME"] = HADOOP_HOME
os.environ["hadoop.home.dir"] = HADOOP_HOME

path_lower = os.environ.get("PATH", "").lower()
if HADOOP_BIN.lower() not in path_lower:
    os.environ["PATH"] = os.environ.get("PATH", "") + ";" + HADOOP_BIN

spark = SparkSession.builder \
    .appName("Domashna3") \
    .master("local[*]") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3") \
    .getOrCreate()

features = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity",
    "Fruits", "Veggies", "HvyAlcoholConsump",
    "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex",
    "Age", "Education", "Income"
]

label = "Diabetes_binary"

model_path = r".\spark_models\best_model_Gradient%20Boosting"
model = PipelineModel.load(model_path)

schema = StructType([StructField(f, DoubleType(), True) for f in features])

kafka_df = (spark.readStream.format("kafka").option("kafka.bootstrap.servers", "localhost:9092")
            .option("subscribe", "health_data").load())

parsed_df = kafka_df.selectExpr("CAST(value AS STRING)").select(from_json(col("value"), schema).alias("data")).select("data.*")

preds = model.transform(parsed_df)
out_df = preds.select(*[col(f) for f in features],
    col("prediction").cast("int").alias("predicted_class"),
    col("probability").alias("probability")
)
out_kafka_df = out_df.select(to_json(struct(*out_df.columns)).alias("value"))

query = (out_kafka_df.writeStream
         .format("kafka")
         .option("kafka.bootstrap.servers", "localhost:9092")
         .option("topic", "health_data_predicted")
         .option("checkpointLocation", "checkpoints/health_data_predicted_sparkmodel")
         .outputMode("append")
         .start())

query.awaitTermination()
