from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
import os

HADOOP_BIN = r"C:\hadoop\bin"
os.environ["HADOOP_HOME"] = r"C:\hadoop"
path_lower = os.environ.get("PATH", "").lower()
if HADOOP_BIN.lower() not in path_lower:
    os.environ["PATH"] = os.environ.get("PATH", "") + ";" + HADOOP_BIN

print("HADOOP_HOME (python):", os.environ.get("HADOOP_HOME"))
print("PATH has hadoop\\bin (python):", HADOOP_BIN.lower() in os.environ.get("PATH","").lower())

def build_pipeline(estimator, features):
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    return Pipeline(stages=[assembler, estimator])


def crossval_fit(pipeline, param_grid, train_df, evaluator, folds=3, seed=42, parallelism=4):
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=folds,
        seed=seed,
        parallelism=parallelism
    )
    return cv.fit(train_df)

def main():
    spark = SparkSession.builder \
        .appName("Domashna3") \
        .master("local[*]") \
        .getOrCreate()

    df = spark.read.csv("offline.csv", header=True, inferSchema=True)
    df.show(10)

    features = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
        "Stroke", "HeartDiseaseorAttack", "PhysActivity",
        "Fruits", "Veggies", "HvyAlcoholConsump",
        "AnyHealthcare", "NoDocbcCost", "GenHlth",
        "MentHlth", "PhysHlth", "DiffWalk", "Sex",
        "Age", "Education", "Income"
    ]

    data = df.withColumn("label", col("Diabetes_binary").cast("double"))
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    evaluator = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction", metricName="f1")

    lr = LogisticRegression(featuresCol="features", labelCol="label")
    lr_pipe = build_pipeline(lr, features)
    lr_grid = (ParamGridBuilder().
                 addGrid(lr.regParam, [0.001, 0.01, 0.1])
                 .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                 .addGrid(lr.maxIter, [10, 30, 80])
                 .build())

    lr_cv_model = crossval_fit(lr_pipe, lr_grid, train_data, evaluator)
    lr_best = lr_cv_model.bestModel
    lr_f1 = evaluator.evaluate(lr_best.transform(test_data))
    print(f"Logistic Regression F1-score: {lr_f1}")

    rf = RandomForestClassifier(featuresCol="features", labelCol="label", seed=42)
    rf_pipe = build_pipeline(rf, features)

    rf_grid = (ParamGridBuilder()
               .addGrid(rf.numTrees, [50, 100])
               .addGrid(rf.maxDepth, [5, 10])
               .addGrid(rf.featureSubsetStrategy, ["auto", "sqrt"])
               .build())

    rf_cv_model = crossval_fit(rf_pipe, rf_grid, train_data, evaluator)
    rf_best = rf_cv_model.bestModel
    rf_f1 = evaluator.evaluate(rf_best.transform(test_data))
    print(f"Random Forest F1-score: {rf_f1}")

    gbt = GBTClassifier(featuresCol="features", labelCol="label", seed=42)
    gbt_pipe = build_pipeline(gbt, features)

    gbt_grid = (ParamGridBuilder()
                .addGrid(gbt.maxDepth, [3, 5])
                .addGrid(gbt.maxIter, [20, 50])
                .addGrid(gbt.stepSize, [0.05, 0.1])
                .build())

    gbt_cv_model = crossval_fit(gbt_pipe, gbt_grid, train_data, evaluator)
    gbt_best = gbt_cv_model.bestModel
    gbt_f1 = evaluator.evaluate(gbt_best.transform(test_data))
    print(f"Gradient Boosting F1-score: {gbt_f1}")

    best_models = [(lr_best, lr_f1, "Logistic Regression"),(rf_best, rf_f1, "Random Forest"),(gbt_best, gbt_f1, "Gradient Boosting")]
    model, f1, name = max(best_models, key=lambda x: x[1])
    print(f"Best model: {model} with F1-score {f1} and {name}")

    out_dir = Path("C:/Temp/spark_models") / f"best_model_{name}"
    out_uri = out_dir.resolve().as_uri()
    print("Saving model to:", out_uri)
    model.write().overwrite().save(out_uri)

if __name__ == '__main__':
    main()

#
# |Diabetes_binary|HighBP|HighChol|CholCheck| BMI|Smoker|Stroke|HeartDiseaseorAttack|PhysActivity|Fruits|Veggies|HvyAlcoholConsump|AnyHealthcare|NoDocbcCost|GenHlth|MentHlth|PhysHlth|DiffWalk|Sex| Age|Education|Income|
# +---------------+------+--------+---------+----+------+------+--------------------+------------+------+-------+-----------------+-------------+-----------+-------+--------+--------+--------+---+----+---------+------+
# |            0.0|   0.0|     0.0|      1.0|28.0|   1.0|   0.0|                 0.0|         1.0|   1.0|    1.0|              0.0|          1.0|        0.0|    2.0|     0.0|     0.0|     0.0|1.0| 2.0|      4.0|   5.0|
# |            0.0|   1.0|     0.0|      1.0|23.0|   1.0|   0.0|                 0.0|         1.0|   1.0|    1.0|              0.0|          1.0|        0.0|    2.0|     0.0|     0.0|     0.0|1.0|13.0|      4.0|   7.0|
# |            0.0|   1.0|     1.0|      1.0|29.0|   0.0|   0.0|                 0.0|         1.0|   1.0|    1.0|              0.0|          1.0|        0.0|    1.0|     0.0|     0.0|     0.0|1.0| 9.0|      6.0|   8.0|
# |            0.0|   1.0|     1.0|      1.0|39.0|   0.0|   0.0|                 0.0|         0.0|   0.0|    0.0|              0.0|          1.0|        0.0|    4.0|     0.0|     0.0|     0.0|1.0| 7.0|      4.0|   7.0|
# |            0.0|   0.0|     1.0|      1.0|16.0|   1.0|   0.0|                 0.0|         1.0|   1.0|    1.0|              0.0|          1.0|        1.0|    5.0|    30.0|    30.0|     1.0|0.0| 7.0|      5.0|   1.0|
# |            0.0|   1.0|     0.0|      1.0|32.0|   0.0|   0.0|                 0.0|         1.0|   0.0|    1.0|              0.0|          1.0|        0.0|    2.0|     0.0|     0.0|     0.0|1.0| 7.0|      6.0|   8.0|
# |            1.0|   1.0|     0.0|      1.0|37.0|   1.0|   0.0|                 0.0|         0.0|   0.0|    1.0|              0.0|          1.0|        0.0|    4.0|     0.0|     0.0|     0.0|0.0| 4.0|      5.0|   4.0|
# |            0.0|   0.0|     0.0|      1.0|27.0|   0.0|   0.0|                 0.0|         1.0|   1.0|    0.0|              0.0|          1.0|        0.0|    2.0|     3.0|     1.0|     0.0|1.0| 5.0|      5.0|   6.0|
# |            0.0|   0.0|     0.0|      1.0|24.0|   0.0|   0.0|                 0.0|         1.0|   1.0|    0.0|              0.0|          1.0|        0.0|    3.0|    10.0|    15.0|     1.0|0.0| 5.0|      6.0|   8.0|
# |            0.0|   0.0|     0.0|      1.0|23.0|   1.0|   0.0|                 0.0|         1.0|   1.0|    1.0|              1.0|          1.0|        0.0|    2.0|     0.0|     3.0|     0.0|0.0| 6.0|      6.0|   8.0|
# +---------------+------+--------+---------+----+------+------+--------------------+------------+------+-------+-----------------+-------------+-----------+-------+--------+--------+--------+---+----+---------+------+
# only showing top 10 rows
#
# Logistic Regression F1-score: 0.8307658510684387
# Random Forest F1-score: 0.8241547930166929
# Gradient Boosting F1-score: 0.8343585722026596
# Best model: PipelineModel_58bdafdf4ea4 with F1-score 0.8343585722026596 and Gradient Boosting