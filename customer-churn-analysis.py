from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnAnalysis").getOrCreate()

# Load dataset
data_path = "customer_churn.csv"
raw_df = spark.read.csv(data_path, header=True, inferSchema=True)

# Output file path
output_file_path = "model_outputs.txt"

# Task 1: Data Preprocessing and Feature Engineering

def preprocess_customer_data(raw_df):
    cleaned_df = raw_df.withColumn(
        "TotalCharges",
        when(col("TotalCharges").isNull(), 0).otherwise(col("TotalCharges").cast("double"))
    )

    categorical_columns = ["gender", "PhoneService", "InternetService"]
    string_indexers = [StringIndexer(inputCol=col_name, outputCol=f"{col_name}Index") for col_name in categorical_columns]
    one_hot_encoders = [OneHotEncoder(inputCol=f"{col_name}Index", outputCol=f"{col_name}Vec") for col_name in categorical_columns]

    label_indexer = StringIndexer(inputCol="Churn", outputCol="label")

    numerical_columns = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    encoded_columns = [f"{col_name}Vec" for col_name in categorical_columns]
    all_features = numerical_columns + encoded_columns

    assembler = VectorAssembler(inputCols=all_features, outputCol="features")

    pipeline_stages = string_indexers + one_hot_encoders + [label_indexer, assembler]
    pipeline = Pipeline(stages=pipeline_stages)
    pipeline_model = pipeline.fit(cleaned_df)
    transformed_df = pipeline_model.transform(cleaned_df)

    # ✅ Write preprocessing output to file
    with open(output_file_path, "a") as f:
        f.write("=== Data Preprocessing ===\n")
        f.write("Sample processed rows (features and label):\n")
        for row in transformed_df.select("features", "label").take(5):
            f.write(f"{row}\n")
        f.write("\n")

    return transformed_df.select("features", "label")

# Task 2: Train and Evaluate Logistic Regression Model
def train_and_evaluate_logistic_regression(processed_df, output_file_path):
    train_data, test_data = processed_df.randomSplit([0.8, 0.2], seed=42)
    logistic_model = LogisticRegression()
    trained_model = logistic_model.fit(train_data)
    predictions = trained_model.transform(test_data)

    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    auc_score = evaluator.evaluate(predictions)

    with open(output_file_path, "a") as f:
        f.write("=== Logistic Regression ===\n")
        f.write(f"AUC: {auc_score:.4f}\n\n")

# Task 3: Feature Selection using Chi-Square Test
def perform_feature_selection(processed_df, output_file_path):
    selector = ChiSqSelector(
        numTopFeatures=5,
        featuresCol="features",
        outputCol="selectedFeatures",
        labelCol="label"
    )
    selected_features_df = selector.fit(processed_df).transform(processed_df)

    with open(output_file_path, "a") as f:
        f.write("=== Feature Selection (Chi-Square) ===\n")
        f.write("Top 5 selected features (first 5 rows):\n")
        for row in selected_features_df.select("selectedFeatures", "label").take(5):
            f.write(f"{row}\n")
        f.write("\n")

# Task 4: Hyperparameter Tuning and Model Comparison
def tune_and_compare_classification_models(processed_df, output_file_path):
    train_data, test_data = processed_df.randomSplit([0.8, 0.2], seed=42)
    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

    model_candidates = {
        "LogisticRegression": LogisticRegression(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "GBTClassifier": GBTClassifier()
    }

    hyperparam_grids = {
        "LogisticRegression": ParamGridBuilder().addGrid(model_candidates["LogisticRegression"].regParam, [0.01, 0.1]).build(),
        "DecisionTree": ParamGridBuilder().addGrid(model_candidates["DecisionTree"].maxDepth, [3, 5]).build(),
        "RandomForest": ParamGridBuilder().addGrid(model_candidates["RandomForest"].numTrees, [10, 20]).build(),
        "GBTClassifier": ParamGridBuilder().addGrid(model_candidates["GBTClassifier"].maxIter, [10, 20]).build(),
    }

    best_auc = 0.0
    best_model_type = ""
    best_model_instance = None

    with open(output_file_path, "a") as f:
        f.write("=== Model Tuning and Comparison ===\n")

        for model_name, model_instance in model_candidates.items():
            cross_validator = CrossValidator(
                estimator=model_instance,
                estimatorParamMaps=hyperparam_grids[model_name],
                evaluator=evaluator,
                numFolds=5
            )
            cv_model = cross_validator.fit(train_data)
            auc_score = evaluator.evaluate(cv_model.transform(test_data))
            f.write(f"{model_name} AUC: {auc_score:.4f}\n")
            if auc_score > best_auc:
                best_auc = auc_score
                best_model_type = model_name
                best_model_instance = cv_model.bestModel

        f.write(f"Best model: {best_model_type} with AUC = {best_auc:.4f}\n\n")

# Reset the output file with header
with open(output_file_path, "w") as f:
    f.write("Customer Churn Modeling output\n")
    f.write("==============================\n\n")

# Execute All Tasks
processed_data_df = preprocess_customer_data(raw_df)
train_and_evaluate_logistic_regression(processed_data_df, output_file_path)
perform_feature_selection(processed_data_df, output_file_path)
tune_and_compare_classification_models(processed_data_df, output_file_path)

# Stop the Spark session
spark.stop()