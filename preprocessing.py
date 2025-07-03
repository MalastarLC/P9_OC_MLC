# -*- coding: utf-8 -*-

# 1. Imports
import pandas as pd
from PIL import Image
import numpy as np
import io
import os
from pathlib import Path

# Important: Matplotlib is for local plotting, not needed on EMR for processing.
# We comment it out or remove it to avoid potential display/backend errors on a headless server.
# import matplotlib.pyplot as plt

# Here are the steps we follow

# We create a spark session hosted in the cloud,
# Spark's ressource manager YARN automatically handles distributing the work 
# across the cores and nodes that we configured when we created the cluster
# we tell spark how to load the data (binary images)
# and where to find them (my S3 bucket)
# we create an intermediary spark df that stores the names of the images we want to use and their paths, 
# their labels, the actual content of the image which will be used by our preprocessing function
# then we do an initial load of the model we want to use as a feature extractor,
# we remove its top layer
# then we do brodcast_weights = sc.broadcast(new_model.get_weights()) to be able to inject its weights 
# when the feature extraction will be running at the very beginning 
# so that every worker does not have to get the weights at every step
# then we define a preprocessing pipeline to prepare the data to be used by the workers
# we prepare a featurize function which outputs a single features vector easier to store into a spark df
# then we create a @pandas user definied function which return an array of flattened features vectors
# then we can call on the actual operation to get our features and then we do an initial pca on it 
# then we add a step to automatically compute the number of components to retain 95% of the variance
# and then we obtain the actual features with their dimensions reduced

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import Model
from pyspark.sql.functions import col, pandas_udf, PandasUDFType, element_at, split, udf
from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors, VectorUDT

def main():
    # 2. Spark Session Creation (Modified for EMR)
    # The .master("local[*]") is removed. EMR's YARN will manage resources.
    spark = (SparkSession
             .builder
             .appName('P9-Fruit-Preprocessing')
             .getOrCreate())
    sc = spark.sparkContext
    
    # Announce the Spark UI URL for logging purposes (useful for debugging)
    print(f"Spark UI available at: {sc.uiWebUrl}")

    # 3. Define S3 Paths
    # Replace 'your-bucket-name' with the actual name of your S3 bucket
    bucket_name = "p9-fruit-app-maxime-lacroux-20250703" 
    input_data_path = f"s3://{bucket_name}/data/Test1"
    output_features_path = f"s3://{bucket_name}/results/features_raw.parquet"
    output_pca_path = f"s3://{bucket_name}/results/features_pca_optimal.parquet"
    
    print(f"Reading images from: {input_data_path}")
    print(f"Will write raw features to: {output_features_path}")
    print(f"Will write PCA features to: {output_pca_path}")

    # 4. Load Images from S3
    images_df = (spark.read.format("binaryFile")
                 .option("pathGlobFilter", "*.jpg")
                 .option("recursiveFileLookup", "true")
                 .load(input_data_path))

    # Extract labels from path
    images_df = images_df.withColumn('label', element_at(split(col('path'), '/'), -2))
    images_df.persist() # We keep the dataframe created in the cache since it is reused when extracting the features

    # when it comes to Dataframes spark does not actually creates it but defines a plan on how it would create it 
    # after some processing so we need to keep this in memory 
    # or it would have to compute everything again if it needs to reuse it

    # In Spark, a DataFrame represents a logical plan for computation, not the data itself. 
    # Operations that define a DataFrame are called Transformations, and they are evaluated lazily. 
    # The computation only happens when an Action is called. 
    # If a DataFrame is needed by multiple actions, Spark will re-compute it from scratch each time 
    # unless we explicitly instruct it to persist() or cache() the result in memory after the first computation.
    
    print("Image DataFrame schema:")
    images_df.printSchema()
    print(f"Found {images_df.count()} images.")

    # 5. Model Broadcasting
    # Load the base model once on the driver node
    model = MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    new_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    
    # Broadcast the weights to all worker nodes
    broadcasted_weights = sc.broadcast(new_model.get_weights())

    def model_fn():
        """
        Returns a MobileNetV2 model with top layer removed 
        and broadcasted pretrained weights.
        """
        model = MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
        # No need to set trainable to False, as we are only doing inference
        new_model = Model(inputs=model.input, outputs=model.layers[-2].output)
        new_model.set_weights(broadcasted_weights.value)
        return new_model



    # 6. Define Preprocessing and Featurization UDF
    def preprocess(content):
        img = Image.open(io.BytesIO(content)).resize([224, 224])
        arr = img_to_array(img)
        return preprocess_input(arr)

    def featurize_series(model, content_series):
        input_data = np.stack(content_series.map(preprocess))
        preds = model.predict(input_data)
        output = [p.flatten() for p in preds]
        return pd.Series(output)

    # array<float> defines the type of the new column that the Pandas UDF will return 
    # By knowing the exact return type in advance, Spark can build a highly optimized execution plan 
    # and manage memory efficiently without having to guess what the function will produce

    # SCALAR: This means it's a "scalar" UDF. 
    # It takes one or more columns as input and produces exactly one column as output. 
    # The number of rows in the output will be the same as the number of rows in the input batch. 
    # This is the most common type of UDF.

    # Ce que fait ITER c'est que sans lui a chaque batch de data l'UDF devrait etre relancé
    # ce qui est inefficace alors que avec iter les batchs de données sont recupérés avec une iteration dans 
    # l'udf ce qui ne permet de n'avoir a exécuter l'udf qu'une seule fois


    @pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
    def featurize_udf(content_series_iter):
        model = model_fn()
        for content_series in content_series_iter:
            yield featurize_series(model, content_series)

    # 7. Feature Extraction

    # .repartition(16): We shuffle the data into 16 partitions to ensure good parallelism across the cluster.

    print("Extracting features using MobileNetV2...")
    features_df = images_df.repartition(16).select(
        col("path"),
        col("label"),
        featurize_udf("content").alias("features")
    )
    
    # Write raw features to S3 (optional but good practice)
    features_df.write.mode("overwrite").parquet(output_features_path)
    print(f"Raw features saved to {output_features_path}")

    # 8. PCA Dimensionality Reduction
    print("Performing PCA...")
    
    # Convert array<float> to vector for PCA
    array_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
    df_with_vectors = features_df.withColumn("features_vec", array_to_vector_udf(col("features")))

    # Step 1: Fit PCA with a high number of components to find optimal k
    # Since the dataset is small, 200 is a safe upper bound.
    K_COMPONENTS = 200 
    pca = PCA(k=K_COMPONENTS, inputCol="features_vec", outputCol="pca_features_temp")
    pca_model_temp = pca.fit(df_with_vectors)
    
    # We add a step to ensure we have reached 95% otherwise
    # The script will not crash but we will end up with only 1 component
    # Find optimal k for 95% variance
    cumulative_variance = np.cumsum(pca_model_temp.explainedVariance)
    
    max_variance_achieved = cumulative_variance[-1] # The variance explained by all components

    print(f"Maximum variance explained by {K_COMPONENTS} components: {max_variance_achieved:.2%}")

    # Check if the 95% threshold was met
    if max_variance_achieved < 0.95:
        print("WARNING: 95% variance not achieved with the initial number of components.")
        # Use all the components we calculated
        k_optimal = K_COMPONENTS
        print(f"Falling back to using all {K_COMPONENTS} components.")
    else:
        # This is the original logic, which is safe to run now
        k_optimal = np.argmax(cumulative_variance >= 0.95) + 1
        print(f"Optimal number of components to retain 95% variance: {k_optimal}")
    
    # Step 2: Rerun PCA with the optimal k
    pca_final = PCA(k=int(k_optimal), inputCol="features_vec", outputCol="pca_features")
    pca_model_final = pca_final.fit(df_with_vectors)
    
    df_pca_final = pca_model_final.transform(df_with_vectors)

    # 9. Save Final Results
    print(f"Saving final PCA-reduced features to {output_pca_path}...")
    (df_pca_final
     .select("path", "label", "pca_features")
     .write.mode("overwrite")
     .parquet(output_pca_path))
     
    print("Processing complete.")
    
    # Release the persisted DataFrame from memory
    images_df.unpersist()

    # Stop the Spark session
    spark.stop()

# Entry point for the script
if __name__ == "__main__":
    main() # This allows us to make sure the script only runs when executed diresctly 
           # but not when its for example called in another script