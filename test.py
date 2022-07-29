# Wine Quality Prediction Model Validation

import numpy as np
import pandas as pd
import sys

def createSparkSession(appName):
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.sql import SparkSession

    conf = SparkConf().setAppName(appName).setMaster("local")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    return sc, spark


def readDataSet(spark):
    path = sys.argv[1]
    df = spark.read.format("csv").load(path, header = True, sep =";")

    return df


def convertDf(df):
    from pyspark.sql.functions import col

    for name in df.columns[0:-1]+['""""quality"""""']:
        df = df.withColumn(name, col(name).cast('float'))

    df = df.withColumnRenamed('""""quality"""""', "label")

    return df


def getFeatures(df):
    features = np.array(df.select(df.columns[0:-1]).collect())
    label = np.array(df.select('label').collect())

    return features, label


def featureVector(df):
    from pyspark.ml.feature import VectorAssembler

    VectorAssembler = VectorAssembler(inputCols = df.columns[0:-1], outputCol = 'features')
    df_fv = VectorAssembler.transform(df)
    df_fv = df_fv.select(['features', 'label'])

    return df_fv


def convertLabeledPoints(sc, features, label):
    from pyspark.mllib.regression import LabeledPoint

    points = [LabeledPoint(y,x) for x,y in zip(features, label)]

    return sc.parallelize(points)


def randomForestAlg(data_set):
    from pyspark.mllib.tree import RandomForest

    rf_model = RandomForest.trainClassifier(data_set, categoricalFeaturesInfo={}, numClasses = 10, numTrees = 21, maxDepth = 25, maxBins = 30, impurity = 'gini')
    
    return rf_model

def loadRandomForestModel(sc):
    from pyspark.mllib.tree import RandomForestModel

    rf_model = RandomForestModel.load(sc, "/cs643-dataset/training_model.model/")

    return rf_model


def rfPredictions(rf_model, data):
    predictions = rf_model.predict(data.map(lambda x: x.features))
    lap = data.map(lambda lp: lp.label).zip(predictions)
    lap_df = lap.toDF()
    label_pred = lap.toDF(["label", "Prediction"])
    label_pred.show()
    label_pred_df = label_pred.toPandas()

    return lap, label_pred_df


def calF1Score(df):
    from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score

    f1Score = f1_score(df['label'], df['Prediction'], average='micro')
    print("F1-Score = ", f1Score)
    print(confusion_matrix(df['label'], df['Prediction']))
    print(classification_report(df['label'], df['Prediction']))
    print("Accuracy", accuracy_score(df['label'], df['Prediction']))


def showdf(df):
    df.printSchema()
    df.show()

def logMessage(msg):
    print("################################################")
    print(" ")
    print("Message: ", msg)
    print(" ")
    print("################################################")

def main():
    sc, spark = createSparkSession("winequality")

    # Read dataset
    logMessage("Reading Data Set")
    df = readDataSet(spark)
    showdf(df)

    # df shows that column "quality" is int64 so we will convert it to "label"
    logMessage("Converting Data")
    df = convertDf(df)
    showdf(df)
    
    # Grab Features and Labels
    logMessage("Grabbing Features")
    features, label = getFeatures(df)
    print("Features: ", features)
    print("Label: ", label)

    # Create Feature Vector
    logMessage("Generate Feature Vector")
    df_fv = featureVector(df)
    print("FV: ", df_fv)
    
    # Parallelize data points
    logMessage("Parallelize Data Points")
    data = convertLabeledPoints(sc, features, label)
    print("Data: ", data)

    # # Split into training and test set
    # logMessage("Splitting Data set into train and Test")
    # train_set, test_set = data.randomSplit([0.8, 0.2], seed = 30)
    # print("Train Set: ", train_set)
    # print("Test Set: ", test_set)

    # Random Forest Classifier
    logMessage("Random Forest Model")
    #rf_model = randomForestAlg(train_set)
    rf_model = loadRandomForestModel(sc)
    print("Loaded Model Successfully")

    # Predictions
    logMessage("Making Predictions")
    lap, label_pred_df = rfPredictions(rf_model, data)

    # F1_Score
    logMessage("Calculating F1_Score")
    calF1Score(label_pred_df)

    # Test error
    logMessage("Calculating Test Error")
    err = lap.filter(lambda lp: lp[0] != lp[1]).count() / float (data.count())
    print("Test Error = ", str(err))

    # # Save Model
    # logMessage("Saving Model")
    # rf_model.save(sc, 's3://dataset-cs643/training_model.model')


if __name__ == "__main__":
    main()
