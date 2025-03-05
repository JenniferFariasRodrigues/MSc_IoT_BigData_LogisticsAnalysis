from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Spark Testing").master("local[*]").config("spark.driver.memory", "4g").config("spark.executor.memory", "4g").getOrCreate()

data = [(1, "Jennifer"), (2, "Mestrado")]
df = spark.createDataFrame(data, ["id", "nome"])
spark.sparkContext.setLogLevel("ERROR")


df.show()


# from pyspark.sql import SparkSession
# spark = SparkSession.builder.appName(" Spark testing").getOrCreate()
# data=[(1, "Jennifer"), (2,"Mestrado")]
# df=spark.createDataFrame(data, ["id", "nome"])
#
#
# df.show()
#
#
