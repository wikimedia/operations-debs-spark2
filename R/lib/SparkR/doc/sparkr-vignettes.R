## ----setup, include=FALSE------------------------------------------------
library(knitr)
opts_hooks$set(eval = function(options) {
  # override eval to FALSE only on windows
  if (.Platform$OS.type == "windows") {
    options$eval = FALSE
  }
  options
})

## ---- message=FALSE------------------------------------------------------
library(SparkR)

## ---- include=FALSE------------------------------------------------------
install.spark()

## ---- message=FALSE, results="hide"--------------------------------------
sparkR.session()

## ------------------------------------------------------------------------
cars <- cbind(model = rownames(mtcars), mtcars)
carsDF <- createDataFrame(cars)

## ------------------------------------------------------------------------
head(carsDF)

## ------------------------------------------------------------------------
carsSubDF <- select(carsDF, "model", "mpg", "hp")
carsSubDF <- filter(carsSubDF, carsSubDF$hp >= 200)
head(carsSubDF)

## ------------------------------------------------------------------------
carsGPDF <- summarize(groupBy(carsDF, carsDF$gear), count = n(carsDF$gear))
head(carsGPDF)

## ------------------------------------------------------------------------
carsGP <- collect(carsGPDF)
class(carsGP)

## ------------------------------------------------------------------------
model <- spark.glm(carsDF, mpg ~ wt + cyl)

## ------------------------------------------------------------------------
summary(model)

## ---- eval=FALSE---------------------------------------------------------
#  write.ml(model, path = "/HOME/tmp/mlModel/glmModel")

## ---- eval=FALSE---------------------------------------------------------
#  sparkR.session.stop()

## ---- eval=FALSE---------------------------------------------------------
#  install.spark()

## ---- eval=FALSE---------------------------------------------------------
#  sparkR.session(sparkHome = "/HOME/spark")

## ---- eval=FALSE---------------------------------------------------------
#  spark_warehouse_path <- file.path(path.expand('~'), "spark-warehouse")
#  sparkR.session(spark.sql.warehouse.dir = spark_warehouse_path)

## ---- echo=FALSE, tidy = TRUE--------------------------------------------
paste("Spark", packageVersion("SparkR"))

## ---- eval=FALSE---------------------------------------------------------
#  sparkR.session(master = "spark://local:7077")

## ---- eval=FALSE---------------------------------------------------------
#  sparkR.session(master = "yarn")

## ------------------------------------------------------------------------
df <- as.DataFrame(faithful)
head(df)

## ---- eval=FALSE---------------------------------------------------------
#  sparkR.session(sparkPackages = "com.databricks:spark-avro_2.11:3.0.0")

## ---- eval=FALSE---------------------------------------------------------
#  df <- read.df(csvPath, "csv", header = "true", inferSchema = "true", na.strings = "NA")

## ------------------------------------------------------------------------
filePath <- paste0(sparkR.conf("spark.home"),
                         "/examples/src/main/resources/people.json")
readLines(filePath, n = 2L)

## ------------------------------------------------------------------------
people <- read.df(filePath, "json")
count(people)
head(people)

## ------------------------------------------------------------------------
printSchema(people)

## ------------------------------------------------------------------------
people <- read.json(paste0(Sys.getenv("SPARK_HOME"),
                           c("/examples/src/main/resources/people.json",
                             "/examples/src/main/resources/people.json")))
count(people)

## ---- eval=FALSE---------------------------------------------------------
#  write.df(people, path = "people.parquet", source = "parquet", mode = "overwrite")

## ---- eval=FALSE---------------------------------------------------------
#  sql("CREATE TABLE IF NOT EXISTS src (key INT, value STRING)")
#  
#  txtPath <- paste0(sparkR.conf("spark.home"), "/examples/src/main/resources/kv1.txt")
#  sqlCMD <- sprintf("LOAD DATA LOCAL INPATH '%s' INTO TABLE src", txtPath)
#  sql(sqlCMD)
#  
#  results <- sql("FROM src SELECT key, value")
#  
#  # results is now a SparkDataFrame
#  head(results)

## ------------------------------------------------------------------------
carsDF

## ------------------------------------------------------------------------
printSchema(carsDF)

## ------------------------------------------------------------------------
head(select(carsDF, "mpg"))

## ------------------------------------------------------------------------
head(filter(carsDF, carsDF$mpg < 20))

## ------------------------------------------------------------------------
numCyl <- summarize(groupBy(carsDF, carsDF$cyl), count = n(carsDF$cyl))
head(numCyl)

## ------------------------------------------------------------------------
carsDF_km <- carsDF
carsDF_km$kmpg <- carsDF_km$mpg * 1.61
head(select(carsDF_km, "model", "mpg", "kmpg"))

## ------------------------------------------------------------------------
carsSubDF <- select(carsDF, "model", "mpg", "cyl")
ws <- orderBy(windowPartitionBy("cyl"), "mpg")
carsRank <- withColumn(carsSubDF, "rank", over(rank(), ws))
head(carsRank, n = 20L)

## ------------------------------------------------------------------------
carsSubDF <- select(carsDF, "model", "mpg")
schema <- structType(structField("model", "string"), structField("mpg", "double"),
                     structField("kmpg", "double"))
out <- dapply(carsSubDF, function(x) { x <- cbind(x, x$mpg * 1.61) }, schema)
head(collect(out))

## ------------------------------------------------------------------------
out <- dapplyCollect(
         carsSubDF,
         function(x) {
           x <- cbind(x, "kmpg" = x$mpg * 1.61)
         })
head(out, 3)

## ------------------------------------------------------------------------
schema <- structType(structField("cyl", "double"), structField("max_mpg", "double"))
result <- gapply(
    carsDF,
    "cyl",
    function(key, x) {
        y <- data.frame(key, max(x$mpg))
    },
    schema)
head(arrange(result, "max_mpg", decreasing = TRUE))

## ------------------------------------------------------------------------
result <- gapplyCollect(
    carsDF,
    "cyl",
    function(key, x) {
         y <- data.frame(key, max(x$mpg))
        colnames(y) <- c("cyl", "max_mpg")
        y
    })
head(result[order(result$max_mpg, decreasing = TRUE), ])

## ------------------------------------------------------------------------
costs <- exp(seq(from = log(1), to = log(1000), length.out = 5))
train <- function(cost) {
  stopifnot(requireNamespace("e1071", quietly = TRUE))
  model <- e1071::svm(Species ~ ., data = iris, cost = cost)
  summary(model)
}

## ------------------------------------------------------------------------
model.summaries <- spark.lapply(costs, train)

## ------------------------------------------------------------------------
class(model.summaries)

## ---- include=FALSE------------------------------------------------------
ops <- options()
options(max.print=40)

## ------------------------------------------------------------------------
print(model.summaries[[2]])

## ---- include=FALSE------------------------------------------------------
options(ops)

## ------------------------------------------------------------------------
people <- read.df(paste0(sparkR.conf("spark.home"),
                         "/examples/src/main/resources/people.json"), "json")

## ------------------------------------------------------------------------
createOrReplaceTempView(people, "people")

## ------------------------------------------------------------------------
teenagers <- sql("SELECT name FROM people WHERE age >= 13 AND age <= 19")
head(teenagers)

## ------------------------------------------------------------------------
splitDF_list <- randomSplit(carsDF, c(0.7, 0.3), seed = 0)
carsDF_train <- splitDF_list[[1]]
carsDF_test <- splitDF_list[[2]]

## ------------------------------------------------------------------------
count(carsDF_train)
head(carsDF_train)

## ------------------------------------------------------------------------
count(carsDF_test)
head(carsDF_test)

## ---- warning=FALSE------------------------------------------------------
df <- createDataFrame(iris)
# Create a DataFrame containing two classes
training <- df[df$Species %in% c("versicolor", "virginica"), ]
model <- spark.logit(training, Species ~ ., regParam = 0.00042)
summary(model)

## ------------------------------------------------------------------------
fitted <- predict(model, training)

## ---- warning=FALSE------------------------------------------------------
df <- createDataFrame(iris)
# Note in this case, Spark infers it is multinomial logistic regression, so family = "multinomial" is optional.
model <- spark.logit(df, Species ~ ., regParam = 0.056)
summary(model)

## ---- warning=FALSE------------------------------------------------------
df <- createDataFrame(iris)
# fit a Multilayer Perceptron Classification Model
model <- spark.mlp(df, Species ~ ., blockSize = 128, layers = c(4, 3), solver = "l-bfgs", maxIter = 100, tol = 0.5, stepSize = 1, seed = 1, initialWeights = c(0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9))

## ---- include=FALSE------------------------------------------------------
ops <- options()
options(max.print=5)

## ------------------------------------------------------------------------
# check the summary of the fitted model
summary(model)

## ---- include=FALSE------------------------------------------------------
options(ops)

## ------------------------------------------------------------------------
# make predictions use the fitted model
predictions <- predict(model, df)
head(select(predictions, predictions$prediction))

## ------------------------------------------------------------------------
titanic <- as.data.frame(Titanic)
titanicDF <- createDataFrame(titanic[titanic$Freq > 0, -5])
naiveBayesModel <- spark.naiveBayes(titanicDF, Survived ~ Class + Sex + Age)
summary(naiveBayesModel)
naiveBayesPrediction <- predict(naiveBayesModel, titanicDF)
head(select(naiveBayesPrediction, "Class", "Sex", "Age", "Survived", "prediction"))

## ---- warning=FALSE------------------------------------------------------
library(survival)
ovarianDF <- createDataFrame(ovarian)
aftModel <- spark.survreg(ovarianDF, Surv(futime, fustat) ~ ecog_ps + rx)
summary(aftModel)
aftPredictions <- predict(aftModel, ovarianDF)
head(aftPredictions)

## ------------------------------------------------------------------------
gaussianGLM <- spark.glm(carsDF, mpg ~ wt + hp)
summary(gaussianGLM)

## ------------------------------------------------------------------------
gaussianFitted <- predict(gaussianGLM, carsDF)
head(select(gaussianFitted, "model", "prediction", "mpg", "wt", "hp"))

## ------------------------------------------------------------------------
y <- c(3.0, 6.0, 8.0, 5.0, 7.0)
x <- c(1.0, 2.0, 3.5, 3.0, 4.0)
w <- rep(1.0, 5)
data <- data.frame(y = y, x = x, w = w)
df <- createDataFrame(data)
isoregModel <- spark.isoreg(df, y ~ x, weightCol = "w")
isoregFitted <- predict(isoregModel, df)
head(select(isoregFitted, "x", "y", "prediction"))

## ------------------------------------------------------------------------
newDF <- createDataFrame(data.frame(x = c(1.5, 3.2)))
head(predict(isoregModel, newDF))

## ---- warning=FALSE------------------------------------------------------
df <- createDataFrame(longley)
gbtModel <- spark.gbt(df, Employed ~ ., type = "regression", maxDepth = 2, maxIter = 2)
summary(gbtModel)
predictions <- predict(gbtModel, df)

## ---- warning=FALSE------------------------------------------------------
df <- createDataFrame(longley)
rfModel <- spark.randomForest(df, Employed ~ ., type = "regression", maxDepth = 2, numTrees = 2)
summary(rfModel)
predictions <- predict(rfModel, df)

## ------------------------------------------------------------------------
X1 <- data.frame(V1 = rnorm(4), V2 = rnorm(4))
X2 <- data.frame(V1 = rnorm(6, 3), V2 = rnorm(6, 4))
data <- rbind(X1, X2)
df <- createDataFrame(data)
gmmModel <- spark.gaussianMixture(df, ~ V1 + V2, k = 2)
summary(gmmModel)
gmmFitted <- predict(gmmModel, df)
head(select(gmmFitted, "V1", "V2", "prediction"))

## ------------------------------------------------------------------------
kmeansModel <- spark.kmeans(carsDF, ~ mpg + hp + wt, k = 3)
summary(kmeansModel)
kmeansPredictions <- predict(kmeansModel, carsDF)
head(select(kmeansPredictions, "model", "mpg", "hp", "wt", "prediction"), n = 20L)

## ------------------------------------------------------------------------
corpus <- data.frame(features = c(
  "1 2 6 0 2 3 1 1 0 0 3",
  "1 3 0 1 3 0 0 2 0 0 1",
  "1 4 1 0 0 4 9 0 1 2 0",
  "2 1 0 3 0 0 5 0 2 3 9",
  "3 1 1 9 3 0 2 0 0 1 3",
  "4 2 0 3 4 5 1 1 1 4 0",
  "2 1 0 3 0 0 5 0 2 2 9",
  "1 1 1 9 2 1 2 0 0 1 3",
  "4 4 0 3 4 2 1 3 0 0 0",
  "2 8 2 0 3 0 2 0 2 7 2",
  "1 1 1 9 0 2 2 0 0 3 3",
  "4 1 0 0 4 5 1 3 0 1 0"))
corpusDF <- createDataFrame(corpus)
model <- spark.lda(data = corpusDF, k = 5, optimizer = "em")
summary(model)

## ------------------------------------------------------------------------
posterior <- spark.posterior(model, corpusDF)
head(posterior)

## ------------------------------------------------------------------------
perplexity <- spark.perplexity(model, corpusDF)
perplexity

## ------------------------------------------------------------------------
ratings <- list(list(0, 0, 4.0), list(0, 1, 2.0), list(1, 1, 3.0), list(1, 2, 4.0),
                list(2, 1, 1.0), list(2, 2, 5.0))
df <- createDataFrame(ratings, c("user", "item", "rating"))
model <- spark.als(df, "rating", "user", "item", rank = 10, reg = 0.1, nonnegative = TRUE)

## ------------------------------------------------------------------------
stats <- summary(model)
userFactors <- stats$userFactors
itemFactors <- stats$itemFactors
head(userFactors)
head(itemFactors)

## ------------------------------------------------------------------------
predicted <- predict(model, df)
head(predicted)

## ---- warning=FALSE------------------------------------------------------
df <- createDataFrame(longley)
afStats <- head(select(df, mean(df$Armed_Forces), sd(df$Armed_Forces)))
afMean <- afStats[1]
afStd <- afStats[2]

test <- spark.kstest(df, "Armed_Forces", "norm", c(afMean, afStd))
testSummary <- summary(test)
testSummary

## ---- warning=FALSE------------------------------------------------------
irisDF <- createDataFrame(iris)
gaussianGLM <- spark.glm(irisDF, Sepal_Length ~ Sepal_Width + Species, family = "gaussian")

# Save and then load a fitted MLlib model
modelPath <- tempfile(pattern = "ml", fileext = ".tmp")
write.ml(gaussianGLM, modelPath)
gaussianGLM2 <- read.ml(modelPath)

# Check model summary
summary(gaussianGLM2)

# Check model prediction
gaussianPredictions <- predict(gaussianGLM2, irisDF)
head(gaussianPredictions)

unlink(modelPath)

## ---- echo=FALSE---------------------------------------------------------
sparkR.session.stop()

