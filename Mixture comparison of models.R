## Load the "Mixture" dataset.
library("ElemStatLearn")
# Extract the components that we want.
x <- mixture.example$x
y <- mixture.example$y
xnew <- mixture.example$xnew
# px1 <- mixture.example$px1
# px2 <- mixture.example$px2
prob <- mixture.example$prob

## Plot training data. First, we need to make a dataframe for ggplot use.
df.train <- data.frame(x1=x[ , 1], x2=x[ ,2], y=y)
# Make y to be a factor.
df.train$y <- as.factor(df.train$y)
summary(df.train)
## Plot the true boundary.Again, we need to make a dataframe for ggplot use.
df.grid <- data.frame(x1=xnew[ ,1], x2=xnew[ ,2])
df.grid$prob <- prob
summary(df.grid)

library("ggplot2")
# To plot 200 training points.
# ggplot() + geom_point(data=df.train, aes(x=x1, y=x2, color=y), size=4) + scale_color_manual(values=c("green", "red")) + theme_bw()
# To plot boundary.
# stat_contour(data=df.grid, aes(x=x1, y=x2, z=prob), breaks=c(0.5))
# Plot 200 training points and boundary.
ggplot() + geom_point(data=df.train, aes(x=x1, y=x2, color=y), size=4) + scale_color_manual(values=c("green", "red")) + theme_bw() + stat_contour(data=df.grid, aes(x=x1, y=x2, z=prob), breaks=c(0.5))

## Generate test data.
set.seed(9876)
# 1st 5000 rows are random numbers betw 1 & 10, last 5000 rows are random numbers betw 11 & 20.
centers <- c(sample(1:10, 5000, replace=TRUE), sample(11:20, 5000, replace=TRUE))
# means consists of 20 different combinations of mixture centers. 
means <- mixture.example$means
# If 3rd and 10th integer of centers is 2, the 3rd and 10th row of means will take the 2nd combination of mixture centers.
means <- means[centers, ]
# Now the 10000 rows of means have 20 unique combinations.
library("mvtnorm")
x.test <- rmvnorm(10000, c(0, 0), 0.2 * diag(2))
x.test <- x.test + means
# replicate zeros and ones 5000 times each
y.test <- c(rep(0, 5000), rep(1, 5000)) 
# Make a dataframe containing the 10000 test values.
df.test <- data.frame(x1=x.test[ ,1], x2=x.test[ ,2], y=y)
df.test$y <- as.factor(df.test$y)
summary(df.test)

## Calculate irreducible error which is the error that comes from the data generating model.
bayes.error <- sum(mixture.example$marginal * (prob * I(prob < 0.5) + (1-prob) * I(prob >= 0.5)))

## K-Nearest Neighbour Classification with specific k value, for example k = 9.
library("FNN")
knn9 <- knn(x, x.test, y, k=9, prob=TRUE)
# Display the structure.
str(knn9)
# Plot the decision boundary of knn9 model.
knn9b <- knn(x, xnew, y, k=9, prob=TRUE)
prob.knn9b <- attr(knn9b, "prob")
prob.knn9b <- ifelse(knn9b == "1", prob.knn9b, 1 - prob.knn9b)
df.grid$prob.knn9b <- prob.knn9b
ggplot() + geom_point(data=df.train, aes(x=x1, y=x2, color=y), size=4) + scale_color_manual(values=c("green", "red")) + theme_bw() + stat_contour(data=df.grid, aes(x=x1, y=x2, z=prob.knn9b), breaks=c(0.5))
# Predict using the knn model.  
prob.knn9 <- attr(knn9, "prob")
prob.knn9 <- ifelse(knn9 == "1", prob.knn9, 1 - prob.knn9)
df.test$prob.knn9 <- prob.knn9
df.test$pred.knn9 <- knn9
# vertical axis shows predictions, horizontal axis shows actual values.
table(df.test$pred.knn9, df.test$y)

## Logistic Regression classification.
# Fit a Logistic Regression model with 7th-order polynomials.
lr7 <- glm(y ~ poly(x1,7) + poly(x2,7), data=df.train, family=binomial())
summary(lr7)
# Plot the decision boundary of lr model.
df.grid$prob.lr7 <- predict(lr7, newdata=df.grid, type="response")
ggplot() + geom_point(data=df.train, aes(x=x1, y=x2, color=y), size=4) + scale_color_manual(values=c("green", "red")) + theme_bw() + stat_contour(data=df.grid, aes(x=x1, y=x2, z=prob.lr7), breaks=c(0.5))
# Predict using the lr model.
df.test$prob.lr7 <- predict(lr7, newdata=df.test, type="response")
df.test$pred.lr7 <- as.factor(ifelse(df.test$prob.lr7 > 0.5, 1, 0))
table(df.test$pred.lr7, df.test$y)
