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
# Fit a Logistic Regression model with 5th-order polynomials.
lr5 <- glm(y ~ poly(x1,5) + poly(x2,5), data=df.train, family=binomial())
summary(lr5)
# Plot the decision boundary of lr model.
df.grid$prob.lr5 <- predict(lr5, newdata=df.grid, type="response")
ggplot() + geom_point(data=df.train, aes(x=x1, y=x2, color=y), size=4) + scale_color_manual(values=c("green", "red")) + theme_bw() + stat_contour(data=df.grid, aes(x=x1, y=x2, z=prob.lr5), breaks=c(0.5))
# Predict using the lr model.
df.test$prob.lr5 <- predict(lr5, newdata=df.test, type="response")
df.test$pred.lr5 <- as.factor(ifelse(df.test$prob.lr5 > 0.5, 1, 0))
table(df.test$pred.lr5, df.test$y)

## Decision Tree.
library("tree")
tree1 <- tree(y ~ ., data=df.train)
# View details of each decision node. Asterisks are the terminal nodes or leaves.
tree1
summary(tree1)
# Plot the tree before cross-validation.
plot(tree1)
text(tree1)
# Use cross-validation to determine optimal tree size.
set.seed(6789)
tree1.cv <- cv.tree(tree1, method="misclass")
tree1.cv
plot(tree1.cv)
optimal <- which.min(tree1.cv$dev)
optimal.size <- tree1.cv$size[optimal]
# Prune the tree using the optimal tree size determined by cross-validation.
tree1.prune <- prune.tree(tree1, best=optimal.size, method="misclass")
# tree1.prune
# summary(tree1.prune)
# plot(tree1.prune)
# text(tree1.prune)
# Plot the partition. We can see the leaves determined by the pruned tree.
# plot(df.train$x1, df.train$x2, col=ifelse(df.train$y==1, 2, 3), pch=20, cex=2)
# partition.tree(tree1.prune, ordvars=c("x1", "x2"), add=TRUE)
# Plot the decision boundary of the pruned tree model.
# df.grid$prob.tree <- predict(tree1.prune, newdata=df.grid, type="vector")[, 2]
# ggplot() + geom_point(data=df.train, aes(x=x1, y=x2, color=y), size=4) + scale_color_manual(values=c("green", "red")) + theme_bw() + stat_contour(data=df.grid, aes(x=x1, y=x2, z=prob.tree), breaks=c(0.5))
# Predict using the pruned tree model.
df.test$prob.tree <- predict(tree1.prune, newdata=df.test, type="vector")[, 2]
df.test$pred.tree <- predict(tree1.prune, newdata=df.test, type="class")
# table(df.test$pred.tree, df.test$y)

## Random Forest
library("randomForest")
set.seed(9876)
rf <- randomForest(y ~ ., data=df.train, mtry=1, ntree=500, xtest=df.test[, 1:2], y.test=df.test[, 3], keep.forest=TRUE)
plot(rf)
# Partial dependence plots
partialPlot(rf, df.train, x.var="x1", which.class="1")
partialPlot(rf, df.train, x.var="x2", which.class="1")
# Plot the decision boundary of the Random Forest model.
# df.grid$prob.rf <- predict(rf, newdata=df.grid, type="prob")[, 2]
# ggplot() + geom_point(data=df.train, aes(x=x1, y=x2, color=y), size=4) + scale_color_manual(values=c("green", "red")) + theme_bw() + stat_contour(data=df.grid, aes(x=x1, y=x2, z=prob.rf), breaks=c(0.5))
# Predict using the random forest model.
df.test$prob.rf <- rf$test$votes[ ,2]
df.test$pred.rf <- rf$test$predicted
# table(df.test$pred.rf, df.test$y)

## Bagging (this is randomForest when mtry is equal to the number of variables in x)
set.seed(9876)
bag <- randomForest(y ~ ., data=df.train, mtry=2, ntree=500, xtest=df.test[, 1:2], y.test=df.test[, 3], keep.forest=TRUE)
# plot(bag)
# Plot the decision boundary of the bagging model.
# df.grid$prob.bag <- predict(bag, newdata=df.grid, type="prob")[, 2]
# ggplot() + geom_point(data=df.train, aes(x=x1, y=x2, color=y), size=4) + scale_color_manual(values=c("green", "red")) + theme_bw() + stat_contour(data=df.grid, aes(x=x1, y=x2, z=prob.bag), breaks=c(0.5))
# Predict using the bagging model.
df.test$prob.bag <- bag$test$votes[ ,2]
df.test$pred.bag <- bag$test$predicted
# table(df.test$pred.bag, df.test$y)

## Gradient Boosting Machine
library("gbm")
y.train <- as.numeric(df.train$y) - 1
set.seed(9876)
boost <- gbm(y.train ~ x1 + x2, data=df.train, distribution="bernoulli", n.trees=5000, shrinkage=0.001, interaction.depth=4)
gbm.perf(boost)
# Partial effects of each variable
plot(boost, i=1, type="response")
plot(boost, i=2, type="response")
# Plot the decision boundary of the GBM.
# df.grid$prob.gbm <- predict(boost, newdata=df.grid, n.trees=5000, type="response")
# ggplot() + geom_point(data=df.train, aes(x=x1, y=x2, color=y), size=4) + scale_color_manual(values=c("green", "red")) + theme_bw() + stat_contour(data=df.grid, aes(x=x1, y=x2, z=prob.gbm), breaks=c(0.5))
df.test$prob.gbm <- predict(boost, newdata=df.test, n.trees=5000, type="response")
df.test$pred.gbm <- as.factor(ifelse(df.test$prob.gbm > 0.5, 1, 0))
# table(df.test$pred.gbm, df.test$y)

# Compare error rates, ROC and AUC values.
library('ROCR')
# Create prediction objects.
knn.pred <- prediction(df.test$prob.knn9, df.test$y)
lr.pred <- prediction(df.test$prob.lr5, df.test$y)
tree.pred <- prediction(df.test$prob.tree, df.test$y)
rf.pred <- prediction(df.test$prob.rf, df.test$y)
bag.pred <- prediction(df.test$prob.bag, df.test$y)
gbm.pred <- prediction(df.test$prob.gbm, df.test$y)
# Create performance objects for the misclassification errors.
knn.error <- performance(knn.pred, measure="err")
lr.error <- performance(lr.pred, measure="err")
tree.error <- performance(tree.pred, measure="err")
rf.error <- performance(rf.pred, measure="err")
bag.error <- performance(bag.pred, measure="err")
gbm.error <- performance(gbm.pred, measure="err")
# Plot misclassification errors.
plot(knn.error, ylim=c(0.2,0.5), col="red")
plot(lr.error, add=TRUE, col="yellow")
plot(tree.error, add=TRUE, col="green")
plot(rf.error, add=TRUE, col="purple")
plot(bag.error, add=TRUE, col="blue")
plot(gbm.error, add=TRUE, col="grey")
abline(h=bayes.error, lty=2)
# Create performance objects for the ROCs.
knn.ROC <- performance(knn.pred, measure="tpr", x.measure="fpr")
lr.ROC <- performance(lr.pred, measure="tpr", x.measure="fpr")
tree.ROC <- performance(tree.pred, measure="tpr", x.measure="fpr")
rf.ROC <- performance(rf.pred, measure="tpr", x.measure="fpr")
bag.ROC <- performance(bag.pred, measure="tpr", x.measure="fpr")
gbm.ROC <- performance(gbm.pred, measure="tpr", x.measure="fpr")
# Plot ROC curves.
plot(knn.ROC, col="red")
plot(lr.ROC, add=TRUE, col="yellow")
plot(tree.ROC, add=TRUE, col="green")
plot(rf.ROC, add=TRUE, col="purple", lwd=2)
plot(bag.ROC, add=TRUE, col="blue")
plot(gbm.ROC, add=TRUE, col="grey")
abline(a=0, b=1, lty=2)
# AUC values
as.numeric(performance(knn.pred, "auc")@y.values)
as.numeric(performance(lr.pred, "auc")@y.values)
as.numeric(performance(tree.pred, "auc")@y.values)
as.numeric(performance(rf.pred, "auc")@y.values)
as.numeric(performance(bag.pred, "auc")@y.values)
as.numeric(performance(gbm.pred, "auc")@y.values)
