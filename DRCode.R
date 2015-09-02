# загружаем данные
library(readr)
data_train <- read_csv("train.csv")
data_test <- read_csv("test.csv")
data_train$label <- as.factor(data_train$label)
require(magrittr)
require(dplyr)
require(caret)

# Рисуем цифры
colors<-c('white','black')
cus_col<-colorRampPalette(colors=colors)

default_par <- par()
par(mfrow=c(6,6),pty='s',mar=c(1,1,1,1),xaxt='n',yaxt='n')
for(i in 1:36)
{
        z<-array(train[i,-1],dim=c(28,28))
        z<-z[,28:1] ##right side up
        image(1:28,1:28,z,main=train[i,1],col=cus_col(256))
        print(i)
}
par(default_par)

# разделяем на выборки
set.seed(111)
split <- createDataPartition(data_train$label, p = 0.6, list = FALSE)
train <- slice(data_train, split)
test <- slice(data_train, -split)

library(rpart)
# удаляем признаки с нулевой вариацией и сравниваем модели
zero_var_col <- nearZeroVar(train, saveMetrics = T, freqCut = 99/1)
train_nzv <- train[, !zero_var_col$nzv]
test_nzv <- test[, !zero_var_col$nzv]
model_tree2 <- rpart(label ~ ., data = train, method="class" )
predict_data_test2 <- predict(model_tree2, newdata = test, type = "class")
model_tree <- rpart(label ~ ., data = train_nzv, method="class" )
predict_data_test <- predict(model_tree, newdata = test_nzv, type = "class")
sum(test_nzv$label != predict_data_test)/nrow(test_nzv)

train <- train[, !zero_var_col$nzv]
test <- test[, !zero_var_col$nzv]



# рисуем learning curve
learn_curve_data <- data.frame(integer(),
                               double(),
                               double())
for (n in 1:5 )
{
        for (i in seq(1, 2000, by = 200))
        {
                train_learn <- train[sample(nrow(train), size = i),]
                test_learn <- test[sample(nrow(test), size = i),]
                model_tree_learn <- rpart(label ~ ., data = train_learn, method="class" )
                predict_train_learn <- predict(model_tree_learn, type = "class")
                error_rate_train_rpart <- sum(train_learn$label != predict_train_learn)/i
                predict_test_learn <- predict(model_tree_learn, newdata = test_learn, type = "class")
                error_rate_test_rpart <- sum(test_learn$label != predict_test_learn)/i
                learn_curve_data <- rbind(learn_curve_data, c(i, error_rate_train_rpart, error_rate_test_rpart))
        }
}


# рисуем кривую
colnames(learn_curve_data) <- c("Size", "Train_Error_Rate", "Test_Error_Rate")
learn_curve_data_long <- melt(learn_curve_data, id = "Size")
ggplot(data=learn_curve_data_long, aes(x=Size, y=value, colour=variable)) + geom_point() + stat_smooth(method = "gam", formula = y ~ s(x), size = 1)
ggplot(data=learn_curve_data_long, aes(x=Size, y=value, colour=variable)) + geom_line()
ggplot(data = learn_curve_data, aes(Size)) + geom_line(aes(y = Train_Error_Rate)) + geom_line(aes(y = Test_Error_Rate))

# 0.63700 Простое дерево загрузка данных на сайт  удалить zero-var
model_tree <- rpart(label ~ ., data = train, method="class" )
predict_data_test <- predict(model_tree, newdata = data_test, type = "class")
predictions <- data.frame(ImageId=1:nrow(data_test), Label=levels(predict_data_test)[predict_data_test])
write_csv(predictions, "submission.csv")


learn_curve_data <- data.frame(integer(),
                               double(),
                               double())

for (n in 1:5 )
{
        for (i in seq(100, 3100, by = 500))
        {
                train_learn <- train[sample(nrow(train), size = i),]
                test_learn <- test[sample(nrow(test), size = i),]
                model_learn <- randomForest(label ~ ., data = train_learn)
                predict_train_learn <- predict(model_learn)
                error_rate_train <- sum(train_learn$label != predict_train_learn)/i
                predict_test_learn <- predict(model_learn, newdata = test_learn)
                error_rate_test <- sum(test_learn$label != predict_test_learn)/i
                learn_curve_data <- rbind(learn_curve_data, c(i, error_rate_train, error_rate_test))
        }
}
rf_model <- train(label ~ ., data = train, method = "rf")

f <- as.formula(paste('lable ~', paste(names(train_nn)[!n %in% 'y'], collapse = ' + ')))


set.seed(100)
train_1000 <- data_train[sample(nrow(data_train), size = 1000),]
colors<-c('white','black')
cus_col<-colorRampPalette(colors=colors)

default_par <- par()

number_row <- 28
number_col <- 28
par(mfrow=c(6,6),pty='s',mar=c(1,1,1,1),xaxt='n',yaxt='n')
for(i in 1:36)
{
        z<-array(as.matrix(train_1000)[i,-1],dim=c(number_row,number_col))
        z<-z[,number_col:1] 
        image(1:number_row,1:number_col,z,main=train_1000[i,1],col=cus_col(256))
}
par(default_par)

zero_var_col <- nearZeroVar(train_1000, saveMetrics = T)
train_1000_cut <- train_1000[, !zero_var_col$nzv]

test <- data.frame(matrix(NA, nrow = 1000, ncol = ncol(train_1000)))
zero_col_number <- 1
for (i in 1:ncol(train_1000)) {
        if (zero_var_col$nzv[i] == F) {
                test[, i] <- restr[, zero_col_number]
                zero_col_number <- zero_col_number + 1
        }
        else test[, i] <- train_1000[, i]
}

pc <- princomp(train_1000[, -1], cor=TRUE, scores=TRUE)
pca <- prcomp(train_1000_cut[, -1], center = TRUE, scale = TRUE)

restr <- pca$x[,1:70] %*% t(pca$rotation[,1:70])

if(pca$scale != FALSE){
        restr <- scale(restr, center = FALSE , scale=1/pca$scale)
}
if(all(pca$center != FALSE)){
        restr <- scale(restr, center = -1 * pca$center, scale=FALSE)
}

par(mfrow=c(6,6),pty='s',mar=c(1,1,1,1),xaxt='n',yaxt='n')
for(i in 1:36)
{
        z<-array(as.matrix(test)[i,-1],dim=c(number_row,number_col))
        z<-z[,number_col:1] 
        image(1:number_row,1:number_col,z,main=test[i,1],col=cus_col(256))
}
par(default_par)
