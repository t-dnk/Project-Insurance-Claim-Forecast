library(moments)
#install.packages('rsample')
library(rsample)
insurance_data <- read.csv("C:\\Users\\treic\\OneDrive\\Documents\\Projet Neural Network\\insurance_data.csv")
insurance_data



summary(insurance_data)
'The mean is 13 253 USD and the median is 9370 USD. This gap can be explained by the fact that most the claims
are below 10k USD but there are some extreme cases where the claim exceed 60k USD.'

Claims <- insurance_data$claim
Smoker<- insurance_data$smoker
diabetic <- insurance_data$diabetic
Age <- insurance_data$age
Bmi <- insurance_data$bmi
Blood <- insurance_data$bloodpressure
Region <- insurance_data$region
Gender <- insurance_data$gender

hist(Claims, main= "Claims distribution", xlab= 'Claims') 
'We can see that we have an asymetrical distribution'
skewness(Claims)
' skewness = 1.5'

quantile(Claims, probs = 0.25)
'25% of Claims are below 4719 USD'
quantile(Claims, probs = 0.5)
'50% of the claims are below 9 365 USD'
quantile(Claims, probs = 0.9)
quantile(Claims, probs = 0.95)
'95% of the claims are below 41 125 USD'
sum(is.na(Claims))
'No missing values'

amount_ext<-sum(Claims[Claims>41125.38])
'The total amount of the 5% higher claims is 3 121 214 USD'
tot_amount <- sum(Claims)
tot_amount 
' Total amount of the claims is equal to 17 758 679 USD'
amount_ext_prop = amount_ext/tot_amount
amount_ext_prop

'The proportion of the amount of extreme claims is 18%'


plot(Claims, insurance_data$age)
cor(Claims, Age, method = "pearson")
cor(Claims, Age, method = "spearman")

plot(Claims, Bmi)
cor(Claims, Bmi, method = "pearson")
cor(Claims, Bmi, method = "spearman")

plot(Claims, Blood)
cor(Claims, Blood, method = "pearson")
cor(Claims, Blood, method = "spearman")

aggregate(Claims ~ smoker, data = insurance_data, FUN=mean)
aggregate(Claims ~ smoker, data = insurance_data, FUN=median)
aggregate(Claims ~ diabetic, data = insurance_data, FUN=mean)
aggregate(Claims ~ diabetic, data = insurance_data, FUN=median)

boxplot(Claims ~smoker, data= insurance_data)
boxplot(Claims ~ diabetic, data= insurance_data)

plot(Bmi, Claims, col = ifelse(insurance_data$smoker == "Yes", "red","green"))
by(insurance_data, insurance_data$smoker, function(x){
  cor(x$bmi, x$claim, method="pearson", use = "complete.obs")
})
by(insurance_data, insurance_data$smoker, function(x){
  cor(x$bmi, x$claim, method="spearman", use = "complete.obs")
})
'The link between BMI and claims can vary with the smoker status'

#Preparation of the data for the GLM'

set.seed(123)

split_data <- initial_validation_split(insurance_data, prop = c(0.7,0.15))

train_data <- training(split_data)
test_data <- testing(split_data)
val_data <- validation(split_data)

dim(train_data)
dim(test_data)
dim(val_data)

colSums(is.na(train_data))
colSums(is.na(test_data))
colSums(is.na(val_data))

age_median<- median(train_data$age, na.rm = TRUE)
train_data[is.na(train_data)] <- age_median
train_data


test_data[is.na(test_data)] <- age_median
test_data


val_data[is.na(val_data)] <- age_median
val_data


table(Region) 
table(Gender)

#Encoding
train_data$Smoker_encode <-ifelse(train_data$smoker == 'Yes',1,0)
train_data$diab_encode <- ifelse(train_data$diabetic == 'Yes', 1,0)
train_data$gend_encode <- ifelse(train_data$gender == 'Female', 1, 0)

test_data$Smoker_encode <-ifelse(test_data$smoker == 'Yes',1,0)
test_data$diab_encode <- ifelse(test_data$diabetic == 'Yes', 1,0)
test_data$gend_encode <- ifelse(test_data$gender == 'Female', 1, 0)


val_data$Smoker_encode <-ifelse(val_data$smoker == 'Yes',1,0)
val_data$diab_encode <- ifelse(val_data$diabetic == 'Yes', 1,0)
val_data$gend_encode <- ifelse(val_data$gender == 'Female', 1, 0)

reg <-model.matrix(~ train_data$region -1)
reg_test <-model.matrix(~ test_data$region -1)
reg_val <-model.matrix(~ val_data$region -1)

reg <- reg[,colnames(reg) != "train_data$region", drop = FALSE]
reg_test <- reg_test[,colnames(reg_test) != "test_data$region", drop = FALSE]
reg_val<- reg_val[,colnames(reg_val) != "val_data$region", drop = FALSE]
reg_val
#Standardization

age_mean <- mean(train_data$age)
age_sd <- sd(train_data$age)
train_data$age_scaled <- as.numeric(scale(train_data$age))


val_data$age_scaled <- as.numeric(scale(val_data$age, center = age_mean, scale = age_sd))
test_data$age_scaled <- as.numeric(scale(test_data$age, center = age_mean, scale = age_sd))

bmi_mean <- mean(train_data$bmi)
bmi_sd <- sd(train_data$bmi)
train_data$bmi_scaled <- as.numeric(scale(train_data$bmi))


val_data$bmi_scaled <- as.numeric(scale(val_data$bmi, center = bmi_mean, scale = bmi_sd))
test_data$bmi_scaled <- as.numeric(scale(test_data$bmi, center = bmi_mean, scale = bmi_sd))

blood_mean <- mean(train_data$bloodpressure)
blood_sd <- sd(train_data$bloodpressure)
train_data$blood_scaled <- as.numeric(scale(train_data$bloodpressure))


val_data$blood_scaled <- as.numeric(scale(val_data$bloodpressure, center = blood_mean, scale = blood_sd))
test_data$blood_scaled <- as.numeric(scale(test_data$bloodpressure, center = blood_mean, scale = blood_sd))

child_mean <- mean(train_data$children)
child_sd <- sd(train_data$children)
train_data$child_scaled <- as.numeric(scale(train_data$children))


val_data$child_scaled <- as.numeric(scale(val_data$children, center = child_mean, scale = child_sd))
test_data$child_scaled <- as.numeric(scale(test_data$children, center = child_mean, scale = child_sd))

names(train_data)
var_scaled <- data.frame(train_data$age_scaled, train_data$bmi_scaled, train_data$blood_scaled, train_data$child_scaled)
mat_scaled <- as.matrix(var_scaled)

X_train<- cbind(mat_scaled, reg, train_data$Smoker_encode, train_data$diab_encode, train_data$gend_encode)
X_train
dim(X_train)

colnames(X_train)

colnames(X_train)[9:11] <- c("smoker","diabetic","gender")
anyNA(X_train)
is.numeric(X_train)
var_test_scaled <- data.frame(test_data$age_scaled, test_data$bmi_scaled, test_data$blood_scaled, test_data$child_scaled)
mat_test_scaled <- as.matrix(var_test_scaled)
X_test<- cbind(mat_test_scaled, reg_test, test_data$Smoker_encode, test_data$diab_encode, test_data$gend_encode)
X_test
dim(X_test)

var_val_scaled <- data.frame(val_data$age_scaled, val_data$bmi_scaled, val_data$blood_scaled, val_data$child_scaled)
mat_val_scaled <- as.matrix(var_val_scaled)
X_val<- cbind(mat_val_scaled, reg_val, val_data$Smoker_encode, val_data$diab_encode, val_data$gend_encode)
X_val
dim(X_val)

colnames(X_test)[9:11] <- c("smoker","diabetic","gender")
colnames(X_val)[9:11] <- c("smoker","diabetic","gender")
colnames(train_data)

# GLM
?glm
glm_model <- glm(claim ~age + gender + bmi + bloodpressure + diabetic + smoker + region, family = Gamma(link="log"), data = train_data)

summary(glm_model)

pred_val <- predict(object = glm_model, newdata = val_data, type = "response")
length(pred_val)
summary(pred_val)

exp(1.1450762)

mae <- mean(abs(val_data$claim - pred_val))
mae
rmse <- sqrt(mean((val_data$claim - pred_val)^2))
rmse

# Is the GLM performant enough ?
# Constant prediction
baseline_mean <- mean(train_data$claim)
baseline_mean
pred_baseline <- rep(baseline_mean, time = nrow(val_data))

mae_baseline <- mean(abs(val_data$claim - pred_baseline))
rmse_baseline <- sqrt(mean((val_data$claim - pred_baseline)^2))
mae_baseline
rmse_baseline

# Multiple Linear Regression

lm_model <- lm(claim ~age + gender + bmi + bloodpressure + diabetic + smoker + region, data = train_data)
summary(lm_model)
pred_val_lm <- stats::predict.lm(lm_model, newdata = val_data )

min(pred_val_lm)
mae_lm<- mean(abs(val_data$claim - pred_val_lm))
rmse_lm <- sqrt(mean((val_data$claim - pred_val_lm)^2))

mae_lm
rmse_lm

#Neural Network
#install.packages("keras3")

library(reticulate)

library(keras3)

nn_model_final <- keras3::keras_model_sequential(
  input_shape = c(11)
)

summary(nn_model_final)
nn_model_final <- nn_model_final|>layer_dense(units = 8, activation = "relu")

nn_model_final <- nn_model_final |>layer_dense(units =1 , activation = "linear")

summary(nn_model_final)

nn_model_final <-nn_model_final|>compile(optimizer = "adam", loss = 'mse', metrics = 'mae')
summary(nn_model_final)

y_train <- train_data$claim
length(y_train)
y_val <- val_data$claim
length(y_val)
?fit.keras.src.models.model.Model
history<-nn_model_final |> fit(X_train, y_train, batch_size = 32, epochs = 100, validation_data= list(X_val, y_val) )


nn_model_redim <- keras3::keras_model_sequential(
  input_shape = c(11)
)

summary(nn_model_final)
nn_model_redim <- nn_model_redim|>layer_dense(units = 8, activation = "relu")

nn_model_redim <- nn_model_redim |>layer_dense(units =1 , activation = "linear")

summary(nn_model_redim)

nn_model_redim <-nn_model_redim|>compile(optimizer = "adam", loss = 'mse', metrics = 'mae')
summary(nn_model_redim)
X_train_redim <- X_train/1000
X_val_redim <- X_val/1000
y_train_redim <- train_data$claim / 1000
y_train_redim
y_val_redim <- val_data$claim/1000
history2<-nn_model_redim |> fit(X_train_redim, y_train_redim, batch_size = 32, epochs = 100, validation_data= list(X_val_redim, y_val_redim) )

names(history$metrics)
which.min(history$metrics$val_mae)


which.min(history2$metrics$val_mae)
min(history2$metrics$val_mae[47])
early_stop <- callback_early_stopping(monitor = "val_mae", patience = 10, restore_best_weights = TRUE)


y_train_nn <- as.numeric(train_data$claim) / 1000
y_val_nn <- as.numeric(val_data$claim) / 1000

nn_model_correct <- keras_model_sequential(
  input_shape = c(11)
)

nn_model_correct <- nn_model_correct |>
  layer_dense(units = 16, activation = "relu") |>
  layer_dense(units = 1, activation = "linear")


nn_model_correct |> compile(optimizer = "adam",loss = "mse", metrics = "mae")

history_correct <- nn_model_correct |> fit(x = X_train,y = y_train_nn, batch_size = 32, epochs = 150, validation_data = list(X_val, y_val_nn),callbacks = list(early_stop),verbose = 1)
pred_correct <- as.numeric( predict(nn_model_correct, X_val)) * 1000

mae_nn <- mean(abs(val_data$claim - pred_correct))
rmse_nn <- sqrt(mean((val_data$claim - pred_correct)^2))
mae_nn
rmse_nn

tapply(val_data$claim, val_data$smoker, mean)

tapply(pred_correct, val_data$smoker, mean)

#Applying all models on the test data

pred_nn_test<- as.numeric(predict(nn_model_correct, X_test))*1000
length(pred_nn_test)
summary(pred_nn_test)

mae_nn_test <- mean(abs(test_data$claim - pred_nn_test))
rmse_nn_test <- sqrt(mean((test_data$claim - pred_nn_test)^2))

mae_nn_test
rmse_nn_test

pred_glm_test <- predict(glm_model, newdata = test_data, type = "response")
mae_glm_test <- mean(abs(test_data$claim - pred_glm_test))
rmse_glm_test <- sqrt(mean((test_data$claim - pred_glm_test)^2))

mae_glm_test
rmse_glm_test

pred_lm_test <- predict(lm_model, newdata = test_data)
mae_lm_test <- mean(abs(test_data$claim - pred_lm_test))
rmse_lm_test <- sqrt(mean((test_data$claim - pred_lm_test)^2))

mae_lm_test
rmse_lm_test

resultats_test <- data.frame(Model = c("GLM Gamma", "Linear Regression", "Neural Network"), MAE = c(mae_glm_test, mae_lm_test, mae_nn_test),RMSE = c(rmse_glm_test, rmse_lm_test, rmse_nn_test))
resultats_test