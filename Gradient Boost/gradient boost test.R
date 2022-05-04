library(gbm)
library(caret)
model_gbm = gbm(label ~.,
                data = train,
                distribution = "multinomial",
                cv.folds = 10,
                shrinkage = .01,
                n.minobsinnode = 10,
                n.trees = 500)  
#use model to make predictions on test data
pred_test = predict.gbm(object = model_gbm,
                        newdata = test,
                        n.trees = 500,           # 500 tress to be built
                        type = "response")

class_names = colnames(pred_test)[apply(pred_test, 1, which.max)]

conf_mat = confusionMatrix(factor(test$label), factor(class_names))
print(conf_mat)