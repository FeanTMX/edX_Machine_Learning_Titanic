library(titanic)  
library(caret)
library(tidyverse)
library(rpart)


options(digits = 3)

# clean the data - `titanic_train` is loaded with the titanic package, provided in courseware
titanic_clean <- titanic_train %>%
  mutate(Survived = factor(Survived),
         Embarked = factor(Embarked),
         Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age), # NA age to median age
         FamilySize = SibSp + Parch + 1) %>%    # count family members
  select(Survived,  Sex, Pclass, Age, Fare, SibSp, Parch, FamilySize, Embarked)

#Split data to train and test sets
set.seed(42,sample.kind='Rounding')
ind<-createDataPartition(titanic_clean$Survived,p=0.2,list=FALSE)
test_set<-slice(titanic_clean,ind)
train_set<-slice(titanic_clean,-ind)
filter(train_set,Survived == '1')%>%
  nrow(.)/712

#Method 1:  Guessing
guess<-sample(c(0,1),179,replace=TRUE)
cm_1<-confusionMatrix(as.factor(guess),reference = test_set$Survived)
cm_1$overall['Accuracy']

#Method 2: Guessing by Sex
p_male_survived<-filter(train_set,Sex == 'male')%>%
  .$Survived %in% '1'%>%
  mean()
p_female_survived<-filter(train_set,Sex == 'female')%>%
  .$Survived %in% '1'%>%
  mean()%>%
  print()
guess_2<-ifelse(test_set$Sex=='female',1,0)
cm_2<-confusionMatrix(as.factor(guess_2),reference = test_set$Survived)
cm_2$overall['Accuracy']

#Method 3: Guessing by pClass
grouped<-train_set%>%
  group_by(Pclass)
tab<-summarise(grouped,tot=n())
grouped%>%
  filter(Survived=='1')%>%
  summarize(n=n())%>%
  full_join(tab)%>%
  mutate(p=n/tot)
guess_3<-ifelse(test_set$Pclass=='1',1,0)
cm_3<-confusionMatrix(as.factor(guess_3),reference = test_set$Survived)
cm_3$overall['Accuracy']

#Method 4: Guessing by pClass & Sex
grouped2<-train_set%>%
  group_by(Pclass,Sex)
tab2<-summarise(grouped,tot=n())
grouped2%>%
  filter(Survived=='1')%>%
  summarize(n=n())%>%
  full_join(tab)%>%
  mutate(p=n/tot)
guess_4<- ifelse(test_set$Sex=='female'&test_set$Pclass != 3,1,0)
cm_4<-confusionMatrix(as.factor(guess_4),reference = test_set$Survived)
cm_4$overall['Accuracy']

#Confusion Matrix
cm_2
cm_3
cm_4

#F1 Scores
F_meas(data=as.factor(guess_2),reference = test_set$Survived)
F_meas(data=as.factor(guess_3),reference = test_set$Survived)
F_meas(data=as.factor(guess_4),reference = test_set$Survived)

#Method 5: lda/qda
set.seed(1,sample.kind = 'Rounding')
lda_fit<-train(Survived~Fare,method = 'lda',data=train_set)
lda_res<-predict(lda_fit,test_set)
confusionMatrix(lda_res,reference = test_set$Survived)$overall['Accuracy']

set.seed(1,sample.kind = 'Rounding')
qda_fit<-train(Survived~Fare,method = 'qda',data=train_set)
qda_res<-predict(qda_fit,test_set)
confusionMatrix(qda_res,reference = test_set$Survived)$overall['Accuracy']

#Method 6: linear regression model
set.seed(1,sample.kind = 'Rounding')
glm_fit<-train(Survived~Age,method = 'glm',data=train_set)
glm_res<-predict(glm_fit,test_set)
confusionMatrix(glm_res,reference = test_set$Survived)$overall['Accuracy']

set.seed(1,sample.kind = 'Rounding')
glm_fit2<-train(Survived~Age+Sex+Fare+Pclass,method = 'glm',data=train_set)
glm_res2<-predict(glm_fit2,test_set)
confusionMatrix(glm_res2,reference = test_set$Survived)$overall['Accuracy']

set.seed(1,sample.kind = 'Rounding')
glm_fit3<-train(Survived~.,method = 'glm',data=train_set)
glm_res3<-predict(glm_fit3,test_set)
confusionMatrix(glm_res3,reference = test_set$Survived)$overall['Accuracy']

#Method 7: kNN model
set.seed(6,sample.kind = 'Rounding')
knn_fit<-train(Survived~.,method='knn',tuneGrid=data.frame(k = seq(3, 51, 2)),data=train_set)
knn_res<-predict(knn_fit,test_set)
plot(knn_fit)
confusionMatrix(knn_res,reference = test_set$Survived)$overall['Accuracy']

set.seed(8,sample.kind = 'Rounding')
control<-trainControl(method = 'cv',number=10,p=0.1)
knn_fit2<-train(Survived~.,method='knn',tuneGrid=data.frame(k = seq(3, 51, 2)),trControl=control,data=train_set)
knn_res2<-predict(knn_fit2,test_set)
confusionMatrix(knn_res2,reference = test_set$Survived)$overall['Accuracy']

#Method 8: Classification Tree Model
set.seed(10,sample.kind = 'Rounding')
rpart_fit<-train(Survived~.,data=train_set,method = 'rpart',tuneGrid=data.frame(cp = seq(0, 0.05, 0.002)))
rpart_res<-predict(rpart_fit,test_set)
confusionMatrix(rpart_res,reference = test_set$Survived)$overall['Accuracy']

plot(rpart_fit$finalModel)
text(rpart_fit$finalModel)

#Method 9: Random Tree Model
set.seed(14,sample.kind = 'Rounding')
rf_fit<-train(Survived~.,method = 'rf',data=train_set,tuneGrid=data.frame(mtry=1:7),ntree=100)
rf_res<-predict(rf_fit,test_set)
confusionMatrix(rf_res,reference = test_set$Survived)$overall['Accuracy']
varImp(rf_fit)
