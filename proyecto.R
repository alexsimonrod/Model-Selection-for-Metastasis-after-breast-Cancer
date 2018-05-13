load("project.RData")

library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
library(randomForest)
library(e1071)
library(neuralnet)
library(FNN)
library(ModelMetrics)


############################################################################
#Searching NAs
contador = 0;
max = 77;

for(i in 1:22289){
  for(j in 1:258){
    if(is.na(trainSet[j,i])){
      contador = contador+1
    }
  }
  if(contador > max){
    #trainmod[,i] <- NULL
    print("Cont")
    print(contador)
    print("col")
    print(i)
    print("#####################################################")
  }
  contador = 0
  
}

#############################################################################
#Converting categorical variables into numeric variables(Making Dummies)
trainSet1 = trainSet

#Finding the location of such variables
for(i in 1:22289){
  if(!is.numeric(trainSet[,i]))
    print(i)
  
}


trainSet1[,22288] <- as.numeric(trainSet[,22288])
trainSet1[,22285] <- as.numeric(trainSet[,22285])

#Dummies results

result <- as.numeric(trainLabels)



##############################################################################
#Filling NAs with the mean of the variable.

trainmod = trainSet1
mean_84 = mean(trainSet1[,22284],na.rm = TRUE)
mean_87 = mean(trainSet1[,22287],na.rm = TRUE)
mean_89 = mean(trainSet1[,22289],na.rm = TRUE)


for(i in 1:258){
  if(is.na(trainmod[i,22289])){
    trainmod[i,22289] = mean_89
  }
  if(is.na(trainmod[i,22287])){
    trainmod[i,22287] = mean_87
  }
  if(is.na(trainmod[i,22284])){
    trainmod[i,22284] = mean_84
  }
}

trainmod[,22286] <- NULL
trainmod[,22287] <- NULL


##############################################################################
#Filling NAs with the median of the variable.
trainmod1 <- trainSet1

median_84 = median(trainSet1[,22284],na.rm = TRUE)
median_87 = median(trainSet1[,22287],na.rm = TRUE)
median_89 = median(trainSet1[,22289],na.rm = TRUE)


for(i in 1:258){
  if(is.na(trainmod1[i,22289])){
    trainmod1[i,22289] = median_89
  }
  if(is.na(trainmod1[i,22287])){
    trainmod1[i,22287] = median_87
  }
  if(is.na(trainmod1[i,22284])){
    trainmod1[i,22284] = median_84
  }
}

trainmod1[,22286] <- NULL
trainmod1[,22287] <- NULL
#############################################################################
#Scaling the data

trainmodn1 <- trainmod
trainmodn2 <- trainmod

trainmod11 <- trainmod1
trainmod12 <- trainmod1


#Normalization of the data
#For the one treat with mean
max = apply(trainmodn1[,c(1:22287)],2,max)
min = apply(trainmodn1[,c(1:22287)],2,min)
trainmodn1 = as.data.frame(scale(trainmodn1[,c(1:22287)],center=min,scale = max-min))

#For the one treat with median
max = apply(trainmod11[,c(1:22287)],2,max)
min = apply(trainmod11[,c(1:22287)],2,min)
trainmod11 = as.data.frame(scale(trainmod11[,c(1:22287)],center=min,scale = max-min))


#Apply to the testSet
max = apply(testSet[,c(1:22284)],2,max)
min = apply(testSet[,c(1:22284)],2,min)
testSet1 = as.data.frame(scale(testSet[,c(1:22284)],center=min,scale = max-min))



#Standarization
#For the one treat with mean
for(i in 1:22287){
  trainmodn2[,i] = (trainmodn2[,i]-mean(trainmodn2[,i]))/sqrt(var(trainmodn2[,i]))
}
#For the one treat with median
for(i in 1:22287){
  trainmod12[,i] = (trainmod12[,i]-mean(trainmod12[,i]))/sqrt(var(trainmod12[,i]))
}



######################################################################################
#Multicolineality analysis
vcol <- numeric();
vfil <- numeric();

already <- function(x,vcol){
  for(k in 1:length(vcol)){
    if(vcol[k] == x)
      return(TRUE)
  }
  return(FALSE)
  
}

for(i in 1:22287){
  for(j in i:22287){
    if((det(cor(trainmod11[,c(i,j)])) < 0.5) && (i!=j) && !already(j,vcol)){
      vcol = c(vcol,j)
    }
  }
}

#####################################################################################
#Studying of feature selection applying Random Forest


names(trainmod11)[22285] = "ERStatus"
names(trainmod11)[22286] = "TumorSize"
names(trainmod11)[22287] = "TumorGrade"

fs = rep(0,22287)
n_a = rep(0,22287)


for(k in 1:1000){
  muestra = sample(c(1:22287),1000)
  
  train.forest = trainmod11[,muestra]
  
  #Number of time that each variable is taken
  for(i in 1:length(muestra)){
    n_a[muestra[i]] = n_a[muestra[i]]+1
  }
  
  data <- cbind(trainLabels = trainLabels, train.forest)
  
  fit <- randomForest(trainLabels ~ ., data,ntree=500)
  
  total = sum(importance(fit))
  
  #Taking the variables with more importance.
  for(i in 1:length(importance(fit))){
    if(importance(fit)[i]/total > 0.003)
      fs[muestra[i]] = fs[muestra[i]]+1
  }
  print(k)
}

#Conditions to select variable
var5 = numeric()
cont=0
for(i in 1:length(fs)){
  if(fs[i] > 20 && ((fs[i]/n_a[i]) == 1)){
    var5 = c(var5,i)
  }
}




######################################################################################
#Random Forest 

#Cross-Validation for tuning parameters
#Initialization
R = 1
S = 10
g=numeric()
varian = numeric()
for(s in 1:10){
  acc = numeric()
  for(j in 1:100){
    #Splitting the data
    smp <- sample(1:258,size = 215,replace = F)
    train.forest = trainmod11[smp,var12]
    test.forest = trainmod11[-smp,var12]
      
    data <- cbind(trainLabels = trainLabels[smp], train.forest)
    
    #Using Random Forest
    fit <- randomForest(trainLabels ~ ., data,ntree=500,nodesize = 25,replace = T)
    summary(fit)
    
    #Predict Output 
    predicted_rf= predict(fit,test.forest,type = "response")
    
    #Accuracy

    test.label = trainLabels[-smp]
    acc = c(acc,1-ce(test.label,predicted_rf))
  }
  
  g = c(g,mean(acc))
  varian = c(varian,sd(acc))
  
  R = R+10
  print(s)
}


#Ploting graphics for the report
plot(c(1:10),g,main = "Accuracy and Variance ~ nodesize",xlab = "mtry = 1,10,20,30,40,50,60,70,80,90,100",ylab = "Accuracy and Variance",col = "blue",type = "l",ylim = c(0,1))
lines(c(1:10),varian,col = "red",type = "l")


##############################################################################################33
#GBM

#Cross validation with GBM
#Initialization of variables
R = 0.1
g=numeric()
varian=numeric()

for(j in 1:20){
  acc = numeric()
  vbcr = numeric()

  for(k in 1:20){
    #Splitting the data
    smp <- sample(1:258,size = 188,replace = F)
    train.forest = trainmodn2[smp,var12]
    test.forest = trainmodn2[-smp,var12]
    
    data <- cbind(trainLabels = trainLabels[smp], train.forest)
    
    #Setting the parameters
    fitControl <- trainControl(method = "cv",number = 10)
    
    gbmGrid <-  expand.grid(interaction.depth = 2,n.trees = 500,shrinkage = 1,
                              n.minobsinnode = 15)
    
    #Using GBM algorithm
    fit <- train(trainLabels ~ ., data, method = "gbm",trControl = fitControl,
                 verbose = FALSE,tuneGrid = gbmGrid,bag.fraction = 0.5,distribution = "adaboost")
    
    #Predicted Output
    predicted_gbm= predict(fit,test.forest)
  
    #Accuracy
    test.label = trainLabels[-smp]
    acc = c(acc,1-ce(test.label,predicted_gbm))
    
    #Study of BCR
    for(i in 1:20){
      if(predicted_gbm[i] == test.label[i]){
        if(test.label[i] == "Metastasis"){
          tp = tp +1
        }
        else
          tn = tn+1
      }
      else if(predicted_gbm[i] != test.label[i]){
        if(predicted_gbm[i] == "Metastasis"){
          fn = fn +1
        }
        else{
          fp = fp +1
        }
      }
    }
    bcr = 1/2*(tp/(tp + fn) + tn/(tn+fp))
    vbcr = c(vbcr,bcr)
  }
  
  mean(acc)
  sd(acc)

  mean(vbcr,na.rm = T)
  sd(vbcr,na.rm = T)
  
  g = c(g,mean(acc))
  varian = c(varian,sd(acc))
  R = R+0.1
  print(j)
}


#Shrink
a = g
v = varian


a1 = g
v1 = varian

#Bag_fraction
a2 = g
a2 = varian

#Plotting graphics for the report
plot(c(1:20),g,main = "Accuracy and Variance ~ Learning Rate",ylab = "Accuracy and Variance", xlab = "Depth values: 0.1-2",ylim = c(0,1),
     type = "l",col = "blue")
lines(c(1:20),varian,col = "red")


###########################################################################################
#XGBOOST

#Dummy variables
y = numeric()
for(i in 1:258){
  if(trainLabels[i] == "Metastasis")
    y = c(y,1)
  else
    y = c(y,0)
}

#Cross Validation for tuning parameters
#Initialization of variables
g = numeric()
varian = numeric()
R = 0.01
for(s in 1:10){
  acc = numeric()
  
    for(i in 1:100){
      #Splitting the data
      smp <- sample(1:258,size = 188,replace = F)
      train.xg = trainmod11[smp,var12]
      test.xg = trainmod11[-smp,var12]
        
      data <- cbind(trainLabels = y[smp], train.xg)
      data <- data.matrix(data)
      
      #Using XGBOOST algorithm
      xgb <- xgboost(data = data[,-1], label = data[,1] , eta = 0.01,max_depth = 8, nround=500, 
                     subsample = 0.5,colsample_bytree = 0.8,
                     objective = "binary:logistic",nthread = 3,silent=1)
      
      #Predict values in test set
      y_pred <- predict(xgb, data.matrix(test.xg))
    
      #Accuracy
      cont=0
      test.label = trainLabels[-smp]
      for(i in 1:70){
        if(y_pred[i] > 0.5){
          if(test.label[i] == "Metastasis")
            cont = cont +1
        }
        else
          if(test.label[i] == "No Metastasis")
            cont=cont+1
      }
      acc = c(acc,cont/70)
    
    }
  
  mean(acc)
  sd(acc)
  g = c(g,mean(acc))
  varian = c(varian,sd(acc))
  R = R+0.01
}

#0.1-1 eta
a = g
v = varian
x=c(g,a)
v=c(varian,v)

#Plotting graphics for the report
plot(c(1:20),x,main = "Accuracy ~ Eta",ylab = "Accuracy ", xlab = "Eta values: 0.1-2",
              type = "l",col = "blue")
lines(c(1:20),v,col = "red")
############################################################################################

###########################################################################################
#Feature Selection with gbm

fs_1 = rep(0,22287)
n_a1 = rep(0,22287)


for(k in 1:100){
  muestra = sample(c(1:22287),1000)
  
  train.forest = trainmod11[,muestra]
  
  #Number of time that each variable is taken
  for(i in 1:length(muestra)){
    n_a1[muestra[i]] = n_a1[muestra[i]]+1
  }
  
  train.forest = trainmodn2[,muestra]
  
  
  data <- cbind(trainLabels = trainLabels, train.forest)
  
  fitControl <- trainControl(method = "cv",number = 10)
  
  gbmGrid <-  expand.grid(interaction.depth = 2,n.trees = 500,shrinkage = 0.1,
                          n.minobsinnode = 10)
  
  
  fit <- train(trainLabels ~ ., data, method = "gbm",trControl = fitControl,
               verbose = FALSE,tuneGrid = gbmGrid,bag.fraction = 0.5,distribution = "adaboost")
  
  print("Paso gbm")
  ImpMeasure<-data.frame(varImp(fit)$importance)
  ImpMeasure$Vars<-row.names(ImpMeasure)
  ImpMeasure[order(-ImpMeasure$Overall),][1:100,]
  
  aux = 0
  #Taking the variables with more importance.
  for(i in 1:50){
    gen = ImpMeasure[order(-ImpMeasure$Overall),][1:50,]$Vars[i]
    for(l in 1:dim(ImpMeasure)[1]){
      if(gen == ImpMeasure$Vars[l])
        aux = l
    }
      fs_1[muestra[aux]] = fs_1[muestra[aux]]+1
  }
  print(k)
}

var13 = numeric()
var10 = numeric()
var9 = numeric()
cont=0

#Selecting variables according to their proportion of nº selected/ nº choosen
for(i in 1:length(fs_1)){
  if(n_a1[i] != 0 && (fs_1[i]/n_a1[i]) > 0.8){
    var13 = c(var13,i)
  }
  if(n_a1[i] == 0)
    var10 = c(var10,i)
}

############################################################################################
#K-Nearest Neighbours

#Cross validation with knn
#Initialization of the data
mn = numeric()
for(j in 1:50){
    acc = numeric()
    
    for(k in 1:100){
      #Splitting the data
      smp <- sample(1:258,size = 215,replace = F)
      trainknn = trainmod11[smp,var5]
      testknn = trainmod11[-smp,var5]
      
      
      #Calcule of K-Nearest-Neighbors and the prediction of the model
      pred_knn = knn(train = trainknn, test = testknn, trainLabels[smp], k = j)
      
      #Accuracy
      cont=0
      test.label = trainLabels[-smp]
      for(i in 1:43){
        if(test.label[i] == "Metastasis" && pred_knn[i] == "Metastasis")
          cont = cont +1
        if(test.label[i] == "No Metastasis" && pred_knn[i] == "No Metastasis")
          cont = cont +1
      }
      
      acc = c(acc,cont/43)
  }
  mn = c(mn,mean(acc))

}

#Plotting to tune number of neighbours
plot(c(1:49),mn,type = "l",col = "blue",main = "Analysis of precision ~ nº of neighbours",xlab = "nº of neighbours",ylab = "Accuracy")




#############################################################################################
#SVM
require(e1071) 

g = numeric()
varian = numeric()
f = numeric()
h = numeric()

R = 1
#Cross-Validation for tuning parameters
for(s in 1:10){
  acc = numeric()
  vbcr = numeric()
  
  for(j in 1:100){
    #Splitting data
    smp <- sample(1:258,size = 173,replace = F)
    train.svm = trainmod11[smp,var12]
    test.svm = trainmod11[-smp,var12]
    
    datasvm <- cbind(train.svm,trainLabels = trainLabels[smp])
    
    #Model
    model <- svm(trainLabels ~ .,data=datasvm,kernel='radial',gamma = 0.001,cost=1000,cross = 10)
    
    #Predict Output
    pred_svm <- predict(model,test.svm)
    
    #Accuracy

    test.label = trainLabels[-smp]
    
    acc = c(acc,1-ce(test.label,pred_svm))

    tp = tn = fp = fn = 0
    for(i in 1:85){
      if(pred_svm[i] == test.label[i]){
        if(test.label[i] == "Metastasis"){
          tp = tp +1
        }
        else
          tn = tn+1
      }
      else if(pred_svm[i] != test.label[i]){
        if(pred_svm[i] == "Metastasis"){
          fn = fn +1

        }
        else{
          fp = fp +1

        }
      }
    }
    
    bcr = 1/2*(tp/(tp + fn) + tn/(tn+fp))
    
    vbcr = c(vbcr,bcr)

  }
  mean(acc)
  sd(acc)
  
  mean(vbcr,na.rm = T)
  sd(vbcr,na.rm = T)
  
  d= mean(acc)
  g = c(g,d)
  e =sd(1-acc)
  varian = c(varian,e)
  f = c(f,mean(vbcr,na.rm = T))
  h = c(h,sd(vbcr,na.rm = T))

  R = 10*R
}

cost2 = g
cost_var2 = varian

cost_3 = g
cost_var3 = varian

x = c(1:6)
names(x) = c("0.1","10","100","1000","10000","100000")

plot(x,cost_3,ylim = c(0,1),type = "l",col = "blue",main = "Accuracy and Variance error ~ Cost ",xlab = "Cost values: 0.1 10 100 1000 10000 100000",ylab = "Accuracy and Variance")
lines(c(1:6),cost_var3,col = "red")

x = c(cost_3,cost2)
y = c(cost_var3,cost_var2)

plot(c(1:26),x,ylim = c(0,1),type = "l",col = "blue",main = "Accuracy and Variance error ~ Gamma ",xlab = "Gamma values: 0.5-2",ylab = "Accuracy and Variance")
lines(c(1:26),y,col = "red")

############################################################################################
#Naives Bayes Clasificator
g=numeric()
varian = numeric()
R=0
for(j in 1:5){
  acc = numeric()
  for(k in 1:100){
    smp = sample(1:258,size=208)
    train.bayes = trainmod11[smp,var13]
    test.bayes = trainmod11[-smp,var13]
    
    databayes <- cbind(train.bayes,trainLabels = trainLabels[smp])
    
    model <- naiveBayes(trainLabels ~ ., data = databayes,laplace = 0)
    
    predbayes <- predict(model,test.bayes,type = "class")
    
    #Accuracy
    label = trainLabels[-smp]
    acc = c(acc,1-ce(label,predbayes))
  }
mean(acc)
sd(acc)
g = c(g,mean(acc))
varian = c(varian,sd(acc))
R = R+100
}


c = g
v = varian

x = c(1:5)

plot(x,c,ylim = c(0,1),type = "l",col = "blue",main = "Accuracy and Variance error ~ Laplace ",xlab = "Laplace value: 0 1 2 3 4",ylab = "Accuracy and Variance")
lines(c(1:5),v,col = "red")

############################################################################################
#Essemble modeling using neural network


acc = acc2 = acc3 = pred = numeric()
vbcr = vbcr1 = vbcr2 = vbcr3 = numeric()

for(j in 1:50){
  m_t = NULL
  m_r = NULL
  tp = fp = tn = fn = 0
  tp1 = fp1 = tn1 = fn1 = 0
  tp2 = fp2 = tn2 = fn2 = 0
  tp3 = fp3 = tn3 = fn3 = 0
  
  #Splitting data
  smp_first = sample(1:258,size = 70,replace= F)
  test.ess = trainmod11[smp_first,var12]
  train.valid = trainmod11[-smp_first,var12]
  
  test.label = trainLabels[smp_first]
  label = trainLabels[-smp_first]
  
  ########################################
  smp <- sample(1:188,size = 118,replace = F)
  train.ess = train.valid[smp,]
  valid.ess = train.valid[-smp,]
  valid.label = label[-smp]
  
  
  label_f = trainLabels[-smp_first]
  data.ess <- cbind(train.ess,trainLabels = label[smp])
  
  #Training Models
  
  #SVM
  svm <- svm(trainLabels ~ .,data=data.ess,kernel='radial',gamma = 0.001,cost=10000)
  
  #Bayes
  bayes <- naiveBayes(trainLabels ~ ., data = data.ess)
  
  #GBM
  
  fitControl <- trainControl(method = "cv",number = 10)
  
  gbmGrid <-  expand.grid(interaction.depth = 2,n.trees = 500,shrinkage = 0.7,
                          n.minobsinnode = 15)
  
  
  gbm <- train(trainLabels ~ ., data.ess, method = "gbm",trControl = fitControl,
               verbose = FALSE,tuneGrid = gbmGrid,bag.fraction = 0.5,distribution = "adaboost")
  
  #Generating data to train the neural network
  
  #Predicting models
  
  #Predict Output SVM
  pred_svm <- predict(svm,valid.ess)
 
  #Predict Output GBM
  pred_gbm = predict(gbm,valid.ess)
  
  #Predict Output Bayes
  predbayes <- predict(bayes,valid.ess)
  
  
  
  #Training the neural network
  #t.knn = c(0)
  t.gbm = numeric()
  t.svm = numeric()
  t.bayes = numeric()
  t.label = numeric()
  
  
  for(z in 1:70){
  if("Metastasis" == predbayes[z])
    t.bayes = c(t.bayes,1)
  else
    t.bayes = c(t.bayes,0)
  if("Metastasis" == pred_svm[z])
    t.svm = c(t.svm,1)
  else
    t.svm = c(t.svm,0)
  if("Metastasis" == pred_gbm[z])
    t.gbm = c(t.gbm,1)
  else
    t.gbm = c(t.gbm,0)
  if(valid.label[z] == "Metastasis")
    t.label = c(t.label,1)
  else
    t.label = c(t.label,0)
  
  }
  
 
  m_t = data.frame(t.svm,t.gbm,t.bayes,t.label)
  
  colnames(m_t) = c("svm","gbm","bayes","class")
  
  
  #Calcule the neural network
  NN = neuralnet(class ~ m_t$svm + m_t$gbm + m_t$bayes ,data = data.matrix(m_t), hidden = c(0),rep = 2)
  
  #Predicting models
  
  #Predict Output SVM
  r.svm <- predict(svm,test.ess)
  
  #Predict Output GBM
  r.gbm = predict(gbm,test.ess)
  
  #Predict Output Bayes
  r.bayes <- predict(bayes,test.ess)
  
  
  
  #Training the neural network
  #t.knn = c(0)
  t.gbm = numeric()
  t.svm = numeric()
  t.bayes = numeric()
  t.label = numeric()
  
  
  for(z in 1:50){
    if("Metastasis" == r.bayes[z])
      t.bayes = c(t.bayes,1)
    else
      t.bayes = c(t.bayes,0)
    if("Metastasis" == r.svm[z])
      t.svm = c(t.svm,1)
    else
      t.svm = c(t.svm,0)
    if("Metastasis" == r.gbm[z])
      t.gbm = c(t.gbm,1)
    else
      t.gbm = c(t.gbm,0)
  }
  
  m_r = data.matrix(data.frame(svm = t.svm,gbm = t.gbm,bayes = t.bayes))
  
  #Predict of neural network
  predict_testNN = compute(NN,m_r)
  
  #Accuracy
  cont=0
  
  for(i in 1:25){
    if(predict_testNN$net.result[i] > 0.5){
      if(test.label[i] == "Metastasis"){
        cont = cont +1
        tp = tp+1
      }
      else
        fn = fn+1
    }
    else
      if(test.label[i] == "No Metastasis"){
        cont=cont+1
        tn = tn+1
      }
      else
        fp = fp+1
  }
  
  acc = c(acc,cont/25)
  acc2 = c(acc2,1-ce(test.label,r.svm))
  acc3 = c(acc3,1-ce(test.label,r.gbm))
  acc4 = c(acc4,1-ce(test.label,r.bayes))
  
  
  bcr = 1/2*(tp/(tp + fn) + tn/(tn+fp))
  
  vbcr = c(vbcr,bcr)
  
  
  for(i in 1:70){
    if(test.label[i] == "Metastasis"){
      if(r.svm[i] == "Metastasis")  
        tp1 = tp1 +1
      else
        fn1 = fn1+1
      if(r.gbm[i] == "Metastasis")  
        tp2 = tp2 +1
      else
        fn2 = fn2+1
      if(r.bayes[i] == "Metastasis")  
        tp3 = tp3 +1
      else
        fn3 = fn3+1
    }
    else if(test.label[i] == "No Metastasis"){
      if(r.svm[i] == "No Metastasis")  
        tn1 = tn1 +1
      else
        fp1 = fp1 +1
      if(r.gbm[i] == "No Metastasis")  
        tn2 = tn2 +1
      else
        fp2 = fp2 +1
      if(r.bayes[i] == "No Metastasis")  
        tn3 = tn3 +1
      else
        fp3 = fp3 +1
    }
  }
  
  
  bcr1 = 1/2*(tp1/(tp1 + fn1) + tn1/(tn1+fp1))
  
  vbcr1 = c(vbcr1,bcr1)
  
  bcr2 = 1/2*(tp2/(tp2 + fn2) + tn2/(tn2+fp2))
  
  vbcr2 = c(vbcr2,bcr2)
  
  bcr3 = 1/2*(tp3/(tp3 + fn3) + tn3/(tn3+fp3))
  
  vbcr3 = c(vbcr3,bcr3)
  
  print(j)
  
}

mean(acc)
mean(acc2)
mean(acc3)
mean(acc4)

mean(vbcr)
mean(vbcr1)
mean(vbcr2)
mean(vbcr3)

sd(acc)
sd(acc2)
sd(acc3)
sd(acc4)

sd(vbcr)
sd(vbcr1)
sd(vbcr2)
sd(vbcr3)




plot(c(1:50),vbcr[1:50],main = "BCR ~ Average,SVM,GBM,Bayes",type = "l",col = "blue",ylim = c(0.75,1),
     xlab = "Test number i",ylab = "BCR")
lines(c(1:50),vbcr1[1:50],col = "black")
lines(c(1:50),vbcr2[1:50],col = "red")
lines(c(1:50),vbcr3[1:50],col = "green")



############################################################################################
#Essembling technique using the average


acc = acc2 = acc3 = pred = numeric()
vbcr = vbcr1 = vbcr2 = vbcr3 = numeric()


for(j in 1:100){
  m_t = NULL
  m_r = NULL
  tp = fp = tn = fn = 0
  tp1 = fp1 = tn1 = fn1 = 0
  tp2 = fp2 = tn2 = fn2 = 0
  tp3 = fp3 = tn3 = fn3 = 0
  
  #Splitting data
  smp_first = sample(1:258,size = 188,replace= F)
  test.ess = trainmod11[-smp_first,var12]
  train.ess = trainmod11[smp_first,var12]
  
  test.label = trainLabels[-smp_first]
  label = trainLabels[smp_first]
  
  l = numeric()
  
  for(k in 1:188){
    if(label[k] == "Metastasis")
      l = c(l,1)
    else
      l = c(l,0)
    
  }
  
  data.ess <- cbind(train.ess,trainLabels = label)
  data2.ess = data.matrix(cbind(train.ess,trainLabels = l))
  
  #Training Models
  
  #SVM
  svm <- svm(trainLabels ~ .,data=data.ess,kernel='radial',gamma = 0.001,cost=1000,cv=10)
  
  #Bayes
  bayes <- naiveBayes(trainLabels ~ ., data = data.ess)
  
  #KNN
  #knn = knn(train = data.ess, test = test.ess, trainLabels[smp], k = 4)
  
  #Random Forest
  rf <- randomForest(trainLabels ~ ., data.ess,ntree=500,replace = T,nodesize = 25)
  
  
  
  #GBM
  
  fitControl <- trainControl(method = "cv",number = 10)
  
  gbmGrid <-  expand.grid(interaction.depth = 2,n.trees = 500,shrinkage = 1,
                          n.minobsinnode = 15)
  
  
  gbm <- train(trainLabels ~ ., data.ess, method = "gbm",trControl = fitControl,
               verbose = FALSE,tuneGrid = gbmGrid,bag.fraction = 0.5,distribution = "adaboost")
  
  
  #Predicting models
  
  #Predict Output SVM
  r.svm <- predict(svm,test.ess)
  
  #Predict Output Random Forest
  r.rf= predict(rf,test.ess)
  
 
  
  #Predict Output GBM
  r.gbm = predict(gbm,test.ess)
  
  #Predict Output Bayes
  r.bayes <- predict(bayes,test.ess)
  
  #Training the neural network
  #t.knn = c(0)
  t.gbm = t.svm = t.bayes = t.xgb =  t.rf = t.label = numeric()
  
  
  for(z in 1:70){
    if("Metastasis" == r.bayes[z])
      t.bayes = c(t.bayes,1)
    else
      t.bayes = c(t.bayes,0)
    if("Metastasis" == r.svm[z])
      t.svm = c(t.svm,1)
    else
      t.svm = c(t.svm,0)
    if("Metastasis" == r.rf[z])
      t.rf = c(t.rf,1)
    else
      t.rf = c(t.rf,0)
    if("Metastasis" == r.gbm[z])
      t.gbm = c(t.gbm,1)
    else
      t.gbm = c(t.gbm,0)
    if(r.xgb[z] > 0.5)
      t.xgb = c(t.xgb,1)
    else
      t.xgb = c(t.xgb,0)
    
  }
  
  m_r = data.matrix(data.frame(svm = t.svm,gbm = t.gbm,bayes = t.bayes))

  
  #Accuracy
  for(i in 1:70){
    if(sum(m_r[i,]) > 1.5){
      if(test.label[i] == "Metastasis")
        tp = tp+1
      else
        fp = fp+1
    }
    if(sum(m_r[i,]) < 1.5){
      if(test.label[i] == "No Metastasis")
        tn = tn +1
      else
        fn = fn +1
    }
  }
  acc2 = c(acc2,1-ce(test.label,r.svm))
  acc3 = c(acc3,1-ce(test.label,r.gbm))
  acc4 = c(acc4,1-ce(test.label,r.bayes))
  
  #Media ponderada
  for(i in 1:70){
    if(sum(m_r[i,]) > 1.5){
      if(test.label[i] == "Metastasis")
        tp = tp+1
      else
        fp = fp+1
    }
    if(sum(m_r[i,]) < 1.5){
      if(test.label[i] == "No Metastasis")
        tn = tn +1
      else
        fn = fn +1
    }
  }
  
  for(i in 1:70){
    if(test.label[i] == "Metastasis"){
      if(r.svm[i] == "Metastasis")  
        tp1 = tp1 +1
      else
        fn1 = fn1+1
      if(r.gbm[i] == "Metastasis")  
        tp2 = tp2 +1
      else
        fn2 = fn2+1
      if(r.bayes[i] == "Metastasis")  
        tp3 = tp3 +1
      else
        fn3 = fn3+1
    }
    else if(test.label[i] == "No Metastasis"){
      if(r.svm[i] == "No Metastasis")  
        tn1 = tn1 +1
      else
        fp1 = fp1 +1
      if(r.gbm[i] == "No Metastasis")  
        tn2 = tn2 +1
      else
        fp2 = fp2 +1
      if(r.bayes[i] == "No Metastasis")  
        tn3 = tn3 +1
      else
        fp3 = fp3 +1
    }
  }
  
  bcr = 1/2*(tp/(tp + fn) + tn/(tn+fp))
  
  vbcr = c(vbcr,bcr)
  
  bcr1 = 1/2*(tp1/(tp1 + fn1) + tn1/(tn1+fp1))
  
  vbcr1 = c(vbcr1,bcr1)
  
  bcr2 = 1/2*(tp2/(tp2 + fn2) + tn2/(tn2+fp2))
  
  vbcr2 = c(vbcr2,bcr2)
  
  bcr3 = 1/2*(tp3/(tp3 + fn3) + tn3/(tn3+fp3))
  
  vbcr3 = c(vbcr3,bcr3)
  
  
  print(j)
  
}

#Results

mean(acc2)
mean(acc3)
mean(acc4)

mean(vbcr)
mean(vbcr1)
mean(vbcr2)
mean(vbcr3)

sd(acc2)
sd(acc3)
sd(acc4)

sd(vbcr)
sd(vbcr1)
sd(vbcr2)
sd(vbcr3)




plot(c(1:50),vbcr[1:50],main = "BCR ~ Neural network,SVM,GBM,Bayes",type = "l",col = "blue",ylim = c(0.75,1),
     xlab = "Test number i",ylab = "BCR")
lines(c(1:50),vbcr1[1:50],col = "black")
lines(c(1:50),vbcr2[1:50],col = "red")
lines(c(1:50),vbcr3[1:50],col = "green")



#########################################################################
#BCR = 1/2*(tp/(tp + fn) + tn/(tn+fp))

tp = tn = fp = fn = 0
for(i in 1:58){
  if(predicted[i] == test.label[i]){
    if(test.label[i] == "Metastasis"){
      tp = tp +1
    }
    else
      tn = tn+1
  }
  else if(predicted[i] != test.label[i]){
    if(predicted[i] == "Metastasis"){
      fn = fn +1
    }
    else
      fp = fp +1
  }
}


##########################################################################################
#Final algorithm

  
#Splitting data
smp <- sample(1:258,size = 188,replace = F)
train.svm = trainmod11[smp,var12]
test.svm = testSet1[,var12]

datasvm <- cbind(train.svm,trainLabels = trainLabels[smp])

#Model
model <- svm(trainLabels ~ .,data=datasvm,kernel='radial',gamma = 0.001,cost=1000,cross = 10)

#Predict Output
pred_svm <- predict(model,test.svm)
    



