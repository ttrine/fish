library(pROC)
library(SDMTools)
library(readr)

adam_64 <- read_csv("~/Documents/Projects/fish/analysis/threshold/detect_batch_adam_64.csv", 
                 col_names = FALSE)

adam_128 <- read_csv("~/Documents/Projects/fish/analysis/threshold/detect_batch_adam_128.csv", 
                    col_names = FALSE)

sgd_64 <- read_csv("~/Documents/Projects/fish/analysis/threshold/detect_batch_sgd_64.csv", 
                    col_names = FALSE)

more_feats <- read_csv("~/Documents/Projects/fish/analysis/threshold/detect_batch_more_feats.csv", 
                   col_names = FALSE)

# ROC Curve
roc_curve <- function(thresh_data){
  response <- thresh_data$X1
  predictor <- thresh_data$X2 
  roc_curve <- roc(response,predictor)
  plot(roc_curve)
}

roc_curve(adam_64)
roc_curve(adam_128)
roc_curve(sgd_64)

# Rates
fpr <- function(thresh_data,k){
    response <- thresh_data$X1
    predictor <- thresh_data$X2
    fp <- as.integer(predictor[response==0]>k)
    return(sum(fp)/length(fp))
}

fnr <- function(thresh_data,k){
    response <- thresh_data$X1
    predictor <- thresh_data$X2
    fn <- as.integer(predictor[response==1]<=k)
    return(sum(fn)/length(fn))
}

tpr <- function(thresh_data,k){
    response <- thresh_data$X1
    predictor <- thresh_data$X2
    tp <- as.integer(predictor[response==1]>k)
    return(sum(tp)/length(tp))
}

frs <- function(k){fpr(k)+fnr(k)}

accuracy <- function(k){
	tp <- sum(as.integer(predictor[response==1]>k))
	tn <- sum(as.integer(predictor[response==0]<=k))
	return((tp+tn)/length(predictor))
}