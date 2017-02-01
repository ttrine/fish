library(pROC)
library(SDMTools)
library(readr)

test <- read_csv("~/Documents/Projects/fish/analysis/threshold/detect_batch_more_feats.csv", 
                 col_names = FALSE)response <- as.integer(detect_batch_more_feats$V1)

# ROC Curve
response <- test$X1
predictor <- test$X2
roc_curve <- roc(response,predictor)
plot(roc_curve)

# Rates
fpr <- function(k){
    fp <- as.integer(predictor[response==0]>k)
    return(sum(fp)/length(fp))
}

fnr <- function(k){
    fn <- as.integer(predictor[response==1]<=k)
    return(sum(fn)/length(fn))
}

tpr <- function(k){
    tp <- as.integer(predictor[response==1]>k)
    return(sum(tp)/length(tp))
}

frs <- function(k){fpr(k)+fnr(k)}

accuracy <- function(k){
	tp <- sum(as.integer(predictor[response==1]>k))
	tn <- sum(as.integer(predictor[response==0]<=k))
	return((tp+tn)/length(predictor))
}