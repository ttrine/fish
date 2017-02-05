library(readr)

sgd_64 <- read_csv("~/Documents/Projects/fish/analysis/training/detect_batch_sgd_64.csv", 
                    col_names = FALSE)

adam_64 <- read_csv("~/Documents/Projects/fish/analysis/training/detect_batch_adam_64.csv", 
                   col_names = FALSE)

adam_128 <- read_csv("~/Documents/Projects/fish/analysis/training/detect_batch_adam_128.csv", 
                   col_names = FALSE)

plot(as.matrix(adam_128))
plot(as.matrix(adam_64))
plot(as.matrix(sgd_64))

# Find epoch with lowest observed val_loss
get_best <- function(losses){
  which(losses==min(losses)) - 1
}
get_best(sgd_64)
get_best(adam_64)
get_best(adam_128)
