overlap_pcts <- read_csv("~/Documents/Projects/fish/analysis/misc/train_overlap_percentages.csv", 
                         col_names = FALSE)

coverage <- as.matrix(overlap_pcts[overlap_pcts>0,])
hist(coverage)
