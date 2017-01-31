counts <- read_csv("~/Documents/Projects/fish/analysis/counts/counts.csv", 
                    col_names = FALSE)

plot(table(counts))

# The below shows that 24.5% of images that contain
## fish contain more than one.
num_imgs <- length(counts$X1)
num_imgs_multiple_fish <- length(counts$X1[counts>1])
num_imgs_multiple_fish / num_imgs

# The below shows that 44.9% of fish occur in
## images with multiple fish.
num_fish <- sum(counts$X1)
num_fish_multiple_imgs <- sum(counts$X1[counts > 1])
num_fish_multiple_imgs / num_fish
