rm(list = ls())                                            

library(pacman)
p_load(igraph, gtools, ggplot2, ggrepel, tnet, data.table, dplyr, gtools, ggplot2, igraph)


p_load(stringr, dplyr, quanteda, LSAfun, igraph, data.table, FactoMineR, factoextra, gtools, ggplot2, GGally, ggrepel, psych, reshape, textstem, koRpus, tnet, data.table)

###################################Get interaction network#####################
setwd("/Users/Oscar/Dropbox/MAPSS/Winter/Computational Contant Analysis/Final")
#Read in scene by character matrix:
amc_character_scenes <- read.csv("amc_network.csv", row.names = 1) %>% as.matrix()
gh_character_scenes <- read.csv("gh_network.csv", row.names = 1) %>% as.matrix()
dool_character_scenes <- read.csv("days_network.csv", row.names = 1) %>% as.data.frame()

#Spelling fixes
dool_character_scenes$EJ <- dool_character_scenes$EJ + dool_character_scenes$E.J.
dool_character_scenes$E.J. <- NULL
dool_character_scenes$Shawn_D <- dool_character_scenes$Shawn.D. + dool_character_scenes$Shawn.D
dool_character_scenes$Shawn.D. <- NULL
dool_character_scenes$Shawn.D <- NULL
names(dool_character_scenes)[names(dool_character_scenes) == 'Shawn_D'] <- 'Shawn-D'
dool_character_scenes <- as.matrix(dool_character_scenes)

#Make coocurrence of characters in scenes matrix
amc_character_cooc <- t(amc_character_scenes) %*% amc_character_scenes
gh_character_cooc <- t(gh_character_scenes) %*% gh_character_scenes
dool_character_cooc <- t(dool_character_scenes) %*% dool_character_scenes


#Create igraph objects
amc_net <- as.tnet(amc_character_cooc, type = "weighted one-mode tnet") %>% as.data.frame() %>% tnet_igraph()
gh_net <- as.tnet(gh_character_cooc, type = "weighted one-mode tnet") %>% as.data.frame() %>% tnet_igraph()
dool_net <- as.tnet(dool_character_cooc, type = "weighted one-mode tnet") %>% as.data.frame() %>% tnet_igraph()

#Names:
amc_names <- row.names(amc_character_cooc) %>% as.data.frame()
amc_names$i <- row_number(amc_names)

gh_names <- row.names(gh_character_cooc) %>% as.data.frame()
gh_names$i <- row_number(gh_names)

dool_names <- row.names(dool_character_cooc) %>% as.data.frame()
dool_names$i <- row_number(dool_names)

#Labsizes:
amc_labs <- colnames(amc_character_cooc) %>% as.data.frame()
amc_labs$s <- diag(as.matrix(amc_character_cooc)) %>% as.numeric()

gh_labs <- colnames(gh_character_cooc) %>% as.data.frame()
gh_labs$s <- diag(as.matrix(gh_character_cooc)) %>% as.numeric()

dool_labs <- colnames(dool_character_cooc) %>% as.data.frame()
dool_labs$s <- diag(as.matrix(dool_character_cooc)) %>% as.numeric()



##########################################Get gender Info#####################
p_load(data.table, plyr, stm, stringr, dplyr)

setwd("/Users/Oscar/Dropbox/MAPSS/Winter/Machine Learning/Final/Files")
AMC_ALL <- fread("amc_ml.csv") %>% as.data.frame()
DoOL_ALL <- fread("days_ml.csv") %>% as.data.frame()
GH_ALL <- fread("gh_ml.csv") %>% as.data.frame()


get_gender_mat <- function(turn_df, character_cooc){
  
  #Input: a df with turns annotated by speaker and gender, and a character co-occurrence matrix
  #Output: a matrix that distinguishes male-male ties (4) female-female ties (1) and mixes dies (2)
  
  name_gender <- unique(turn_df[c("speaker", "gender")])
  names_new <- as.data.frame(rownames(character_cooc))
  colnames(names_new) <- "speaker"
  name_gender <- join(name_gender, names_new, by = "speaker", type = "right")
  gender_codes <- ifelse(name_gender$gender == "M", 2, 1) 
  gender_mat <- gender_codes%o%gender_codes
  gender_mat = ifelse(character_cooc == 0,character_cooc, gender_mat)
  gender_ties <- as.tnet(gender_mat, type = "weighted one-mode tnet") %>% 
    as.data.frame() %>% 
    tnet_igraph()
  return(gender_ties)
}

amc_gender_net <- get_gender_mat(AMC_ALL, amc_character_cooc)
gh_gender_net <- get_gender_mat(GH_ALL, gh_character_cooc)
dool_gender_net <- get_gender_mat(DoOL_ALL, dool_character_cooc)

assign_colors <- function(gender_ties_list){
  gender_tie_colors <- ifelse(E(gender_ties_list)$weight == 1, "dodgerblue", NA)
  gender_tie_colors <- ifelse(E(gender_ties_list)$weight == 4, "deeppink", gender_tie_colors)
  gender_tie_colors <- ifelse(E(gender_ties_list)$weight == 2, "chartreuse2", gender_tie_colors)
  return(gender_tie_colors)
}


amc_gender_tie_colors <- assign_colors(amc_gender_net)
gh_gender_tie_colors <- assign_colors(gh_gender_net)
dool_gender_tie_colors <- assign_colors(dool_gender_net)


##########################################Plot############################

#################################
###############AMC###############
#################################

#Plot dimensions:
old.par <- par(mar = c(0, 0, 0, 0))
par(old.par)
l <- layout_with_fr(amc_gender_net)
l <- norm_coords(l, ymin=-1.2, ymax=1.2, xmin=-1.75, xmax=1.75)

#Plot
set.seed(2313442)
plot(amc_net, vertex.shape="none", vertex.size = .001,
     edge.width=E(amc_net)$weight/(dim(AMC_ALL)[1]/1000), vertex.label=amc_names$.,
     layout = l,
     #vertex.color = community2$membership+2 ,
     #vertex.frame.color= community2$membership+2,
     vertex.label.font=1, rescale=F, layout=l*1.2,
     vertex.label.color = "black",
     vertex.label.dist= -.5,
     vertex.label.cex=(amc_labs$s/20000)^.3, edge.color=amc_gender_tie_colors,
     main = "AMC",
     edge.curved=.2)



#################################
################GH###############
#################################
#Set params
old.par <- par(mar = c(0, 0, 0, 0))
par(old.par)
l <- layout_with_fr(gh_gender_net)
l <- norm_coords(l, ymin=-1.2, ymax=1.2, xmin=-1.75, xmax=1.75)

#Plot
set.seed(235213442)
plot(gh_net, vertex.shape="none", vertex.size = .001,
     edge.width=E(gh_net)$weight/(dim(GH_ALL)[1]/1000), vertex.label=gh_names$.,
     layout = l,
     #vertex.color = community2$membership+2 ,
     #vertex.frame.color= community2$membership+2,
     vertex.label.font=1, rescale=F, layout=l*1.2,
     vertex.label.color = "black",
     vertex.label.dist= -.5,
     vertex.label.cex=(gh_labs$s/20000)^.3, edge.color=gh_gender_tie_colors,
     main = "GH",
     edge.curved=.2)

#################################
################DOOL#############
#################################
#Set params
old.par <- par(mar = c(0, 0, 0, 0))
par(old.par)
l <- layout_with_fr(dool_gender_net)
l <- norm_coords(l, ymin=-1.2, ymax=1.2, xmin=-1.75, xmax=1.75)


set.seed(235213442)
plot(dool_net, vertex.shape="none", vertex.size = .001,
     edge.width=E(dool_net)$weight/(dim(DoOL_ALL)[1]/1000), vertex.label=dool_names$.,
     layout = l,
     #vertex.color = community2$membership+2 ,
     #vertex.frame.color= community2$membership+2,
     vertex.label.font=1, rescale=F, layout=l*1.2,
     vertex.label.color = "black",
     vertex.label.dist= -.5,
     vertex.label.cex=(dool_labs$s/15000)^.3, edge.color=dool_gender_tie_colors,
     main = "DoOL",
     edge.curved=.2)



##########################################Average tie strengths############################

get_avg <- function(character_net, gender_net){
  female_tie_avg <- ifelse(E(gender_net)$weight == 1, E(character_net)$weight, NA) %>% 
    na.omit() %>% mean()
  male_tie_avg <- ifelse(E(gender_net)$weight == 4, E(character_net)$weight, NA) %>% 
    na.omit() %>% mean()
  cross_tie_avg <- ifelse(E(gender_net)$weight == 2, E(character_net)$weight, NA) %>% 
    na.omit() %>% mean()
  mat = matrix( c("ff",female_tie_avg, "mm", male_tie_avg, "cross", cross_tie_avg),  nrow=2, ncol=3) 
  return(mat)
}

amc_tie_strength <- get_avg(amc_net, amc_gender_net)
amc_tie_strength
gh_tie_strength <- get_avg(gh_net, gh_gender_net)
gh_tie_strength
dool_tie_strength <- get_avg(dool_net, dool_gender_net)
dool_tie_strength

rbind(amc_tie_strength, gh_tie_strength, dool_tie_strength)

##########################################Betweenness Centrality############################
#Here I compute the top 10 characters for each show in terms of betweenness centrality
#For the betweeness centrality I use the paths that are the "thickest."
get_central_characters <- function(character_net, character_cooc, i){
  nf <- betweenness(character_net, directed = T,
                                         weights = 1/E(character_net)$weight, 
                                         nobigint = F, normalize = F)
  central_list <- data.table(t(rbind(nf, colnames(character_cooc))))
  central_list$nf <- as.numeric(as.character(central_list$nf))
  sorted <- central_list[order(central_list$nf, decreasing = T),]
  sorted <- sorted[1:i,]
  names(sorted)[names(sorted) == 'nf'] <- 'bet_centrality'
  return(sorted)
}

amc_top <- get_central_characters(amc_net, amc_character_cooc, 50)
gh_top <- get_central_characters(gh_net , gh_character_cooc,50)
dool_top <- get_central_characters(dool_net, dool_character_cooc, 48)


setwd("/Users/Oscar/Dropbox/MAPSS/Winter/Computational Contant Analysis/Final")

amc_foo <- as.data.frame(colSums(amc_character_scenes))
amc_foo$V2 <- rownames(amc_foo)
amc_top <- join(amc_top, amc_foo, by = "V2", type = "left")
write.csv(amc_top, "amc_top.csv")

gh_foo <- as.data.frame(colSums(gh_character_scenes))
gh_foo$V2 <- rownames(gh_foo)
gh_top <- join(gh_top, gh_foo, by = "V2", type = "left")
write.csv(gh_top, "gh_top.csv")

dool_foo <- as.data.frame(colSums(dool_character_scenes))
dool_foo$V2 <- rownames(dool_foo)
dool_top <- join(dool_top, dool_foo, by = "V2", type = "left")
write.csv(dool_top, "dool_top.csv")


amc_gender <- unique(AMC_ALL[c("speaker", "gender")])
gh_gender <- unique(GH_ALL[c("speaker", "gender")])
dool_gender <- unique(DoOL_ALL[c("speaker", "gender")])


names(amc_top)[names(amc_top) == 'V2'] <- 'speaker'
names(gh_top)[names(gh_top) == 'V2'] <- 'speaker'
names(dool_top)[names(dool_top) == 'V2'] <- 'speaker'


amc_top <- join(amc_top, amc_gender, by = "speaker", type = "left")
gh_top <- join(gh_top, gh_gender, by = "speaker", type = "left")
dool_top <- join(dool_top, dool_gender, by = "speaker", type = "left")

amc_top$gender <- as.factor(amc_top$gender)
gh_top$gender <- as.factor(gh_top$gender)
dool_top$gender <- as.factor(dool_top$gender)



##########################################Closeness Centrality############################

get_central_characters_closeness <- function(character_net, character_cooc, i){
  nf <- closeness(character_net, mode = "all",
                  weights = 1/E(character_net)$weight, 
                  normalize = F)
  central_list <- data.table(t(rbind(nf, colnames(character_cooc))))
  central_list$nf <- as.numeric(as.character(central_list$nf))
  sorted <- central_list[order(central_list$nf, decreasing = T),]
  sorted <- sorted[1:i,]
  names(sorted)[names(sorted) == 'V2'] <- 'speaker'
  names(sorted)[names(sorted) == 'nf'] <- 'closeness_centrality'
  return(sorted)
}


amc_top_close <- get_central_characters_closeness(amc_net, amc_character_cooc, 50)
gh_top_close <- get_central_characters_closeness(gh_net, gh_character_cooc, 50)
dool_top_close <- get_central_characters_closeness(dool_net, dool_character_cooc, 48)


amc_top <- join(amc_top, amc_top_close, by = "speaker", type = "left")
gh_top <- join(gh_top, gh_top_close, by = "speaker", type = "left")
dool_top <- join(dool_top, dool_top_close, by = "speaker", type = "left")


names(amc_top)[names(amc_top) == 'colSums(amc_character_scenes)'] <- 'scenes'
names(gh_top)[names(gh_top) == 'colSums(gh_character_scenes)'] <- 'scenes'
names(dool_top)[names(dool_top) == 'colSums(dool_character_scenes)'] <- 'scenes'

#Standardize variables
amc_top$bet_centrality <- scale(amc_top$bet_centrality, center = T, scale = T)
amc_top$closeness_centrality <- scale(amc_top$closeness_centrality, center = T, scale = T)
amc_top$scenes <- scale(amc_top$scenes, center = T, scale = T)
gh_top$bet_centrality <- scale(gh_top$bet_centrality, center = T, scale = T)
gh_top$closeness_centrality <- scale(gh_top$closeness_centrality, center = T, scale = T)
gh_top$scenes <- scale(gh_top$scenes, center = T, scale = T)
dool_top$bet_centrality <- scale(dool_top$bet_centrality, center = T, scale = T)
dool_top$closeness_centrality <- scale(dool_top$closeness_centrality, center = T, scale = T)
dool_top$scenes <- scale(dool_top$scenes, center = T, scale = T)



#Bind together
foo <- rbind(amc_top, gh_top, dool_top)

#Write out
p_load(openxlsx)
openxlsx::write.xlsx(foo, file = "centrality_stats_df.xlsx", firstRow=T)

foo$closeness_centrality <- round(foo$closeness_centrality, digits = 2)
write.csv(foo, "centrality_stats_df.csv")




#######Nor run simple OLS model
library(broom)

closeness_lmodel <- lm(foo$closeness_centrality ~ gender + scenes, data=foo)
summary(closeness_lmodel)
closeness_lmodel <- tidy(closeness_lmodel)
closeness_lmodel <-  round(closeness_lmodel[2:5],digits = 2)

#Write out OLS model
p_load(openxlsx)
openxlsx::write.xlsx(closeness_lmodel, file = "closeness_regression_model.xlsx", firstRow=T)


####Histograms of centralities
hist(foo$bet_centrality, breaks = 50, main = "Histogram of betweenness centrality")
hist(foo$closeness_centrality, breaks = 50, main = "Histogram of closeness centrality")


