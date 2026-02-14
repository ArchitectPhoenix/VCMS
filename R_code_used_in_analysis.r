library(modelr)
library(tidyverse)
library(stargazer)
library(clinfun)

#Read the csv file dataset_ColucciFrancoValori2023 to start

dataToAnalyze<-dataset_ColucciFrancoValori2023

#Define some additional variables
dataToAnalyze <- dataToAnalyze %>%
  mutate(cond2=factor(cond, levels=c("buyer","now","day","week","month"),ordered=TRUE))
dataToAnalyze <- dataToAnalyze %>%
  mutate(frameQmod = frameQ)

dataToAnalyze$frameQmod[dataToAnalyze$frameQ==0] <- NA
dataToAnalyze$frameQmod[dataToAnalyze$frameQ==2] <- 0


#Perform regressions
formula1mug<-Mug_assess~cond*frameQmod+Sex+age+`Current Country of Residence`
formula1amazon<-Amazon_assess~cond*frameQmod+Sex+age+`Current Country of Residence`
formula1spotify<-Spotify_assess~cond*frameQmod+Sex+age+`Current Country of Residence`

formula2mug<-Mug_assess~cond*frameQ+Sex+age+`Current Country of Residence`
formula2amazon<-Amazon_assess~cond*frameQ+Sex+age+`Current Country of Residence`
formula2spotify<-Spotify_assess~cond*frameQ+Sex+age+`Current Country of Residence`

formula3mug<-Mug_assess~cond+frameQmod+Sex+age+`Current Country of Residence`
formula3amazon<-Amazon_assess~cond+frameQmod+Sex+age+`Current Country of Residence`
formula3spotify<-Spotify_assess~cond+frameQmod+Sex+age+`Current Country of Residence`

mod1_mug<-lm(formula1mug,data=dataToAnalyze,subset = (Sex!="DATA EXPIRED"))
summary(mod1_mug)
mod1_amazon<-lm(formula1amazon,data=dataToAnalyze,subset = (Sex!="DATA EXPIRED"))
summary(mod1_amazon)
mod1_spotify<-lm(formula1spotify,data=dataToAnalyze,subset = (Sex!="DATA EXPIRED"))
summary(mod1_spotify)
mod2_mug<-lm(formula1mug,data=dataToAnalyze,subset = (Sex!="DATA EXPIRED" & cond !="buyer"))
summary(mod2_mug)
mod2_amazon<-lm(formula1amazon,data=dataToAnalyze,subset = (Sex!="DATA EXPIRED" & cond !="buyer"))
summary(mod2_amazon)
mod2_spotify<-lm(formula1spotify,data=dataToAnalyze,subset = (Sex!="DATA EXPIRED" & cond !="buyer"))
summary(mod2_spotify)
mod3_mug<-lm(formula2mug,data=dataToAnalyze,subset = (Sex!="DATA EXPIRED"))
summary(mod3_mug)
mod3_amazon<-lm(formula2amazon,data=dataToAnalyze,subset = (Sex!="DATA EXPIRED"))
summary(mod3_amazon)
mod3_spotify<-lm(formula2spotify,data=dataToAnalyze,subset = (Sex!="DATA EXPIRED"))
summary(mod3_spotify)
mod4_mug<-lm(formula2mug,data=dataToAnalyze,subset = (Sex!="DATA EXPIRED" & cond !="buyer"))
summary(mod4_mug)
mod4_amazon<-lm(formula2amazon,data=dataToAnalyze,subset = (Sex!="DATA EXPIRED" & cond !="buyer"))
summary(mod4_amazon)
mod4_spotify<-lm(formula2spotify,data=dataToAnalyze,subset = (Sex!="DATA EXPIRED" & cond !="buyer"))
summary(mod4_spotify)
mod5_mug<-lm(formula3mug,data=dataToAnalyze,subset = (Sex!="DATA EXPIRED"))
summary(mod5_mug)
mod5_amazon<-lm(formula3amazon,data=dataToAnalyze,subset = (Sex!="DATA EXPIRED"))
summary(mod5_amazon)
mod5_spotify<-lm(formula3spotify,data=dataToAnalyze,subset = (Sex!="DATA EXPIRED"))
summary(mod5_spotify)
mod6_mug<-lm(formula3mug,data=dataToAnalyze,subset = (Sex!="DATA EXPIRED" & cond !="buyer"))
summary(mod6_mug)
mod6_amazon<-lm(formula3amazon,data=dataToAnalyze,subset = (Sex!="DATA EXPIRED" & cond !="buyer"))
summary(mod6_amazon)
mod6_spotify<-lm(formula3spotify,data=dataToAnalyze,subset = (Sex!="DATA EXPIRED" & cond !="buyer"))
summary(mod6_spotify)

stargazer(mod1_mug,mod2_mug,mod3_mug,mod4_mug,mod5_mug,mod6_mug,type="html",dep.var.labels=c("Mug"),out="models_mug.htm")
stargazer(mod1_amazon,mod2_amazon,mod3_amazon,mod4_amazon,mod5_amazon,mod6_amazon,type="html",dep.var.labels=c("Amazon"),out="models_amazon.htm")
stargazer(mod1_spotify,mod2_spotify,mod3_spotify,mod4_spotify,mod5_spotify,mod6_spotify,type="html",dep.var.labels=c("Spotify"),out="models_spotify.htm")

stargazer(mod5_mug,mod5_amazon,mod5_spotify,type="latex",dep.var.labels=c("Mug","Amazon","Spotify"),star.cutoffs = c(0.05, 0.01, 0.001),out="models_all.tex")

#----------------
###Wilcoxon on 'future' vs Now
wtaM_future<-dataToAnalyze$Mug_WTA[!is.na(dataToAnalyze$Mug_WTA) & dataToAnalyze$cond!="now"]
wtaM_now<-dataToAnalyze$Mug_WTA[!is.na(dataToAnalyze$Mug_WTA) & dataToAnalyze$cond=="now"]
wilcox.test(wtaM_now,wtaM_future)

wtaA_future<-dataToAnalyze$Amazon_WTA[!is.na(dataToAnalyze$Amazon_WTA) & dataToAnalyze$cond!="now"]
wtaA_now<-dataToAnalyze$Amazon_WTA[!is.na(dataToAnalyze$Amazon_WTA) & dataToAnalyze$cond=="now"]
wilcox.test(wtaA_now,wtaA_future)

wtaS_future<-dataToAnalyze$Spotify_WTA[!is.na(dataToAnalyze$Spotify_WTA) & dataToAnalyze$cond!="now"]
wtaS_now<-dataToAnalyze$Spotify_WTA[!is.na(dataToAnalyze$Spotify_WTA) & dataToAnalyze$cond=="now"]
wilcox.test(wtaS_now,wtaS_future)

###

### Table 6: Kruskal-Wallis

with(dataToAnalyze,kruskal.test(list(MugBuyer_WTP,MugNow_WTA,MugDay_WTA, MugWeek_WTA, MugMonth_WTA)))
with(dataToAnalyze,kruskal.test(list(MugNow_WTA,MugDay_WTA, MugWeek_WTA, MugMonth_WTA)))

with(dataToAnalyze,kruskal.test(list(AmazonBuyer_WTP,AmazonNow_WTA,AmazonDay_WTA, AmazonWeek_WTA, AmazonMonth_WTA)))
with(dataToAnalyze,kruskal.test(list(AmazonNow_WTA,AmazonDay_WTA, AmazonWeek_WTA, AmazonMonth_WTA)))

with(dataToAnalyze,kruskal.test(list(SpotifyBuyer_WTP,SpotifyNow_WTA,SpotifyDay_WTA, SpotifyWeek_WTA, SpotifyMonth_WTA)))
with(dataToAnalyze,kruskal.test(list(SpotifyNow_WTA,SpotifyDay_WTA, SpotifyWeek_WTA, SpotifyMonth_WTA)))

### Table 7: Wilcoxon test for WTP with respect to the “Now” WTA
#Mug
wtpM<-dataToAnalyze$Mug_assess[dataToAnalyze$cond=="buyer"]
wtaM_now<-dataToAnalyze$Mug_assess[dataToAnalyze$cond=="now"]
wilcox.test(wtpM,wtaM_now, alternative = 'less')

#Amazon
wtpA<-dataToAnalyze$Amazon_assess[dataToAnalyze$cond=="buyer"]
wtaA_now<-dataToAnalyze$Amazon_assess[dataToAnalyze$cond=="now"]
wilcox.test(wtpA,wtaA_now, alternative = 'less')

#Spotify
wtpS<-dataToAnalyze$Spotify_assess[dataToAnalyze$cond=="buyer"]
wtaS_now<-dataToAnalyze$Spotify_assess[dataToAnalyze$cond=="now"]
wilcox.test(wtpS,wtaS_now, alternative = 'less')

### Table 8: Wilcoxon test for differences with respect to the “Now” WTA

wtaM_day<-dataToAnalyze$Mug_WTA[!is.na(dataToAnalyze$Mug_WTA) & dataToAnalyze$cond=="day"]
wtaM_week<-dataToAnalyze$Mug_WTA[!is.na(dataToAnalyze$Mug_WTA) & dataToAnalyze$cond=="week"]
wtaM_month<-dataToAnalyze$Mug_WTA[!is.na(dataToAnalyze$Mug_WTA) & dataToAnalyze$cond=="month"]
wilcox.test(wtaM_now,wtaM_day, alternative="less")
wilcox.test(wtaM_now,wtaM_week, alternative="less")
wilcox.test(wtaM_now,wtaM_month, alternative="less")


wtaA_day<-dataToAnalyze$Amazon_WTA[!is.na(dataToAnalyze$Amazon_WTA) & dataToAnalyze$cond=="day"]
wtaA_week<-dataToAnalyze$Amazon_WTA[!is.na(dataToAnalyze$Amazon_WTA) & dataToAnalyze$cond=="week"]
wtaA_month<-dataToAnalyze$Amazon_WTA[!is.na(dataToAnalyze$Amazon_WTA) & dataToAnalyze$cond=="month"]
wilcox.test(wtaA_now,wtaA_day, alternative="less")
wilcox.test(wtaA_now,wtaA_week, alternative="less")
wilcox.test(wtaA_now,wtaA_month, alternative="less")

wtaS_day<-dataToAnalyze$Spotify_WTA[!is.na(dataToAnalyze$Spotify_WTA) & dataToAnalyze$cond=="day"]
wtaS_week<-dataToAnalyze$Spotify_WTA[!is.na(dataToAnalyze$Spotify_WTA) & dataToAnalyze$cond=="week"]
wtaS_month<-dataToAnalyze$Spotify_WTA[!is.na(dataToAnalyze$Spotify_WTA) & dataToAnalyze$cond=="month"]
wilcox.test(wtaS_now,wtaS_day, alternative="less")
wilcox.test(wtaS_now,wtaS_week, alternative="less")
wilcox.test(wtaS_now,wtaS_month, alternative="less")


### Table 9: Jonckheere-Terpstra test
jonckheere.test(dataToAnalyze$Mug_assess[dataToAnalyze$cond2!="buyer"],dataToAnalyze$cond2[dataToAnalyze$cond2!="buyer"],alternative = "increasing")
jonckheere.test(dataToAnalyze$Amazon_assess[dataToAnalyze$cond2!="buyer"],dataToAnalyze$cond2[dataToAnalyze$cond2!="buyer"],alternative = "increasing")
jonckheere.test(dataToAnalyze$Spotify_assess[dataToAnalyze$cond2!="buyer"],dataToAnalyze$cond2[dataToAnalyze$cond2!="buyer"],alternative = "increasing")
dataToAnalyze <- dataToAnalyze %>%
  mutate(cond3=ifelse(cond2=='buyer',0,ifelse(cond2=='now',1,2)))
jonckheere.test(dataToAnalyze$Mug_assess,dataToAnalyze$cond3,alternative = "increasing")
jonckheere.test(dataToAnalyze$Amazon_assess,dataToAnalyze$cond3,alternative = "increasing")
jonckheere.test(dataToAnalyze$Spotify_assess,dataToAnalyze$cond3,alternative = "increasing")