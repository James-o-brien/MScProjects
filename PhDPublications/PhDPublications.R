# MSc, Generalised Linear Models, Assessed Practical 

# Section 2: Assessed Exercise

setwd('C:/Users/james/OneDrive/Documents/Oxford/Michaelmas/Applied Statistics')

#
### Q1.
#

set.seed(453)
pub <- read.csv("pub.csv")
dim(pub)
names(pub)
head(pub)

# Define factors:
pub$kids <- factor(pub$kids, levels = c("0", "1", "2", "3"))
pub$female <- factor(pub$female, levels = c("0", "1"))
levels(pub$female) <- c(levels(pub$female), "Female")
levels(pub$female) <- c(levels(pub$female), "Male")
pub$female[pub$female == "1"] <- "Female"
pub$female[pub$female == "0"] <- "Male"
pub$female <- factor(pub$female, levels = c("Male", "Female"))

pub$married <- factor(pub$married, levels = c("0", "1"))
levels(pub$married) <- c(levels(pub$married), "Yes")
levels(pub$married) <- c(levels(pub$married), "No")
pub$married[pub$married == "1"] <- "Yes"
pub$married[pub$married == "0"] <- "No"
pub$married <- factor(pub$married, levels = c("No", "Yes"))

# Numerical summaries:
summary(pub)

# Graphical summaries:
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73",
               "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
library(ggplot2)
library(grid)
library(GGally)

# Histogram of articles:
ggplot(pub, aes(x = articles, color = 0)) +
  geom_histogram(fill = cbPalette[3], position = "dodge", binwidth = 1) +
  ggtitle("Number of articles published") +
  xlab("\n Articles") + ylab("Frequency \n") +
  theme_classic() +
  theme(text = element_text(size = 18),
        plot.title = element_text(hjust = 0.5, size = 20, face = "bold")) +
  theme(legend.position = 'none',
        plot.margin = unit(c(2, 1, 1, 1), "lines"))

# Scatterplots of continuous variables:
par(mfrow=c(1,2))
plot(pub$mentor, (pub$articles), col = pub$female, pch = 16, 
     cex = 1.4,
     cex.main = 1.4,
     cex.axis = 1.2,
     cex.lab = 1.2,
     main = "Relationship between mentor and articles",
     xlab = "Mentor",
     ylab = "Articles")
text(paste("Correlation:", round(cor(pub$mentor, pub$articles), 2)), x = 60, y = 15,
     cex = 1.2)
legend("topright", legend = c("Male", "Female"), col = c("black", "red"), cex = 1.2, pch = 16)
plot(pub$prestige, pub$articles, col = pub$female, pch = 16, 
     cex = 1.4,
     cex.main = 1.4,
     cex.axis = 1.2,
     cex.lab = 1.2,
     main = "Relationship between prestige and articles",
     xlab = "Prestige",
     ylab = "Articles")
text(paste("Correlation:", round(cor(pub$prestige, pub$articles), 2)), x = 3.75, y = 15,
     cex = 1.2)
legend("topright", legend = c("Male", "Female"), col = c("black", "red"), cex = 1.2, pch = 16)

# Histograms of continuous variables:
hist(pub$mentor, breaks = 25)
hist(pub$prestige, breaks = 25)

# Boxplots of categorical variables:
par(mfrow = c(1,3))
par(mar = c(5,5,4,2)+0.1)
boxplot(articles ~ female, main = "Distribution of sex", ylab = "Articles", 
        xlab = "Sex",col = cbPalette[3],
        cex = 2,
        cex.main = 2,
        cex.axis = 1.8,
        cex.lab = 1.8)
boxplot(articles ~ kids, main = "Distribution of kids", ylab = "Articles", 
        xlab = "No. of children", col = cbPalette[3],
        cex = 2,
        cex.main = 2,
        cex.axis = 1.8,
        cex.lab = 1.8)
boxplot(articles ~ married, main = "Distribution of marital status", ylab = "Articles", 
        xlab = "Married", col = cbPalette[3],
        cex = 2,
        cex.main = 2,
        cex.axis = 1.8,
        cex.lab = 1.8)
tapply(articles, female, var) # tapply to get mean, var, etc.
tapply(articles, kids, var)
tapply(articles, married, var)

#
### Q2.
#

# Defining a model with all possible variables and interactions:
pub.glm <- glm(formula = articles ~ female + married + kids + prestige + mentor +
                 female:married + female:kids + female:prestige + female:mentor, 
               data = pub, family = poisson)
summary(pub.glm)
step(pub.glm, direction = "both") # Stepwise Regression

# Model chosen by AIC:
pub.glm2 <- glm(formula = articles ~ female + married + kids + prestige + mentor +
                  female:prestige, data = pub, family = poisson)
summary(pub.glm2) # suggests prestige is not significant

# Model without prestige and female:prestige:
pub.glm3 <- glm(formula = articles ~ female + married + kids + mentor, data = pub, family = poisson)
summary(pub.glm3)

# Likelihood ratio tests. pub.glm vs pub.glm2:
dof <- pub.glm$rank - pub.glm2$rank
lrt <- deviance(pub.glm2) - deviance(pub.glm)
pval <- 1 - pchisq(lrt, dof)
cbind(lrt, dof, pval) # fail to reject

# pub.glm2 vs pub.glm3:
dof2 <- pub.glm2$rank - pub.glm3$rank
lrt2 <- deviance(pub.glm3) - deviance(pub.glm2)
pval2 <- 1 - pchisq(lrt2, dof2)
cbind(lrt2, dof2, pval2) # fail to reject but very close

#
### Q3.
#

# Note that pub.glm2 was also analysed in Q3, Q4 and Q5 using the same code.

# Cameron and Windmeijer (1997) KL/Deviance based R^2:
library(rsq)
rsq.kl(pub.glm3)

# Diagnostics:
par(mfrow=c(1,3))
plot(fitted(pub.glm3), rstandard(pub.glm3),
     xlab = expression(hat(lambda)), ylab = "Deviance Residuals",
     main = "Deviance Residuals",
     pch = 16, col = cbPalette[6],
     cex = 1.2,
     cex.main = 2,
     cex.axis = 1.8,
     cex.lab = 1.8)
p <- pub.glm3$rank
n <- nrow(model.frame(pub.glm3))
plot(influence(pub.glm3)$hat/(p/n), ylab='Leverage / (p/n)', xlab = "Student",
     pch = 16, col = cbPalette[6], main = "Leverage",
     cex = 1.2,
     cex.main = 2,
     cex.axis = 1.8,
     cex.lab = 1.8)
#text(influence(pub.glm3)$hat/(p/n), labels = rownames(pub), ylab='Leverage / (p/n)')
plot(cooks.distance(pub.glm3), ylab = "Cook's Distance", xlab = "Student", 
     pch = 16, col = cbPalette[6], main = "Cook's Distance",
     cex = 1.2,
     cex.main = 2,
     cex.axis = 1.8,
     cex.lab = 1.8)
abline(h = 8/(n-2*p),lty = 2)
#text(cooks.distance(pub.glm3), labels = rownames(pub))

# Removing influential points sequentially and refitting:
pub.out <- pub[-c(328, 915, 901, 914),]
pub.glm.out <- glm(formula = articles ~ female + married + kids + prestige + mentor +
                  female:married + female:kids + female:prestige + female:mentor, 
                data = pub.out, family = poisson)
step(pub.glm.out, direction = "both")
pub.glm.out2 <- glm(formula = articles ~ female + married + kids + prestige + mentor +
                      female:prestige, data = pub.out, family = poisson)
summary(pub.glm.out2) # prestige and female:prestige not significant
pub.glm.out3 <- glm(formula = articles ~ female + married + kids + mentor, 
                    data = pub.out, family = poisson)
summary(pub.glm.out3)

#
### Q4.
#

summary(pub.glm3)

# Confidence Intervals:
options(digits = 4)
confint.default(pub.glm3)
exp(confint.default(pub.glm3))

#
### Q5.
#

dp = sum(residuals(pub.glm3, type ="pearson") ^ 2)/pub.glm3$df.residual
dp

library(AER)
dispersiontest(pub.glm3) 

summary(pub.glm3, dispersion = dp)
exp(summary(pub.glm3, dispersion = dp)$coef[1:7])

# Confidence Intervals:
dp.beta17 <- summary(pub.glm3, dispersion = dp)$coef[1:7, 1]
dp.se17 <- summary(pub.glm3, dispersion = dp)$coef[1:7, 2]
dp.cval <- qnorm(0.975)
dp.lower <- dp.beta17 - dp.cval*dp.se17
dp.upper <- dp.beta17 + dp.cval*dp.se17
dp.ci95 <- cbind(dp.lower, dp.upper)
dp.ci95
exp(dp.ci95)
