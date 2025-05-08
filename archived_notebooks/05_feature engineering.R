
install.packages("dplyr")       
library(dplyr)


stock_and_econ_cleaned <- stock_and_econ_cleaned %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))


logical_columns <- c(
  "Sector_Electrical Utilities & IPPs",
  "Sector_Food & Tobacco",
  "Sector_Healthcare Equipment & Supplies",
  "Sector_Hotels & Entertainment Services",
  "Sector_Insurance",
  "Sector_Investment Banking & Investment Services",
  "Sector_Machinery, Equipment & Components",
  "Sector_Media & Publishing",
  "Sector_Oil & Gas",
  "Sector_Other",
  "Sector_Pharmaceuticals",
  "Sector_Professional & Commercial Services",
  "Sector_Residential & Commercial REIT",
  "Sector_Semiconductors & Semiconductor Equipment",
  "Sector_Software & IT Services"
)


stock_and_econ_cleaned <- stock_and_econ_cleaned %>%
  mutate(across(all_of(logical_columns), ~ as.numeric(.)))


sum(is.na(stock_and_econ_cleaned))  

stock_and_econ_cleaned <- stock_and_econ_cleaned %>%
  mutate(
    Quarter_Num = case_when(
      Quarter == "Q1" ~ 1,
      Quarter == "Q2" ~ 2,
      Quarter == "Q3" ~ 3,
      Quarter == "Q4" ~ 4
    ),
    Quarter_sin = sin(2 * pi * Quarter_Num / 4),
    Quarter_cos = cos(2 * pi * Quarter_Num / 4)
  )

install.packages("writexl")
library(writexl)

write_xlsx(stock_and_econ_cleaned, "stock_and_econ_cleaned.xlsx")



stock_and_econ_cleaned <- stock_and_econ_cleaned %>%
  mutate(PE_Category_Num = case_when(
    PE_Category == "Low" ~ 0,
    PE_Category == "Good" ~ 1,
    PE_Category == "High" ~ 2,
    PE_Category == "Craziness" ~ 3,
    TRUE ~ 0  # Default for unexpected/missing
  ))

write_xlsx(stock_and_econ_cleaned, "stock_and_econ_cleaned.xlsx")

install.packages("factoextra")  
install.packages("ggplot2")    
library(factoextra)
library(ggplot2)


pca_data <- stock_and_econ_cleaned %>%
  select(where(is.numeric)) %>%
  select(-Label)  

pca_data_scaled <- scale(pca_data)


pca_result <- prcomp(pca_data_scaled, center = TRUE, scale. = TRUE)

summary(pca_result)

colnames(stock_and_econ_cleaned)

install.packages("glmnet")
install.packages("caret")  
library(glmnet)
library(caret)


stock_and_econ_cleaned$Label <- as.factor(stock_and_econ_cleaned$Label)


set.seed(123)
split <- createDataPartition(stock_and_econ_cleaned$Label, p = 0.8, list = FALSE)
train_data <- stock_and_econ_cleaned[split, ]
test_data  <- stock_and_econ_cleaned[-split, ]


x_train <- model.matrix(Label ~ ., data = train_data)[, -1]
y_train <- train_data$Label
x_test  <- model.matrix(Label ~ ., data = test_data)[, -1]
y_test  <- test_data$Label



cv_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0.5)


plot(cv_model)


best_lambda <- cv_model$lambda.min

