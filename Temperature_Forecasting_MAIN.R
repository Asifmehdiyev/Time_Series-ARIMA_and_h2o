# **** Import necessary libraries ****
library(tidyverse)
library(inspectdf)
library(dplyr)
library(data.table)
library(lubridate)
library(timetk)
library(skimr)
library(highcharter)
library(h2o)
library(rsample)
library(forecast)
library(tidymodels)
library(modeltime)

# ****** Import dataset and get familirized with it ******
data <- fread("daily-minimum-temperatures-in-me (1).csv")
View(data)

data %>% glimpse()

data %>% inspect_na()

names(data)

names(data) <- data %>% names() %>% gsub(" ", "_",.)

unique((data$Daily_minimum_temperatures))

data$Daily_minimum_temperatures <- parse_number(data$Daily_minimum_temperatures)

data$Date <- data$Date %>% as.Date(.,"%m/%d/%Y")

data %>% 
  plot_time_series(
    Date, Daily_minimum_temperatures, 
    .color_var = lubridate::year(Date),
    # .color_lab = "Year",
    .interactive = T,
    .plotly_slider = T,
    .smooth = F)

# ************* Seasonality plots **********
data %>%
  plot_seasonal_diagnostics(
    Date, Daily_minimum_temperatures, .interactive = T)

data <- data %>% tk_augment_timeseries_signature(Date) %>% select(Daily_minimum_temperatures,everything())

data <- data %>%
  select(-contains("hour"),
         -contains("day"),
         -contains("week"),
         -minute,-second,-am.pm) %>% 
  mutate_if(is.ordered, as.character) %>% 
  mutate_if(is.character,as_factor)

h2o.init()

h2o_data <- data %>% as.h2o()

train <- data %>% filter(year < 1988) %>% as.h2o()
test <- data %>% filter(year >= 1988) %>% as.h2o()

target <- data[,1] %>% names()
features <- data[,-1] %>% names()

# ***************  Modelling ************

model<- h2o.automl(
  x = features, y = target, 
  training_frame = train, 
  validation_frame = test,
  leaderboard_frame = test,
  stopping_metric = "RMSE",
  exclude_algos = c("DRF", "GBM", "GLM", "XGBoost"),
  seed = 123, nfolds = 10,
  max_runtime_secs = 480) 

model@leaderboard %>% as.data.frame() 
model <- model@leader

# ***************** Predicting test results *******
y_pred <- model %>% h2o.predict(newdata = test) %>% as.data.frame()
y_pred$predict

model %>% 
  h2o.rmse(train = T,
           valid = T,
           xval = T)

error_tbl <- data %>% 
  filter(lubridate::year(Date) >= 1988) %>% 
  add_column(pred = y_pred %>% as_tibble() %>% pull(predict)) %>%
  rename(actual = Daily_minimum_temperatures) %>% 
  select(Date,actual,pred)

highchart() %>% 
  hc_xAxis(categories = error_tbl$Date) %>% 
  hc_add_series(data=error_tbl$actual, type='line', color='red', name='Actual') %>% 
  hc_add_series(data=error_tbl$pred, type='line', color='green', name='Predicted') %>% 
  hc_title(text='Predict')

# ************** New data (next 1 year) *************
new_data <- seq(as.Date("1991/01/01"), as.Date("1991/12/01"), "months") %>%
  as_tibble() %>% 
  add_column(Daily_minimum_temperatures=0) %>% 
  rename(Date=value) %>% 
  tk_augment_timeseries_signature() %>%
  select(-contains("hour"),
         -contains("day"),
         -contains("week"),
         -minute,-second,-am.pm) %>% 
  mutate_if(is.ordered, as.character) %>% 
  mutate_if(is.character,as_factor)


# *************** Forecast ***********

new_h2o <- new_data %>% as.h2o()

new_predictions <- h2o_leader %>% 
  h2o.predict(new_h2o) %>% 
  as_tibble() %>%
  add_column(Date=new_data$Date) %>% 
  select(Date,predict) %>% 
  rename(Daily_minimum_temperatures=predict)

data %>% 
  bind_rows(new_predictions) %>% 
  mutate(colors=c(rep('Actual',3650),rep('Predicted',12))) %>% 
  hchart("line", hcaes(Date, Daily_minimum_temperatures, group = colors)) %>% 
  hc_title(text='Forecast') %>% 
  hc_colors(colors = c('red','green'))

# ******************* Model Evaluation ***********

test_set <- test %>% as.data.frame()
residuals= test_set$Daily_minimum_temperatures - y_pred$predict

# *********** Calculate RMSE (Root Mean Square Error) ******
RMSE = sqrt(mean(residuals^2))

# ********** Calculate Adjusted R2 (R Squared) *******
y_test_mean = mean(test_set$Daily_minimum_temperatures)

tss = sum((test_set$Daily_minimum_temperatures - y_test_mean)^2) #total sum of squares
rss = sum(residuals^2) #residual sum of squares

R2 = 1 - (rss/tss); R2

n <- test_set %>% nrow() #sample size
k <- features %>% length() #number of independent variables
Adjusted_R2 = 1-(1-R2)*((n-1)/(n-k-1))

tibble(RMSE = round(RMSE),
       R2, Adjusted_R2)


# ************ arima reg (auto arima) *********
splits <- initial_time_split(data, prop=0.8)

# ********** Auto_arima *************
model_fit_arima <- arima_reg() %>%
  set_engine("auto_arima") %>%
  fit(Daily_minimum_temperatures ~ Date, data=training(splits))
model_table <- modeltime_table(model_fit_arima)

# **************  Arima_boost *******
model_fit_arima <- arima_boost(
  min_n = 2, 
  learn_rate = 0.01
) %>% 
  set_engine(engine="auto_arima_xgboost") %>% 
  fit(Daily_minimum_temperatures ~ Date, 
      data = training(splits))

# ************** Prophet ***********

model_fit_prophet <- prophet_reg() %>%
  set_engine("prophet") %>%
  fit(Daily_minimum_temperatures ~ Date, 
      data = training(splits))
# *********************** Prophet boost ********
model_spec_prophet_boost <- prophet_boost() %>%
  set_engine("prophet_xgboost") 

recipe_spec <- recipe(Daily_minimum_temperatures ~ Date, data = training(splits)) %>%
  step_timeseries_signature(Date) %>%
  step_fourier(Date, period = 365, K = 2) %>%
  step_dummy(all_nominal())


recipe_spec %>% prep() %>% juice()


workflow_fit_prophet_boost <- workflow() %>%
  add_model(model_spec_prophet_boost) %>%
  add_recipe(recipe_spec) %>%
  fit(data = training(splits))


models_tbl <- modeltime_table(
  model_fit_arima,
  model_fit_prophet,
  workflow_fit_prophet_boost)

# *********** Calibration ********
calibration_tbl <- models_tbl %>% 
  modeltime_calibrate(new_data = testing(splits))

# *********** Forecast *************
calibration %>% 
  #filter(.model_id == 3) %>% 
  modeltime_forecast(
    new_data = testing(splits),
    actual_data = data
  ) %>% 
  plot_modeltime_forecast(
    .legend_max_width = 25,
    .interactive = T
  )

calibration_tbl %>% modeltime_accuracy() %>% 
  table_modeltime_accuracy(.interactive = T)

# *************** Forecast Forward *********
calibration %>%
  filter(.model_id %in% 2) %>% # best model
  modeltime_refit(data) %>%
  modeltime_forecast(h = "1 year", 
                     new_data = testing(splits),
                     actual_data = data) %>%
  plot_modeltime_forecast(.interactive = T,
                          .plotly_slider = T,
                          .legend_show = F)

