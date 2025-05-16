# --- Install and Load Libraries ---
if (!requireNamespace("xgboost", quietly = TRUE)) install.packages("xgboost")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("xts", quietly = TRUE)) install.packages("xts")

library(xgboost)
library(dplyr)
library(xts)

# --- Load Data and Feature Engineering (Assuming you have this) ---
data <- read.csv("C:/r/sp500.csv")
data$Date <- as.Date(data$Date)
data <- data %>% arrange(Date)

lags <- 10
for (i in 1:lags) {
  data[[paste0("Lag_", i)]] <- lag(data$GSPC.Adjusted, n = i)
}
data$Month <- as.integer(format(data$Date, "%m"))
data$DayOfWeek <- as.integer(format(data$Date, "%w"))
data <- data %>% drop_na()
feature_names <- paste0("Lag_", 1:10)
features <- data[, feature_names]
target <- lead(data$GSPC.Adjusted, n = 1)
features <- head(features, -1)
target <- head(target, -1)

# Split Data
train_size <- floor(0.7 * nrow(features))
validation_size <- floor(0.15 * nrow(features))
train_features <- features[1:train_size, ]
validation_features <- features[(train_size + 1):(train_size + validation_size), ]
test_features <- features[(train_size + validation_size + 1):nrow(features), ]
train_target <- target[1:train_size]
validation_target <- target[(train_size + 1):(train_size + validation_size)]
test_target <- target[(train_size + validation_size + 1):nrow(features)]
test_dates <- data$Date[(train_size + validation_size + 2):nrow(data)] # Dates for test predictions

# --- Train XGBoost Model (Assuming you have a trained model named 'model_xgb') ---
dtrain <- xgb.DMatrix(as.matrix(train_features), label = train_target)
dvalidate <- xgb.DMatrix(as.matrix(validation_features), label = validation_target)
model_xgb <- xgb.train(params = list(objective = "reg:squarederror", eval_metric = "mae"),
                       data = dtrain,
                       nrounds = 100, # Reduced for faster simulation
                       watchlist = list(train = dtrain, eval = dvalidate),
                       early_stopping_rounds = 10,
                       verbose = 0)

# --- Make Predictions on Test Set ---
predictions_xgb <- predict(model_xgb, xgb.DMatrix(as.matrix(test_features)))

# --- Simulate Real-time Output in Console ---
cat("Simulated Real-time Forecast:\n")
for (i in 1:length(test_dates)) {
  cat(paste0("Date: ", test_dates[i], "\n"))
  cat(paste0("  Actual:   ", round(test_target[train_size + validation_size + i], 2), "\n"))
  cat(paste0("  Forecast: ", round(predictions_xgb[i], 2), "\n"))
  
  # Simple Text-based Plot (Scales to the range of your data)
  scale_factor <- 0.01 # Adjust this to fit your price range
  actual_scaled <- round((test_target[train_size + validation_size + i] - min(test_target)) * scale_factor)
  forecast_scaled <- round((predictions_xgb[i] - min(test_target)) * scale_factor)
  
  plot_line <- paste0(
    "  Plot: ",
    paste(rep(" ", max(0, actual_scaled)), collapse = ""),
    "A", # Actual 
    paste(rep("-", max(0, forecast_scaled - actual_scaled)), collapse = ""),
    ifelse(forecast_scaled >= actual_scaled, "F", ""), # Forecast
    paste(rep("-", max(0, actual_scaled - forecast_scaled)), collapse = ""),
    ifelse(forecast_scaled < actual_scaled, "F", "")  # Forecast
  )
  cat(plot_line, "\n\n")
  Sys.sleep(0.5) # Pause for a short duration to simulate time
}


xgboost
