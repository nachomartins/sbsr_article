####### PRIORITIZATION OF AREAS FOR PREVENTING AND FIGHTING FOREST FIRES #######
                    # FIRES IN THE BRAZILIAN AMAZON #

# This code was elaborated to produce the results of said paper that was accepted
# and will be presented at the 2025 Brazilian Remote Sensing Symposium.
# This work was written by Ignácio Martins Pinho as part of a master's dissertation

# To acess the data needed to run this code and to read the paper that contains
# more information about this work, access:

# This code is divided into 4 parts:
# 1: Installing and loading required packages and loading data
# 2: Correlations between variables
# 3: Running Random Forest Regression Models
# 4: Running best selected model
# 5: Plotting priority classification maps 

#_______________________________________________________________________________
##### 1: Installing and loading required packages and loading data #####

# List of required packages
packages <- c(
  "terra", "dplyr", "tidymodels", "ranger", "tune", "recipes", "workflows", 
  "vip", "magrittr", "tibble", "readr", "writexl", "readxl", "sf"
)

# Check, install missing, and load packages
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# Loading data #
setwd() # change to the directory that contains the needed files
data <- read.table("data_sbsr.txt", header = TRUE)

#_______________________________________________________________________________

##### 2: Correlations between variables #####

# calculating correlations between burned forest area and independent variables

# Defining dependent and independent variables
dependent_var <- "ba" # burned forest area column in data
excluded_vars <- c("area", "id", dependent_var) # excluding area and id columns
independent_vars <- setdiff(names(data), excluded_vars)

# Calculate correlations
correlations <- sapply(independent_vars, function(var) {
  cor(data[[var]], data[[dependent_var]], use = "complete.obs")
})

# Convert to absolute values and sort in descending order
sorted_vars <- names(sort(abs(correlations), decreasing = TRUE))

#_______________________________________________________________________________

##### 3: Running Random Forest Regression Models #####

# Creating function to run models adding variables based on correlation with
# the depedent variable (burned forest area)

run_rf_models <- function(data, dependent_var, var_list, n_iterations, n_bootstraps, grid_size, trees) {

  # Data frame to store results
  results <- data.frame(
    formula = character(),     # Model formula
    r2 = numeric(),            # Model accuracy - R-squared value
    high = numeric(),          # Burned forest area under high priority classification
    medium = numeric(),        # Burned forest area under medium priority classification
    low = numeric(),           # Burned forest area under low priority classification
    stringsAsFactors = FALSE
  )
  
  # Loop to run models adding variables at each iteration
  for (i in 1:n_iterations) {
    message(paste("Starting iteration", i, "of", n_iterations))
    
    # Creating bootstraps
    set.seed(123)
    boot_samples <- rsample::bootstraps(data, times = n_bootstraps)
    
    # Define model formula
    formula <- stats::as.formula(paste(dependent_var, "~", paste(var_list[1:i], collapse = " + ")))
    rf_recipe <- recipes::recipe(formula, data = data)
    
    # Random forest model specification
    rf_model <- parsnip::rand_forest(
      mtry = tune(),    # number of variables randomly selected at each split
      min_n = tune(),   # minimum number of data points in a node
      trees = trees     # number of trees in the forest
    ) %>%
      parsnip::set_engine("ranger") %>% # defines which RF implementation will be used 
      parsnip::set_mode("regression")   # defines regression mode
    
    # Workflow - linking the RF model specification to the formula 
    rf_wf <- workflows::workflow() %>%
      workflows::add_recipe(rf_recipe) %>%
      workflows::add_model(rf_model)
    
    # Setting control to monitor processing progress
    control <- tune::control_grid(verbose = TRUE)
    
    # Hyperparameter tuning
    set.seed(123)
    tuned_results <- rf_wf %>%
      tune::tune_grid(
        resamples = boot_samples,    # defines the data used to train the models
        grid = grid_size,            # number of combinations to be tried
        metrics = metric_set(rmse),  # defining root mean square error as metric
        control = control            # adding control for progress printing
      )
    
    # Best hyperparameters
    best_hp <- tuned_results %>%
      tune::select_best(metric = "rmse")
    
    # Final workflow and model
    final_rf <- rf_wf %>% tune::finalize_workflow(best_hp)
    final_model <- final_rf %>% parsnip::fit(data = data)
    
    # Defining final model R-squared 
    R2 <- final_model$fit$fit$fit$r.squared
    
    # Making predictions from the final model
    prediction <- stats::predict(final_model, new_data = data)
    
    # Creating data frame to compare predictions with real data
    pred_df <- data.frame(
      id = data$id,
      predicted = prediction$.pred,
      ba = data[[dependent_var]]
    )
    
    # Setting priority classification parameters
    percentiles <- quantile(pred_df$predicted, c(0.7, 0.90))
    
    # Classifying priority classes
    pred_df$priority <- ifelse(pred_df$predicted > percentiles[2], "High",
                               ifelse(pred_df$predicted > percentiles[1] & pred_df$predicted < percentiles[2], "Medium", "Low"))
    
    # Summarizing burned area and priority percentages
    merged_df <- pred_df %>%
      group_by(priority) %>%
      summarise(
        Burned_Area = sum(ba)
      ) %>%
      mutate(
        `%_BA` = Burned_Area / sum(Burned_Area) * 100
      )
    
    # Store results
    results <- rbind(results, data.frame(
      formula = paste(deparse(formula), collapse = " "),
      r2 = R2,
      high = merged_df$`%_BA`[merged_df$priority == "High"],
      medium = merged_df$`%_BA`[merged_df$priority == "Medium"],
      low = merged_df$`%_BA`[merged_df$priority == "Low"]
    ))
    
    # Progress message
    print(paste0("Iteration: ", i, " completed"))
  }
  
  return(results)
}

# Running models
results <- run_rf_models(data = data,
                         dependent_var = dependent_var,
                         var_list = sorted_vars,
                         n_iterations = length(sorted_vars),
                         n_bootstraps = 100,
                         grid_size = 5,
                         trees = 1000)

#_______________________________________________________________________________

##### 4: Running best selected model #####
# It is important to consider that the best selected model might not be the one
# with highest R-squared value or greater burned forest area under high priority
# classification. To determine that, make a decision considering this metrics but
# also model parsimony. In this case, the best selected model was the one with 
# six independent variables. For more information about this choice, read the 
# paper on the GitHub repository.

# Creating function to run the best model
run_best_rf_model <- function(data, dependent_var, n_var, n_bootstraps, grid_size, trees) {
  
  # Creating bootstraps
  set.seed(123)
  boot_samples <- rsample::bootstraps(data, times = 100)
  boot_samples
  
  # Defining model formula based on the formula on the formula column on results df
  rf_recipe <- recipes::recipe(as.formula(paste(results$formula[n_var])),
                               data = data)
  
  # Random forest model specification
  rf_model <- parsnip::rand_forest(
    mtry = tune(),    # number of variables randomly selected at each split
    min_n = tune(),   # minimum number of data points in a node
    trees = trees     # number of trees in the forest
  ) %>%
    parsnip::set_engine("ranger") %>% # defines which RF implementation will be used 
    parsnip::set_mode("regression")   # defines regression mode
  
  # Workflow - linking the RF model specification to the formula 
  rf_wf <- workflows::workflow() %>%
    workflows::add_recipe(rf_recipe) %>%
    workflows::add_model(rf_model)
  
  # Setting control to monitor processing progress
  control <- tune::control_grid(verbose = TRUE)
  
  # Hyperparameter tuning
  set.seed(123)
  tuned_results <- rf_wf %>%
    tune::tune_grid(
      resamples = boot_samples,    # defines the data used to train the models
      grid = grid_size,            # number of combinations to be tried
      metrics = metric_set(rmse),  # defining root mean square error as metric
      control = control            # adding control for progress printing
    )
  
  # Best hyperparameters
  best_hp <- tuned_results %>%
    tune::select_best(metric = "rmse")
  
  # Final workflow and model
  final_rf <- rf_wf %>% tune::finalize_workflow(best_hp)
  final_model <- final_rf %>% parsnip::fit(data = data)
  
  return(final_model)
}

# Running best model
best_model <- run_best_rf_model(data = data,
                  dependent_var = dependent_var,
                  n_var = 6,
                  n_bootstraps = 100,
                  grid_size = 5,
                  trees = 1000)

#_______________________________________________________________________________

##### 5: Plotting priority classification maps #####
# A priority classification map based on the models predictions will be plotted
# a reference priority classification map based on real burned forest area data
# will be plotted to compare the predictions quality

# Loading gridded shapefile - base map
grid <- vect("grid.shp")

# Creating function to add priority classification to the grid
classify_priority <- function(grid, model, data) {
  # Using best model to make predictions
  prediction <- stats::predict(model, new_data = data)

  # Creating data frame to compare predictions with real data
  pred_df <- data.frame(
    id = data$id,                    # column id
    predicted = prediction$.pred,    # extracted values from prediction
    ba = data$ba,                    # real burned area data
    area = data$area
  )

  # Setting priority classification parameters for the predictions
  percentiles <- quantile(pred_df$predicted, c(0.7, 0.90))

  # Classifying priority classes based on predictions
  pred_df$priority <- ifelse(pred_df$predicted > percentiles[2], "High",
                             ifelse(pred_df$predicted > percentiles[1] & pred_df$predicted < percentiles[2], "Medium", "Low"))

  # Setting priority classification parameters for real burned area data
  percentiles_ref <- quantile(pred_df$ba, c(0.7, 0.90))

  # Classifying priority classes based on real burned area data
  pred_df$priority_ref <- ifelse(pred_df$ba > percentiles_ref[2], "High",
                             ifelse(pred_df$ba > percentiles_ref[1] & pred_df$ba < percentiles_ref[2], "Medium", "Low"))

  # Adding classification columns to the grid object
  grid$priority <- pred_df$priority
  grid$priority_ref <- pred_df$priority_ref
  
  # Convert to sf (works best with ggplo2)
  grid <- st_as_sf(grid)
  return(grid)
}

# Classifying priority
grid <- classify_priority(grid = grid, model = best_model, data = data)


# Plotting priority maps
# Priority classification map based on the models predictions
priority_map_pred <- ggplot(data = grid) +
  geom_sf(aes(fill = priority)) +
  theme_minimal() +
  labs(title = "Priority classification map based on the models predictions", 
       fill = "Priority class") +
  theme(
    axis.text = element_blank(),
    axis.title = element_blank())

# View plot
priority_map_pred

# Reference priority classification map based on real burned forest area data
priority_map_ref <- ggplot(data = grid) +
  geom_sf(aes(fill = priority_ref)) +
  theme_minimal() +
  labs(title = "Reference priority classification map based on real burned forest area data", 
       fill = "Priority class") +
  theme(
    axis.text = element_blank(),
    axis.title = element_blank())

# View plot
priority_map_ref
