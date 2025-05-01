# suppressing warnings in the console
options(warn = -1)
# loading all relevant libraries and data
library(shiny)
library(bslib)
library(tidyverse)
library(vcdExtra)
library(rpart)
library(rpart.plot)
library(visdat)
library(caret)
library(yardstick)
library(randomForest)
library(sjPlot)
library(gt)
library(DT)
library(class)
library(e1071)
library(xgboost)
library(shiny)
library(patchwork)
library(pROC)
library(reactable)
library(reactablefmtr)
library(jsonlite)

data("Titanicp", package = "vcdExtra")

#----------Data Pre-processing--------------------------------------------------
# age is the only column with missing values
# removing rows with NA age values (only a small amount of data lost)
df = Titanicp |> filter(!is.na(age))

#splitting data into training and testing dataframes
set.seed(6)
n = nrow(df)
test_id = sample(1:n, round(n*0.30))
df_test = df[test_id,]
df_train = df[-test_id,]


# xgb-specific data
X_train <- model.matrix(survived ~ . - 1, data = df_train)
X_test <- model.matrix(survived ~ . - 1, data = df_test)
y_train = df_train$survived
y_test = df_test$survived

y_train = ifelse(y_train == "survived", 1, 0)
y_test = ifelse(y_test == "survived", 1, 0)

xgboost_train = xgb.DMatrix(data=X_train, label=y_train)
xgboost_test = xgb.DMatrix(data=X_test, label=y_test)



#----------Defining Feature Importance funcion-----------------------------------
feature_importance_plot_maker = function(rf_model){
  importance_matrix = importance(rf_model)
  importance_df = as.data.frame(importance_matrix)
  importance_df$Feature = rownames(importance_df)
  
  importance_df |> ggplot(aes(x = reorder(Feature, MeanDecreaseGini),
                              y = MeanDecreaseGini)) +
    geom_bar(stat = "identity", color = "black") +
    coord_flip() +
    labs(y = "Mean Decrease in Gini") +
    xlab(NULL) +
    ggtitle("Feature Importance Plot") +
    theme(
      plot.margin = unit(c(5.5, 14.175, 14.175, 14.175), "pt"),
      plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
      axis.title.x = element_text(size = 14),
      axis.title.y = element_text(size = 14),
      axis.text.x = element_text(size = 13),
      axis.text.y = element_text(size = 13),
    )
}



#----------Defining Accuracy line plot funcitn-----------------------------------
accuracy_plot_maker = function(xgb_model){
  acc_df = xgb_model$evaluation_log
  test_acc = 1-acc_df$eval_error
  train_acc = 1-acc_df$train_error
  
  metric_df = data.frame(
    Round = 1:length(train_acc),
    Train = train_acc,
    Test = test_acc
  )
  metric_df_long = metric_df |>
    pivot_longer(cols = c(Train, Test), names_to = "Dataset",
                 values_to = "Accuracy")
  plot = ggplot(metric_df_long, aes(x = Round, y = Accuracy, color = Dataset)) +
    geom_line(size = 1.2) +
    scale_x_continuous(
      breaks = pretty(1:length(train_acc), n = 10)
    ) +
    labs(
      title = "XGBoost Accuracy Progression",
      x = "Boosting Round",
      y = "Accuracy",
      color = "Dataset"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
      legend.position = "top",
      axis.text.y = element_text(size = 13),
      axis.text.x = element_text(size = 13),
      axis.title.x = element_text(size = 14),
      axis.title.y = element_text(size = 14),
      legend.text = element_text(size = 12),
      legend.title = element_text(size = 15)
    )
  
  plot
}



#----------Defining pca plot function-------------------------------------------
pca_plot_maker = function(model, test_x, test_y, model_type, preds){
  if(model_type == "svm"){
    predictions = predict(model, test_x)
  }
  else{
    predictions = preds
  }
  pcs = prcomp(test_x)
  pc1 = pcs$x[, 1]
  pc2 = pcs$x[, 2]
  
  pca_df = data.frame(
    observation = 1:length(test_y),
    pc1 = pc1,
    pc2 = pc2,
    actual = test_y,
    predicted = predictions
  )
  actual_plot = pca_df |>
    ggplot(aes(x= pc1, y = pc2, color = actual)) +
    geom_point() + labs(title = "Principal Component Analysis") + 
    theme(
      plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
      axis.text.y = element_text(size = 13),
      axis.text.x = element_text(size = 13),
      axis.title.x = element_text(size = 14),
      axis.title.y = element_text(size = 14),
      legend.text = element_text(size = 12),
      legend.title = element_text(size = 15)
    )
  
  predicted_plot = pca_df |>
    ggplot(aes(x= pc1, y = pc2, color = predicted)) +
    geom_point() +
    theme(
      axis.text.y = element_text(size = 13),
      axis.text.x = element_text(size = 13),
      axis.title.x = element_text(size = 14),
      axis.title.y = element_text(size = 14),
      legend.text = element_text(size = 12),
      legend.title = element_text(size = 15)
    )
  
  actual_plot/predicted_plot
}






#----------Defining lolipop function--------------------------------------------
lolipop_plot_maker = function(model, test_data, model_type,
                              threshold = 0.5, preds = NULL, xgbmatrix = NULL,
                              test_data_svm = NULL){
  
  if(model_type == "knn"){
    predictions_test = preds
  }else if (model_type == "xgb"){
    # predicting values using xgb matrix
    predictions_test = predict(model,
                               newdata = xgbmatrix, 
                               type = "response")
  }else if (model_type == "svm"){
    predictions_test = predict(model, newdata = test_data_svm)
    
  }
  else{
    # making and storing predictions
    predictions_test = predict(model,
                               newdata = select(test_data, -survived), 
                               type = model_type)
  }
  
  # formatting predictions according to model type
  if(model_type == "response" | model_type == "xgb"){
    pred_probs = predictions_test
    predictions_test = ifelse(pred_probs > threshold, "survived", "died")
    predictions_test = as.factor(predictions_test)
  }
  
  # creating a dataframe that compares predictions and actual values
  test_results_df = data.frame(
    actual = test_data$survived,
    predicted = predictions_test) |>
    mutate(actual = factor(actual, levels = c("survived", "died")),
           predicted = factor(predicted, levels = c("survived", "died")))
  
  # creating a confusion matrix
  cm_test = confusionMatrix(test_results_df$predicted,
                            test_results_df$actual)
  
  # calculating performance metrics
  accuracy = cm_test$overall["Accuracy"]
  error = 1-accuracy
  balanced_accuracy <- cm_test$byClass["Balanced Accuracy"]
  sensitivity <- cm_test$byClass["Sensitivity"]
  specificity <- cm_test$byClass["Specificity"]
  precision <- cm_test$byClass["Pos Pred Value"]
  f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)
  
  # creating a datframe of metrics
  metrics_df = data.frame(
    metric = c("Accuracy", "Error\nRate", 
               "Balanced\nAccuracy", "Sensitivity", 
               "Specificity", "Precision", 
               "F1\nScore"),
    value = c(accuracy, error, balanced_accuracy, sensitivity, 
              specificity, precision, f1_score)
  )
  ggplot(metrics_df, aes(x = metric, y = value)) +
    geom_segment(aes(x = metric, xend = metric, y = 0, yend = value), 
                 color = "skyblue") +
    geom_point(size = 6, color = "darkgreen") +
    ylim(0, 1) +
    labs(title = "Comparing performance metrics")+
    ylab("Value") +
    xlab(NULL) +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold")) + 
    theme(
      plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
      axis.text.y = element_text(size = 13),
      axis.text.x = element_text(size = 13, vjust = 0.5),
      axis.title.x = element_text(size = 14),
      axis.title.y = element_text(size = 14),
      legend.text = element_text(size = 13)
    )
}


#----------Defining metrics access function-------------------------------------
get_metrics = function(model, test_data, model_type,
                       threshold = 0.5, preds = NULL, xgbmatrix = NULL,
                       test_data_svm = NULL){
  
  if(model_type == "knn"){
    predictions_test = preds
  }else if (model_type == "xgb"){
    # predicting values using xgb matrix
    predictions_test = predict(model,
                               newdata = xgbmatrix, 
                               type = "response")
  }else if (model_type == "svm"){
    predictions_test = predict(model, newdata = test_data_svm)
    
  }
  else{
    # making and storing predictions
    predictions_test = predict(model,
                               newdata = select(test_data, -survived), 
                               type = model_type)
  }
  
  # formatting predictions according to model type
  if(model_type == "response" | model_type == "xgb"){
    pred_probs = predictions_test
    predictions_test = ifelse(pred_probs > threshold, "survived", "died")
    predictions_test = as.factor(predictions_test)
  }
  
  # creating a dataframe that compares predictions and actual values
  test_results_df = data.frame(
    actual = test_data$survived,
    predicted = predictions_test) |>
    mutate(actual = factor(actual, levels = c("survived", "died")),
           predicted = factor(predicted, levels = c("survived", "died")))
  
  # creating a confusion matrix
  cm_test = confusionMatrix(test_results_df$predicted,
                            test_results_df$actual)
  
  # calculating performance metrics
  accuracy = cm_test$overall["Accuracy"]
  error = 1-accuracy
  balanced_accuracy <- cm_test$byClass["Balanced Accuracy"]
  sensitivity <- cm_test$byClass["Sensitivity"]
  specificity <- cm_test$byClass["Specificity"]
  precision <- cm_test$byClass["Pos Pred Value"]
  f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)
  
  # creating a dataframe of metrics
  metrics_df = data.frame(
    metric = c("accuracy", "error_rate", 
               "balanced_accuracy", "sensitivity", 
               "specificity", "precision", 
               "f1_score"),
    value = c(accuracy, error, balanced_accuracy, sensitivity, 
              specificity, precision, f1_score)
  )
  metrics_df$value
}






#----------Defining the in-sample heat map--------------------------------------
in_sample_heatmap_maker = function(model, training_data, model_type,
                                   threshold = 0.5, preds = NULL,
                                   xgbmatrix = NULL,
                                   train_data_svm = NULL){
  
  
  if(model_type == "knn"){
    predictions_train = preds
  }else if (model_type == "xgb"){
    # predicting values using xgb matrix
    predictions_train = predict(model,
                                newdata = xgbmatrix, 
                                type = "response")
  }else if (model_type == "svm"){
    predictions_train = predict(model, newdata = train_data_svm)
    
  }
  else{
    # making and storing predictions
    predictions_train = predict(model, type = model_type)
  }
  
  
  # formatting predictions according to model type
  if(model_type == "response" | model_type == "xgb"){
    pred_probs = predictions_train
    predictions_train = ifelse(pred_probs > threshold, "survived", "died")
    predictions_train = as.factor(predictions_train)
  } 
  
  cm_table_train = table(predicted = predictions_train,
                         actual = training_data$survived)
  cm_df_train = as.data.frame(cm_table_train) |>
    mutate(actual = factor(actual, levels = c("survived", "died")))
  # calculating training model's accuracy
  train_acc = sum(diag(cm_table_train)) / sum(cm_table_train)
  
  
  ggplot(cm_df_train, aes(x = actual, y = predicted, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq),
              size = 5,
              color = ifelse(cm_df_train$Freq > median(cm_df_train$Freq),
                             "black", "white")) +
    scale_x_discrete(position = "top") +
    labs(x = "Actual", y = "Predicted", fill = "Count") +
    ggtitle(paste("In-Sample Performance: Accuracy =", 
                  round(train_acc, 3))) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
      axis.text.y = element_text(angle = 90, size = 13),
      axis.text.x = element_text(size = 13),
      axis.title.x = element_text(size = 14),
      axis.title.y = element_text(size = 14),
      legend.text = element_text(size = 12)
    )
}






#----------Defining the out-of-sample heat map----------------------------------
out_of_sample_heatmap_maker = function(model, test_data, model_type,
                                       threshold = 0.5, preds = NULL,
                                       xgbmatrix = NULL,
                                       test_data_svm = NULL){
  
  
  
  if(model_type == "knn"){
    predictions_test = preds
  }else if (model_type == "xgb"){
    # predicting values using xgb matrix
    predictions_test = predict(model,
                               newdata = xgbmatrix, 
                               type = "response")
  }else if (model_type == "svm"){
    predictions_test = predict(model, newdata = test_data_svm)
    
  }
  else{
    # making and storing predictions
    predictions_test = predict(model,
                               newdata = select(test_data, -survived), 
                               type = model_type)
  }
  
  # formatting predictions according to model type
  if(model_type == "response" | model_type == "xgb"){
    pred_probs = predictions_test
    predictions_test = ifelse(pred_probs > threshold, "survived", "died")
    predictions_test = as.factor(predictions_test)
  }
  
  cm_table_test = table(actual = test_data$survived, 
                        predicted = predictions_test)
  cm_df_test = as.data.frame(cm_table_test) |>
    mutate(actual = factor(actual, levels = c("survived", "died")))
  
  # calculating test model's accuracy
  test_acc = sum(diag(cm_table_test)) / sum(cm_table_test)
  
  # plotting a confusion matrix heat map
  ggplot(cm_df_test, aes(x = actual, y = predicted, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq),
              size = 5,
              color = ifelse(cm_df_test$Freq > median(cm_df_test$Freq),
                             "black", "white")) +
    scale_x_discrete(position = "top") +
    labs(x = "Actual", y = "Predicted", fill = "Count") +
    ggtitle(paste("Out-of-Sample Performance: Accuracy =",
                  round(test_acc, 3))) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
      axis.text.y = element_text(angle = 90, size = 13),
      axis.text.x = element_text(size = 13),
      axis.title.x = element_text(size = 14),
      axis.title.y = element_text(size = 14),
      legend.text = element_text(size = 12)
    )
}

#-------------------------------------------------------------------------------
get_custom_styles <- function(df, best_idxs) {
  styles <- vector("list", nrow(df))
  for (i in seq_len(nrow(df))) {
    row_styles <- rep(NA, ncol(df))
    row_styles[best_idxs[i] + 1] <- "background-color: #b7e4c7; font-weight: bold;"  # +1 for offset (skip 'metric')
    styles[[i]] <- row_styles
  }
  names(styles) <- NULL
  return(styles)
}
#-------------------------------------------------------------------------------

introText = HTML(
  "<div style='font-size:24px;'>
  <b>Welcome to the Classification Playground!</b></br></br>
  </div>
  
  <div style='font-size:18px;'>
  Experiment with machine learning classification models using the famous Titanic Dataset.</br></br>
  
  \u2794 Explore various classification models</br>
  \u2794 Select predictors and tweak hyperparameters</br>
  \u2794 Observe model output plots and performance metrics</br>
  \u2794 Compare results across models</br></br>
  <u>Models available</u>: Logistic Regression, Decision Tree, Random Forest, k-NN, SVM, XGBoost</br></br>

  <u>Training/Test Split</u>: 70% training and 30% testing</br></br>
  </div>"
)


#----------Defining the User Interface------------------------------------------
ui = page_navbar(
  id = "my_navbar",
  title = "Classification Playground",
  nav_panel(
    title = "Home",
    fluidRow(
      column(
        width = 7,
        card(
          style = "height: 450px;border: 2px solid #d0eef7;",
          introText
        )
      ),
      column(
        width = 5,
        card(
          style = "height: 450px;border: 2px solid #d0eef7;",
          HTML(
            "<div style='font-size:24px;'>
            <b>Data Dictionary</b>
            </div>"
          ),
          DT::dataTableOutput("data_dictionary")
        )
      )
    ),
    fluidRow(
      column(
        width = 12,
        card(style = "height: 300px;border: 2px solid #d0eef7;",
             HTML(
               "<div style='font-size:24px;'>
                <b>Models</b>
                </div>"
             ),
             fluidRow(
               column(4, actionButton("go_dtree", "Decision Tree", style = "width: 100%;height: 70px;background-color: #d0eef7")),
               column(4, actionButton("go_rf", "Random Forest", style = "width: 100%;height: 70px;background-color: #d0eef7")),
               column(4, actionButton("go_logistic", "Logistic Regression", style = "width: 100%;height: 70px;background-color: #d0eef7"))
             ),
             fluidRow(
               column(4, actionButton("go_knn", "K-Nearest Neighbours", style = "width: 100%;height: 70px;background-color: #d0eef7")),
               column(4, actionButton("go_svm", "Support Vector Machine", style = "width: 100%;height: 70px;background-color: #d0eef7")),
               column(4, actionButton("go_xgb", "XGBoost", style = "width: 100%;height: 70px;background-color: #d0eef7"))
             )
        )
      )
    )
  ),
  nav_menu(
    title = "Models",
    nav_panel("Decision Tree",
              page_sidebar(
                sidebar = sidebar(
                  h4("Decision Tree"),
                  bslib::tooltip(
                    card(
                      checkboxGroupInput(
                        "predictors_tree",
                        "Predictors",
                        choices = list("Passenger Class" = "pclass",
                                       "Sex" = "sex",
                                       "Siblings/Spouses" = "sibsp",
                                       "Parents/Children" = "parch",
                                       "Age" = "age"),
                        selected = c("pclass", "sex")
                      )
                    ), "Select at least one predictor (input variable)",
                    id = "dt_predictors_tooltip", 
                    placement = "right"),
                  
                  bslib::tooltip(
                    card(
                      numericInput(
                        "cp",
                        "Complexity Parameter",
                        value = 0.01
                      )
                    ), "The complexity parameter sets the minimum improvement
                    needed for a split to be allowed. Higher values 
                    result in simpler trees. Smaller values result in more
                    complex trees. The complexity parameter cannot be negative.",
                    id = "cp_tooltip",
                    placement = "right")
                ),
                fluidRow(
                  column(6,
                         card(
                           plotOutput("tree_plot")
                         ),
                         card(
                           plotOutput("lolipop_tree")
                         )
                  ),
                  column(6,
                         card(
                           plotOutput("in_sample_cm_tree")
                         ),
                         card(
                           plotOutput("out_of_sample_cm_tree")
                         )
                  )
                )
              )
              
    ),
    nav_panel("Random Forest",
              page_sidebar(
                sidebar = sidebar(
                  h4("Random Forest"),
                  bslib::tooltip(
                    card(
                      sliderInput("node_size",
                                  "Minimum size of leaf nodes in each tree",
                                  min = 1,
                                  max = 30,
                                  value = 1
                      )
                    ), "Minimum number of samples required at a leaf node. 
                    Smaller values allow the tree to grow deeper, while larger
                    values help prevent overfitting.",
                    id = "rf_min_leaf_size_tooltip",
                    placement = "right"),
                  bslib::tooltip(
                    card(
                      sliderInput("num_vars",
                                  "Number of variables sampled at each split",
                                  min = 1,
                                  max = 5,
                                  value = round(sqrt(5))
                      )
                    ), "Number of predictors randomly chosen at each split. 
                    Lower values encourage more diverse trees and reduce 
                    overfitting.",
                    id = "rf_num_preds_sampled_tooltip",
                    placement = "right")
                ),
                
                fluidRow(
                  column(6,
                         card(
                           plotOutput("feature_importance_plot_rf")
                         ),
                         card(
                           plotOutput("lolipop_rf")
                         )
                  ),
                  column(6,
                         card(
                           plotOutput("in_sample_cm_rf")
                         ),
                         card(
                           plotOutput("out_of_sample_cm_rf")
                         )
                  )
                )
              )
    ),
    nav_panel("Logistic Regression",
              page_sidebar(
                sidebar = sidebar(
                  h4("Logistic Regression"),
                  bslib::tooltip(
                    card(
                      checkboxGroupInput(
                        "predictors_lr",
                        "Predictors",
                        choices = list("Passenger Class" = "pclass",
                                       "Sex" = "sex",
                                       "Siblings/Spouses" = "sibsp",
                                       "Parents/Children" = "parch",
                                       "Age" = "age"),
                        selected = c("pclass", "sex")
                      )
                    ), "Select at least one predictor (input variable)",
                    id = "lr_predictors_tooltip",
                    placement = "right"),
                  bslib::tooltip(
                    card(
                      numericInput(
                        "threshold",
                        "Threshold for survival",
                        value = 0.5
                      )
                    ),"Probability threshold for predicting survival. 
                    If a passenger’s predicted survival probability is above 
                    this value, they are classified as 'survived'; otherwise, 
                    'died'.",
                    id = "lr_threshold_tooltip",
                    placement = "right")
                ),
                fluidRow(
                  column(6,
                         card(
                           DTOutput("coef_table_lr")
                         ),
                         card(
                           plotOutput("lolipop_lr")
                         )
                  ),
                  column(6,
                         card(
                           plotOutput("in_sample_cm_lr")
                         ),
                         card(
                           plotOutput("out_of_sample_cm_lr")
                         )
                  )
                )
              )
              
    ),
    nav_panel("K-Nearest Neighbours",
              page_sidebar(
                sidebar = sidebar(
                  h4("K-Nearest Neighbours"),
                  bslib::tooltip(
                    card(
                      checkboxGroupInput(
                        "predictors_knn",
                        "Predictors",
                        choices = list("Passenger Class" = "pclass",
                                       "Sex" = "sex",
                                       "Siblings/Spouses" = "sibsp",
                                       "Parents/Children" = "parch",
                                       "Age" = "age"),
                        selected = c("pclass", "sex")
                      )
                    ), "Select at least one predictor (input variable)",
                    id = "knn_predictors_tooltip",
                    placement = "right"),
                  bslib::tooltip(
                    card(
                      sliderInput("k",
                                  "k (number of nearest neighbours considered)",
                                  min = 1,
                                  max = 100,
                                  value = 3
                      )
                    ), "Number of nearest neighbors used to classify a data
                    point. Smaller values make decisions based on fewer, closer
                    examples; larger values smooth out predictions using more
                    neighbors.",
                    id = "knn_k_tooltip",
                    placement = "right")
                ),
                fluidRow(
                  column(6,
                         card(
                           plotOutput("pca_knn")
                         ),
                         card(
                           plotOutput("lolipop_knn")
                         )
                  ),
                  column(6,
                         card(
                           plotOutput("in_sample_cm_knn")
                         ),
                         card(
                           plotOutput("out_of_sample_cm_knn")
                         )
                  )
                )
              )
    ),
    nav_panel("Support Vector Machine",
              page_sidebar(
                sidebar = sidebar(
                  h4("Support Vector Machine"),
                  bslib::tooltip(
                    card(
                      checkboxGroupInput(
                        "predictors_svm",
                        "Predictors",
                        choices = list("Passenger Class" = "pclass",
                                       "Sex" = "sex",
                                       "Siblings/Spouses" = "sibsp",
                                       "Parents/Children" = "parch",
                                       "Age" = "age"),
                        selected = c("pclass", "sex")
                      )
                    ), "Select at least one predictor (input variable)",
                    id = "svm_predictors_tooltip"),
                  bslib::tooltip(
                    card(
                      radioButtons(
                        "kernel",
                        "Kernel",
                        choices = list("Radial" = "radial",
                                       "Linear" = "linear",
                                       "Polynomial" = "polynomial",
                                       "Sigmoid" = "sigmoid"),
                        selected = "radial"
                      )
                    ), "Specifies the function used to map data into higher 
                    dimensions. Different kernels capture different types 
                    of patterns.",
                    id = "svm_kernel_tooltip",
                    placement = "right")
                ),
                fluidRow(
                  column(6,
                         card(
                           plotOutput("pca_svm")
                         ),
                         card(
                           plotOutput("lolipop_svm")
                         )
                  ),
                  column(6,
                         card(
                           plotOutput("in_sample_cm_svm")
                         ),
                         card(
                           plotOutput("out_of_sample_cm_svm")
                         )
                  )
                )
              )
              
    ),
    nav_panel("XGBoost",
              page_sidebar(
                sidebar = sidebar(
                  h4("Extreme Gradient Boosting"),
                  bslib::tooltip(
                    card(
                      sliderInput("nrounds",
                                  "Number of models in the ensemble",
                                  min = 1,
                                  max = 100,
                                  value = 10
                      )
                    ), "Number of boosting rounds (trees) to train. Each round 
                    adds a new tree to correct previous errors. More rounds can
                    improve learning, but too many may cause overfitting.",
                    id = "num_rounds_tooltip",
                    placement = "right"),
                  bslib::tooltip(
                    card(
                      sliderInput("eta",
                                  "Learning Rate",
                                  min = 0.01,
                                  max = 0.5,
                                  value = 0.3
                      )
                    ),"This is the learning rate. It scales the contribution of 
                    each new tree added during training, helping to prevent the
                    model from overfitting. Smaller values make learning slower 
                    but more robust, often requiring more boosting rounds 
                    (nrounds)",
                    id = "eta_tooltip",
                    placement = "right"),
                  bslib::tooltip(
                    card(
                      sliderInput("max_depth",
                                  "Maximum depth of the tree",
                                  min = 1,
                                  max = 15,
                                  value = 6
                      )
                    ), "This sets the maximum depth of each decision tree. 
                    Deeper trees can capture more complex patterns in the data, 
                    but they also increase the risk of overfitting",
                    id = "max_depth_xgb_tooltip",
                    placement = "right")
                ),
                fluidRow(
                  column(6,
                         card(
                           plotOutput("accuracy_plot_xgb")
                         ),
                         card(
                           plotOutput("lolipop_xgb")
                         )
                  ),
                  column(6,
                         card(
                           plotOutput("in_sample_cm_xgb")
                         ),
                         card(
                           plotOutput("out_of_sample_cm_xgb")
                         )
                  )
                )
              )
              
    )
  ),
  nav_panel(
    title = "Comparing Models",
    fluidRow(
      column(
        width = 5,
        card(
          style = "height: 640px;",
          plotOutput("roc_plot")
        )
      ),
      column(
        width = 7,
        card(
          style = "height: 640px;",
          HTML(
            "<div style='font-size:24px;text-align:center;'>
            <b>Out-of-Sample Performance Metrics</b>
            </div>"
          ),
          HTML(
            "<div style='font-size:18px; text-align:center; position:relative;'>
             Green highlights indicate the top-performing model for each metric.
             </div>"
          ),
          reactableOutput("comparison_table")
        )
      )
    )
  )
)

#----------Defining the server's logic-----------------------------------------
server = function(input, output, session){
  
  observeEvent(input$go_dtree, {
    updateNavbarPage(session, "my_navbar", selected = "Decision Tree")
  })
  
  observeEvent(input$go_rf, {
    updateNavbarPage(session, "my_navbar", selected = "Random Forest")
  })
  
  observeEvent(input$go_logistic, {
    updateNavbarPage(session, "my_navbar", selected = "Logistic Regression")
  })
  
  observeEvent(input$go_knn, {
    updateNavbarPage(session, "my_navbar", selected = "K-Nearest Neighbours")
  })
  
  observeEvent(input$go_svm, {
    updateNavbarPage(session, "my_navbar", selected = "Support Vector Machine")
  })
  
  observeEvent(input$go_xgb, {
    updateNavbarPage(session, "my_navbar", selected = "XGBoost")
  })
  
  
  # creating a data dictionary table
  output$data_dictionary = DT::renderDataTable({
    data.frame(
      Variable = c("survived", "sex", "age", "pclass", "sibsp", "parch"),
      Type = c("Categorical",
               "Categorical",
               "Numeric",
               "Categorical",
               "Numeric",
               "Numeric"),
      Description = c("died, survived",
                      "female, male",
                      "age in years (or fractions of a year, for children)",
                      "1st, 2nd, 3rd",
                      "number of siblings aboard",
                      "number of parents or children aboard")
    )
  },
  options = list(
    dom = 't', 
    ordering = FALSE,  
    autoWidth = TRUE,
    pageLength = 6
  ))
  
  
  # building a reactive tree
  tree = reactive({
    # defining the formula based on reactive input
    tree_formula = as.formula(
      paste("survived", "~", paste(input$predictors_tree, collapse = " + "))
    )
    
    # fitting a decision tree model
    rpart(tree_formula,
          data = df_train,
          method = "class",
          control = rpart.control(cp = input$cp))
  })
  
  
  
  # building a reactive random forest
  rf = reactive({
    set.seed(6)
    randomForest(survived ~.,
                 data = df_train,
                 nodesize = input$node_size,
                 mtry = input$num_vars)
  })
  
  
  # building a reactive logistic regression model
  lr = reactive({
    # defining the formula based on reactive input
    lr_formula = as.formula(
      paste("survived", "~", paste(input$predictors_lr, collapse = " + "))
    )
    
    glm(lr_formula,
        family = "binomial",
        data = df_train)
  })
  
  
  # building a reactive xgb model
  xgb = reactive({
    xgb.train(
      params = list(
        booster = "gbtree",
        objective = "binary:logistic",
        eval_metric = "error",
        eta = input$eta,
        max_depth = input$max_depth
      ),
      data = xgboost_train,
      nrounds = input$nrounds,
      watchlist = list(train = xgboost_train,
                       eval = xgboost_test),
      verbose = 0
    )
  })
  
  # building a reactive svm
  svm = reactive({
    svm_formula  = as.formula(
      paste("survived", "~", paste(input$predictors_svm, collapse = " + "), 
            " - 1")
    )
    
    X_all = model.matrix(svm_formula, data = df)
    X_train_svm = as.data.frame(X_all[-test_id,])
    y_train_svm = df_train$survived
    
    svm_model = e1071::svm(x = X_train_svm, y = y_train_svm,
                           kernel = input$kernel,
                           probability = TRUE)
  })
  
  df_test_svm = reactive({
    svm_formula  = as.formula(
      paste("survived", "~", paste(input$predictors_svm, collapse = " + "), 
            " - 1")
    )
    X_all = model.matrix(svm_formula, data = df)
    as.data.frame(X_all[test_id, ])
  })
  
  
  
  # running the kNN algorithm and generating predictions (in-sample)
  knn_in_sample = reactive({
    # defining X training data
    df_train_X = df_train |>
      mutate(
        pclass = as.numeric(factor(pclass, levels = c("1st", "2nd", "3rd"))),
        sex = as.numeric(factor(sex, levels = c("male", "female")))
      ) |>
      select(input$predictors_knn) 
    
    df_train_X = df_train_X |>
      scale() |>
      as.data.frame()
    
    # defining y training data
    train_y = df_train$survived
    
    # defining knn algorithm based on reactive input
    # knn(train = df_train_X, test = df_train_X,
    #   cl = train_y, k = input$k)
    knn_model = caret::knn3(x = df_train_X, y = as.factor(train_y), k = input$k)
    predict(knn_model, newdata = df_train_X, type = "class")
  })
  
  
  
  
  
  # running the kNN algorithm and generating predictions (out-of-sample)
  knn_out_of_sample = reactive({
    # defining X training data
    df_train_X = df_train |>
      mutate(
        pclass = as.numeric(factor(pclass, levels = c("1st", "2nd", "3rd"))),
        sex = as.numeric(factor(sex, levels = c("male", "female")))
      ) |>
      select(input$predictors_knn)
    
    
    df_train_X = df_train_X |>
      scale() |>
      as.data.frame()
    
    # defining y training data
    train_y  = df_train$survived
    
    # defining X testing data
    df_test_X = df_test |>
      mutate(
        pclass = as.numeric(factor(pclass, levels = c("1st", "2nd", "3rd"))),
        sex = as.numeric(factor(sex, levels = c("male", "female")))
      ) |>
      select(input$predictors_knn)
    
    
    df_test_X = df_test_X |>
      scale() |>
      as.data.frame()
    
    
    # defining knn algorithm based on reactive input
    # knn(train = df_train_X, test = df_test_X,
    #     cl = train_y, k = input$k)
    knn_model = caret::knn3(x = df_train_X, y = as.factor(train_y), k = input$k)
    predict(knn_model, newdata = df_test_X, type = "class")
  })
  
  
  knn3_probs = reactive({
    # defining X training data
    df_train_X = df_train |>
      mutate(
        pclass = as.numeric(factor(pclass, levels = c("1st", "2nd", "3rd"))),
        sex = as.numeric(factor(sex, levels = c("male", "female")))
      ) |>
      select(input$predictors_knn)
    
    
    df_train_X = df_train_X |>
      scale() |>
      as.data.frame()
    
    # defining y training data
    train_y  = df_train$survived
    
    # defining X testing data
    df_test_X = df_test |>
      mutate(
        pclass = as.numeric(factor(pclass, levels = c("1st", "2nd", "3rd"))),
        sex = as.numeric(factor(sex, levels = c("male", "female")))
      ) |>
      select(input$predictors_knn)
    
    
    df_test_X = df_test_X |>
      scale() |>
      as.data.frame()
    
    knn_model = caret::knn3(x = df_train_X, y = as.factor(train_y), k = input$k)
    predict(knn_model, newdata = df_test_X)[, 1]
  })
  
  test_labels <- df_test$survived
  
  # calculating decision tree probabilites
  tree_probs <- reactive({
    predict(tree(), newdata = df_test, type = "prob")[, 2]
  })
  
  # calculating random forest probabilites
  rf_probs = reactive({
    predict(rf(), newdata = df_test, type = "prob")[, 2]
  })
  
  # calculating logistic regression probailites
  lr_probs = reactive({
    predict(lr(), newdata = df_test, type = "response")
  })
  
  # calculating knn probabilities
  # done above
  
  # calculating svm probabilities
  svm_probs = reactive({
    preds_temp = predict(svm(), newdata = df_test_svm(), probability = TRUE)
    attr(preds_temp, "probabilities")[,2]
  })
  
  xgb_probs <- reactive({
    predict(xgb(), newdata = xgboost_test)
  })
  
  roc_list = reactive({
    roc_values <- list(
      "Decision Tree" = suppressMessages(roc(test_labels, tree_probs())),
      "Random Forest" = suppressMessages(roc(test_labels, rf_probs())),
      "Logistic Regression" = suppressMessages(roc(test_labels, lr_probs())),
      "k-NN" = suppressMessages(roc(test_labels, knn3_probs())),
      "SVM" = suppressMessages(roc(test_labels, svm_probs())),
      "XGBoost" = suppressMessages(roc(test_labels, xgb_probs()))
    )
  })
  
  output$roc_plot = renderPlot({
    # Convert ROC data to a combined data frame
    roc_df <- do.call(rbind, lapply(names(roc_list()), function(name) {
      r <- roc_list()[[name]]
      data.frame(
        FPR = rev(1 - r$specificities),
        TPR = rev(r$sensitivities),
        Model = name
      )
    }))
    
    ggplot(roc_df, aes(x = FPR, y = TPR, color = Model)) +
      geom_line(linewidth = 1.2) +
      geom_abline(linetype = "dashed", color = "gray") +
      labs(
        title = "ROC Curves - Model Comparison",
        x = "False Positive Rate (FPR)",
        y = "True Positive Rate (TPR)"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 24), 
        axis.title.x = element_text(size = 14, vjust = -1),
        axis.title.y = element_text(size = 14, vjust = 3),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),
        legend.text = element_text(size = 14),  
        legend.position = "bottom"
      )
  })
  
  # visualizing the decision tree
  output$tree_plot = renderPlot({
    par(mar = c(4, 4, 2, 2))
    rpart.plot(tree(), extra = 0, main = "Decision Rules in the Tree",
               cex.main = 1.6)
  })
  
  
  
  # visualizing lolipop plot for decision tree
  output$lolipop_tree = renderPlot({
    lolipop_plot = lolipop_plot_maker(tree(), df_test, "class")
    lolipop_plot
  })
  
  
  
  # visualizing in-sample confusion matrix for decision tree
  output$in_sample_cm_tree = renderPlot({
    in_sample_heatmap = in_sample_heatmap_maker(tree(), df_train, "class")
    in_sample_heatmap
  })
  
  
  
  # visualizing out-of-sample confusion matrix for decision tree
  output$out_of_sample_cm_tree = renderPlot({
    out_of_sample_heatmap = out_of_sample_heatmap_maker(tree(),
                                                        df_test, "class")
    out_of_sample_heatmap
  })
  
  
  
  # visualizing feature importance plot for rf
  output$feature_importance_plot_rf = renderPlot({
    feature_importance_plot = feature_importance_plot_maker(rf())
    feature_importance_plot
  })
  
  
  
  # visualizing lolipop plot for rf
  output$lolipop_rf = renderPlot({
    lolipop_plot = lolipop_plot_maker(rf(), df_test, "class")
    lolipop_plot
  })
  
  
  
  # visualizing in-sample confusion matrix for rf
  output$in_sample_cm_rf = renderPlot({
    in_sample_heatmap = in_sample_heatmap_maker(rf(), df_train, "class")
    in_sample_heatmap
  })
  
  
  
  # visualizing out-of-sample confusion matrix for rf
  output$out_of_sample_cm_rf = renderPlot({
    out_of_sample_heatmap = out_of_sample_heatmap_maker(rf(), df_test, "class")
    out_of_sample_heatmap
  })
  
  
  # visualizing in-sample confusion matrix for lr
  output$in_sample_cm_lr = renderPlot({
    in_sample_heatmap = in_sample_heatmap_maker(lr(), df_train, "response",
                                                threshold = input$threshold)
    in_sample_heatmap
  })
  
  
  # visualizing out-of-sample confusion matrix for rf
  output$out_of_sample_cm_lr = renderPlot({
    out_of_sample_heatmap = out_of_sample_heatmap_maker(lr(),
                                                        df_test, 
                                                        "response",
                                                        threshold = input$threshold)
    out_of_sample_heatmap
  })
  
  
  # displaying logistic regression coefficients
  output$coef_table_lr = renderDT({
    
    df_coefficients_lr = as.data.frame(summary(lr())$coefficients)
    colnames(df_coefficients_lr) = c("Coefficient_Estimate", "stderr",
                                     "zscore", "p.value")
    df_coefficients_lr = df_coefficients_lr |>
      select(Coefficient_Estimate, p.value) |>
      mutate(Coefficient_Estimate = round(Coefficient_Estimate, 2),
             p.value = round(p.value, 2)) |>
      rownames_to_column("Term")
    df_coefficients_lr
  })
  
  
  # visualizing lolipop plot for lr
  output$lolipop_lr = renderPlot({
    lolipop_plot = lolipop_plot_maker(lr(),
                                      df_test,
                                      "response",
                                      threshold = input$threshold)
    lolipop_plot
  })
  
  
  # visualizing in-sample confusion matrix for knn
  output$in_sample_cm_knn = renderPlot({
    in_sample_heatmap = in_sample_heatmap_maker(NULL, df_train, "knn",
                                                preds = knn_in_sample())
    in_sample_heatmap
  })
  
  
  # visualizing in-sample confusion matrix for xgb
  output$in_sample_cm_xgb = renderPlot({
    in_sample_heatmap = in_sample_heatmap_maker(xgb(), df_train, "xgb",
                                                threshold = 0.5,
                                                xgbmatrix = xgboost_train)
    in_sample_heatmap
  })
  
  # visualizing out-of-sample confusion matrix for knn
  output$out_of_sample_cm_knn = renderPlot({
    out_of_sample_heatmap = out_of_sample_heatmap_maker(NULL, df_test, "knn",
                                                        preds = knn_out_of_sample())
    out_of_sample_heatmap
  })
  
  
  # visualizing out-of-sampmle confusion matrix for xgb
  output$out_of_sample_cm_xgb = renderPlot({
    out_of_sample_heatmap = out_of_sample_heatmap_maker(xgb(), df_test, "xgb",
                                                        threshold = 0.5,
                                                        xgbmatrix = xgboost_test)
    out_of_sample_heatmap
  })
  
  
  
  # visualizing lolipop plot for knn
  output$lolipop_knn = renderPlot({
    lolipop_plot = lolipop_plot_maker(NULL, df_test, "knn",
                                      preds = knn_out_of_sample())
    lolipop_plot
  })
  
  # visualizing lolipop plot for xgb
  output$lolipop_xgb = renderPlot({
    lolipop_plot = lolipop_plot_maker(xgb(), df_test, "xgb",
                                      threshold = 0.5,
                                      xgbmatrix = xgboost_test)
    lolipop_plot
  })
  
  
  # visualizing lolipop plot for svm
  output$lolipop_svm = renderPlot({
    
    svm_formula  = as.formula(
      paste("survived", "~", paste(input$predictors_svm, collapse = " + "), 
            " - 1")
    )
    X_all = model.matrix(svm_formula, data = df)
    X_test_svm = as.data.frame(X_all[test_id,])
    
    
    lolipop_plot = lolipop_plot_maker(svm(), df_test, "svm",
                                      test_data_svm = X_test_svm)
    lolipop_plot
  })
  
  
  # visualizing out of sample confusion matrix for svm
  output$out_of_sample_cm_svm = renderPlot({
    
    svm_formula  = as.formula(
      paste("survived", "~", paste(input$predictors_svm, collapse = " + "), 
            " - 1")
    )
    X_all = model.matrix(svm_formula, data = df)
    X_test_svm = as.data.frame(X_all[test_id,])
    
    out_of_sample_heatmap = out_of_sample_heatmap_maker(svm(), df_test, "svm",
                                                        test_data_svm = X_test_svm)
    out_of_sample_heatmap
  })
  
  
  
  # visualizing in-sample confusion matrix for svm
  output$in_sample_cm_svm = renderPlot({
    
    svm_formula  = as.formula(
      paste("survived", "~", paste(input$predictors_svm, collapse = " + "), 
            " - 1")
    )
    
    X_all = model.matrix(svm_formula, data = df)
    X_train_svm = as.data.frame(X_all[-test_id,])
    
    in_sample_heatmap = in_sample_heatmap_maker(svm(), df_train, "svm",
                                                train_data_svm = X_train_svm)
    in_sample_heatmap
  })
  
  
  # visualizing pca plot for svm
  output$pca_svm = renderPlot({
    svm_formula  = as.formula(
      paste("survived", "~", paste(input$predictors_svm, collapse = " + "), 
            " - 1")
    )
    
    X_all = model.matrix(svm_formula, data = df)
    X_test_svm = as.data.frame(X_all[test_id,])
    y_test_svm = df_test$survived
    
    pca_svm_plot = pca_plot_maker(svm(), X_test_svm, y_test_svm, "svm")
    pca_svm_plot
  })
  
  # visualizing pca plot for knn
  output$pca_knn = renderPlot({
    knn_formula  = as.formula(
      paste("survived", "~", paste(input$predictors_knn, collapse = " + "), 
            " - 1")
    )
    
    X_all = model.matrix(knn_formula, data = df)
    X_test_knn = as.data.frame(X_all[test_id,])
    y_test_knn = df_test$survived
    
    pca_knn_plot = pca_plot_maker(knn_out_of_sample(), X_test_knn,
                                  y_test_knn, "knn",
                                  preds = knn_out_of_sample())
    pca_knn_plot
  })
  
  
  output$accuracy_plot_xgb = renderPlot({
    acc_plot = accuracy_plot_maker(xgb())
    acc_plot
  })
  
  
  # creating a comparison table
  output$comparison_table = renderReactable({
    # ordering -> acc, err, balacc, sens, spec, prec, f1
    dt_metrics = get_metrics(tree(), df_test, "class")
    rf_metrics = get_metrics(rf(), df_test, "class")
    lr_metrics = get_metrics(lr(),
                             df_test,
                             "response",
                             threshold = input$threshold)
    knn_metrics = get_metrics(NULL, df_test, "knn",
                              preds = knn_out_of_sample())
    # svm-specific code
    svm_formula  = as.formula(
      paste("survived", "~", paste(input$predictors_svm, collapse = " + "), 
            " - 1")
    )
    X_all = model.matrix(svm_formula, data = df)
    X_test_svm = as.data.frame(X_all[test_id,])
    
    svm_metrics = get_metrics(svm(), df_test, "svm",
                              test_data_svm = X_test_svm)
    # xgb metrics
    xgb_metrics = get_metrics(xgb(), df_test, "xgb",
                              threshold = 0.5,
                              xgbmatrix = xgboost_test)
    
    # making a dataframe of all metrics
    all_df = data.frame(
      metric = c("Accuracy", "Error Rate", "Balanced Accuracy",
                 "Sensitivity", "Specificity", "Precision", "F1 Score"),
      dt = dt_metrics,
      rf = rf_metrics,
      lr = lr_metrics,
      knn = knn_metrics,
      svm = svm_metrics,
      xgb = xgb_metrics
    )
    
    # adding auc values as well
    auc_values = unname(sapply(roc_list(), auc))
    
    auc_row = data.frame(
      metric = "AUC",
      dt = auc_values[1],
      rf = auc_values[2],
      lr = auc_values[3],
      knn = auc_values[4],
      svm = auc_values[5],
      xgb = auc_values[6]
    )
    
    all_df = rbind(all_df, auc_row)
    
    best_acc_idx = which.max(
      as.numeric(all_df[all_df$metric == "Accuracy", -1])
    )
    
    best_err_idx = which.min(
      as.numeric(all_df[all_df$metric == "Error Rate", -1])
    )
    
    best_bal_idx = which.max(
      as.numeric(all_df[all_df$metric == "Balanced Accuracy", -1])
    )
    
    best_sen_idx = which.max(
      as.numeric(all_df[all_df$metric == "Sensitivity", -1])
    )
    
    best_spec_idx = which.max(
      as.numeric(all_df[all_df$metric == "Specificity", -1])
    )
    
    best_prec_idx = which.max(
      as.numeric(all_df[all_df$metric == "Precision", -1])
    )
    
    best_f1_idx = which.max(
      as.numeric(all_df[all_df$metric == "F1 Score", -1])
    )
    
    best_auc_idx = which.max(
      as.numeric(all_df[all_df$metric == "AUC", -1])
    )
    
    best_idxs = c(best_acc_idx,
                  best_err_idx,
                  best_bal_idx,
                  best_sen_idx,
                  best_spec_idx,
                  best_prec_idx,
                  best_f1_idx,
                  best_auc_idx)
    
    reactable(
      all_df,
      columns = list(
        metric = colDef(name = "Metric", minWidth = 120),
        dt = colDef(
          name = "Decision Tree",
          format = colFormat(digits = 2),
          style = function(value, index) {
            if (best_idxs[index] == 1) {
              list(background = "#90EE90")
            } else {
              list(background = "white")
            }
          }
        ),
        rf = colDef(
          name = "Random Forest",
          format = colFormat(digits = 2),
          style = function(value, index) {
            if (best_idxs[index] == 2) {
              list(background = "#90EE90")
            } else {
              list(background = "white")
            }
          }
        ),
        lr = colDef(
          name = "Logistic Regression",
          format = colFormat(digits = 2),
          style = function(value, index) {
            if (best_idxs[index] == 3) {
              list(background = "#90EE90")
            } else {
              list(background = "white")
            }
          }
        ),
        knn = colDef(
          name = "k-Nearest Neighbours",
          format = colFormat(digits = 2),
          style = function(value, index) {
            if (best_idxs[index] == 4) {
              list(background = "#90EE90")
            } else {
              list(background = "white")
            }
          }
        ),
        svm = colDef(
          name = "Support Vector Machine",
          format = colFormat(digits = 2),
          style = function(value, index) {
            if (best_idxs[index] == 5) {
              list(background = "#90EE90")
            } else {
              list(background = "white")
            }
          }
        ),
        xgb = colDef(
          name = "XGBoost",
          format = colFormat(digits = 2),
          style = function(value, index) {
            if (best_idxs[index] == 6) {
              list(background = "#90EE90")
            } else {
              list(background = "white")
            }
          }
        )
      ),
      defaultColDef = colDef(
        style = color_scales(all_df, colors = c("#f7fbff", "#08306b")),
        format = colFormat(digits = 2)
      ),
      theme = reactableTheme(
        borderColor = "#dfe2e5",
        stripedColor = "#f6f8fa",
        highlightColor = "#f0f5f9",
        cellPadding = "8px",
        style = list(fontFamily = "Arial, sans-serif")
      ),
      sortable = TRUE,
      filterable = FALSE,
      highlight = TRUE
    )
    
    
    
  })
  
}



#----------Creating a Shiny App object------------------------------------------
shinyApp(ui = ui, server = server)

