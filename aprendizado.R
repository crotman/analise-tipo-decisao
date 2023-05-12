

library(tidyverse)
library(tidymodels)


dados_tortura <- read_csv("d
                          ados-tortura/aprendizado-estatistico.csv") |>
  janitor::clean_names() |>
  mutate(
    pai_identificado = !is.na(nom_pai_autuado)
  ) |>
  select(
    -nom_pai_autuado
  ) |>
  sample_n(100000)


set.seed(42)

data_split <- initial_split(dados_tortura, prop = 0.7, strata = flg_relato_tortura)
train_data <- training(data_split)
test_data <- testing(data_split)

recipe <- recipe(flg_relato_tortura ~ ., data = train_data) |>
  step_impute_mode(
    all_nominal_predictors()
  ) |>
  step_impute_mean(
    all_numeric_predictors()
  )


cv_folds <- vfold_cv(train_data, v = 5)

rand_forest_model <- rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_engine("randomForest" ) |>
  set_mode(mode = "classification" )

workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(rand_forest_model)

tuning_grid <- grid_regular(
  mtry(range = c(1, 10)),
  trees(range = c(100, 300)),
  min_n(range = c(1, 4)),
  levels = 3
)

doParallel::registerDoParallel()
set.seed(42)

tuned_results <- tune_grid(
  workflow,
  resamples = cv_folds,
  grid = tuning_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = TRUE)
)







