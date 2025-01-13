#!/bin/bash

# Ordered list of notebooks to execute first
priority_notebooks=(
  "notebooks/getting_started.ipynb"
  "notebooks/working_with_results.ipynb"
  "notebooks/checkin_configuration_options.ipynb"
  "notebooks/forecast_with_covariates.ipynb"
  "notebooks/using_covariates_from_POOL.ipynb"
  "notebooks/cov_matcher_and_forecast.ipynb"
  "notebooks/cov_matcher_and_forecast_monthly.ipynb"
  "notebooks/advanced_workflow.ipynb"
  "use_cases/demand_planning/demand_planning.ipynb"
)

# Combine priority notebooks and other notebooks into one list
all_notebooks=("${priority_notebooks[@]}")
for notebook in $(find notebooks -name '*.ipynb' | sort); do
  is_priority=false
  for priority in "${priority_notebooks[@]}"; do
    if [[ "$notebook" == "$priority" ]]; then
      is_priority=true
      break
    fi
  done

  if [[ "$is_priority" == false ]]; then
    all_notebooks+=("$notebook")
  fi
done


# Output the list of notebooks
for notebook in "${all_notebooks[@]}"; do
  echo "$notebook"
done