#!/bin/bash

# Define the Uptime Kuma push URL
UPTIME_KUMA_PUSH_URL="${UPTIME_URL}${UPTIME_PUSH_EXPERT}"

# Start time
START_TIME=$(date +%s%N)

# Initialize the variables
EXIT_STATUS=0

# Execute the command
sed -i "s/'TBD'/'${VERSION_ID}'/g" notebooks/getting_started.ipynb
sed -i "s/'TBD'/'${VERSION_ID}'/g" notebooks/checkin_configuration_options.ipynb

# Get the list of notebooks to execute
notebooks=$(./generate-notebook-list.sh)

# Execute all notebooks in order (replace with echo for testing)
for notebook in $notebooks; do
  if [[ -f "$notebook" ]]; then
    echo "Executing notebook $notebook ..."
    python3 -m nbconvert --to notebook --execute --inplace "$notebook" 2>&1 | tee output.log
    COMMAND_EXIT_CODE=${PIPESTATUS[0]}
    if [ ${COMMAND_EXIT_CODE} -ne 0 ]; then
      EXIT_STATUS=1
      echo "Failed to execute $notebook:\n$(tail -n2 output.log | head -n1)" >> errors.log
    fi
  else
    echo "Warning: $notebook not found. Skipping."
  fi
done

# End time
END_TIME=$(date +%s%N)

# Calculate the duration in milliseconds
DURATION=$(( (END_TIME - START_TIME) / 1000000 ))

# Check the exit status of the command
if [ $EXIT_STATUS -eq 0 ]; then
  # On success, push success to Uptime Kuma
  curl -v -G -d status=up -d ping=$DURATION --data-urlencode msg="Success" ${UPTIME_KUMA_PUSH_URL}
else
  # On failure, report error code to Uptime Kuma
  curl -v -G -d status=down -d ping=$DURATION --data-urlencode msg="$(cat errors.log)" ${UPTIME_KUMA_PUSH_URL}
  exit 1 # exit with error code
fi