#!/bin/bash

# Define the log file directory
LOG_DIR="./logs"

# Check if log directory exists, if not, create it
if [ ! -d "$LOG_DIR" ]; then
    mkdir "$LOG_DIR"
fi

# Function to run the experiment with specific configurations
run_experiment () {
    local counterfactual=$1
    local counterfactual_symbol_result=$2
    local consistent_counterfactual=$3
    local counterfactual_symbol_operands=$4
    local representation=$5
    local intervention_loc=$6
    local NOW=$(date +"%Y-%m-%d_%H-%M-%S")
    local LOG_FILE="$LOG_DIR/logging_$NOW"_cf_"$counterfactual"_csr_"$counterfactual_symbol_result"_ccf_"$consistent_counterfactual"_cso_"$counterfactual_symbol_operands"_rep_"$representation"_il_"$intervention_loc".log
    local FLAG_FILE="$LOG_DIR/flag_$NOW.txt"

    # Ensure the flag file does not exist before starting the experiment
    rm -f "$FLAG_FILE"

    echo "Starting experiment with counterfactual=$counterfactual, counterfactual_symbol_result=$counterfactual_symbol_result, consistent_counterfactual=$consistent_counterfactual, counterfactual_symbol_operands=$counterfactual_symbol_operands, representation=$representation, intervention_loc=$intervention_loc (Logging to $LOG_FILE)..."

    (
        # Run the command
        nohup python -u math_cma.py hydra.run.dir=. hydra/job_logging=disabled hydra/hydra_logging=disabled counterfactual="$counterfactual" counterfactual_symbol_result="$counterfactual_symbol_result" consistent_counterfactual="$consistent_counterfactual" counterfactual_symbol_operands="$counterfactual_symbol_operands" representation="$representation" intervention_loc="$intervention_loc" > "$LOG_FILE" 2>&1

        # Create the flag file to signal completion
        echo "done" > "$FLAG_FILE"
    ) &

    # Wait for the flag file to be created
    while [ ! -f "$FLAG_FILE" ]; do
        sleep 1
    done

    # Optionally, remove the flag file after it's no longer needed
    rm -f "$FLAG_FILE"
    
    echo "Experiment with parameters: $counterfactual, $counterfactual_symbol_result, $consistent_counterfactual, $counterfactual_symbol_operands, $representation, $intervention_loc completed."
}

# Run experiments with the specified configurations
run_experiment "false" "false" "false" "false" "words" "layer"
run_experiment "false" "false" "false" "false" "arabic" "layer"
run_experiment "false" "false" "false" "false" "words" "attention_layer_output"
run_experiment "false" "false" "false" "false" "arabic" "attention_layer_output"

# run_experiment "true" "false" "false" "false" "words" "layer"
# run_experiment "true" "false" "false" "false" "arabic" "layer"
# run_experiment "true" "false" "false" "false" "words" "attention_layer_output"
# run_experiment "true" "false" "false" "false" "arabic" "attention_layer_output"

# run_experiment "true" "false" "true" "false" "words" "layer"
# run_experiment "true" "false" "true" "false" "arabic" "layer"
# run_experiment "true" "false" "true" "false" "words" "attention_layer_output"
# run_experiment "true" "false" "true" "false" "arabic" "attention_layer_output"

# run_experiment "true" "true" "false" "false" "words" "layer"
# run_experiment "true" "true" "false" "false" "arabic" "layer"
# run_experiment "true" "true" "false" "false" "words" "attention_layer_output"
# run_experiment "true" "true" "false" "false" "arabic" "attention_layer_output"

# run_experiment "true" "false" "false" "true" "words" "layer"
# run_experiment "true" "false" "false" "true" "arabic" "layer"
# run_experiment "true" "false" "false" "true" "words" "attention_layer_output"
# run_experiment "true" "false" "false" "true" "arabic" "attention_layer_output"

# run_experiment "true" "false" "false" "false" "words" "attention_layer_output"
# run_experiment "true" "false" "false" "false" "arabic" "attention_layer_output"
# run_experiment "true" "false" "true" "false" "words" "attention_layer_output"
# run_experiment "true" "false" "true" "false" "arabic" "attention_layer_output"
# run_experiment "true" "true" "false" "false" "words" "attention_layer_output"
# run_experiment "true" "true" "false" "false" "arabic" "attention_layer_output"
# run_experiment "true" "false" "false" "true" "words" "attention_layer_output"
# run_experiment "true" "false" "false" "true" "arabic" "attention_layer_output"

# run_experiment "true" "false" "true" "true" "words" "mixed"
# run_experiment "true" "false" "true" "true" "words" "mixed"

echo "All experiments have been completed."

