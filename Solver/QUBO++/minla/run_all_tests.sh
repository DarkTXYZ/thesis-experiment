#!/bin/bash

# Script to run easy_solver for each test case individually
# This avoids exceeding the 10000 variable limit when processing all files in batch

PROCESSED_DIR="../processed"
RESULTS_DIR="../results"
SOLVER="./easy_solver"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Get timestamp for output file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_CSV="$RESULTS_DIR/easy_solver_results_$TIMESTAMP.csv"

# Write CSV header
echo "filename,vertices,edges,feasibility,penalty_param,cost,time_ms" > "$OUTPUT_CSV"

# Check if solver exists
if [ ! -f "$SOLVER" ]; then
    echo "Error: Solver not found at $SOLVER"
    echo "Please compile the solver first using: make easy_solver"
    exit 1
fi

# Check if processed directory exists
if [ ! -d "$PROCESSED_DIR" ]; then
    echo "Error: Processed directory not found at $PROCESSED_DIR"
    exit 1
fi

# Count test files
NUM_FILES=$(find "$PROCESSED_DIR" -name "*.txt" | wc -l | tr -d ' ')
echo "=== Running MINLA Tests ==="
echo "Found $NUM_FILES test files in $PROCESSED_DIR"
echo "Results will be saved to: $OUTPUT_CSV"
echo ""

# Process each .txt file in the processed directory
COUNTER=0
for GRAPH_FILE in "$PROCESSED_DIR"/*.txt; do
    # Check if file exists (in case no .txt files are found)
    if [ ! -f "$GRAPH_FILE" ]; then
        echo "No test files found"
        exit 1
    fi
    
    COUNTER=$((COUNTER + 1))
    FILENAME=$(basename "$GRAPH_FILE")
    
    echo "[$COUNTER/$NUM_FILES] Processing: $FILENAME"
    
    # Run solver for single file and append to output CSV
    TEMP_OUTPUT="$RESULTS_DIR/temp_single_result.csv"
    "$SOLVER" "$GRAPH_FILE" > /dev/null
    
    # Extract the result line (skip header) from single result file and append to main output
    if [ -f "$RESULTS_DIR/easy_solver_single_result.csv" ]; then
        tail -n 1 "$RESULTS_DIR/easy_solver_single_result.csv" >> "$OUTPUT_CSV"
        rm "$RESULTS_DIR/easy_solver_single_result.csv"
    else
        echo "Warning: No result generated for $FILENAME"
    fi
    
    echo ""
done

echo "=== All Tests Complete ==="
echo "Results saved to: $OUTPUT_CSV"
echo ""
echo "Summary:"
cat "$OUTPUT_CSV" | column -t -s ','
