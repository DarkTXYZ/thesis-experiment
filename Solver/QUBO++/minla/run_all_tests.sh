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
    # Redirect both stdout and stderr, and capture exit code
    TEMP_LOG="$RESULTS_DIR/temp_solver.log"
    
    if timeout 300 "$SOLVER" "$GRAPH_FILE" > "$TEMP_LOG" 2>&1; then
        # Success - extract the result
        if [ -f "$RESULTS_DIR/easy_solver_single_result.csv" ]; then
            tail -n 1 "$RESULTS_DIR/easy_solver_single_result.csv" >> "$OUTPUT_CSV"
            rm "$RESULTS_DIR/easy_solver_single_result.csv"
            echo "  ✓ Completed successfully"
        else
            echo "  ⚠ Warning: No result file generated"
            echo "$FILENAME,0,0,ERROR,0,0,0" >> "$OUTPUT_CSV"
        fi
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            echo "  ✗ TIMEOUT (exceeded 300 seconds)"
            echo "$FILENAME,0,0,TIMEOUT,0,0,300000" >> "$OUTPUT_CSV"
        else
            echo "  ✗ SOLVER ERROR (exit code: $EXIT_CODE)"
            # Try to extract error message from log
            ERROR_MSG=$(grep -m 1 "what():" "$TEMP_LOG" | sed 's/.*what():  //' || echo "Unknown error")
            echo "     Error: $ERROR_MSG"
            echo "$FILENAME,0,0,SOLVER_ERROR,0,0,0" >> "$OUTPUT_CSV"
        fi
        # Clean up any partial result file
        rm -f "$RESULTS_DIR/easy_solver_single_result.csv"
    fi
    
    rm -f "$TEMP_LOG"
    echo ""
done

echo "=== All Tests Complete ==="
echo "Results saved to: $OUTPUT_CSV"
echo ""

# Count results by status
TOTAL=$(tail -n +2 "$OUTPUT_CSV" | wc -l | tr -d ' ')
SUCCESS=$(tail -n +2 "$OUTPUT_CSV" | grep -c "Feasible\|Infeasible" || echo "0")
ERRORS=$(tail -n +2 "$OUTPUT_CSV" | grep -c "ERROR\|TIMEOUT\|SOLVER_ERROR" || echo "0")

echo "Summary Statistics:"
echo "  Total files:      $TOTAL"
echo "  Successful:       $SUCCESS"
echo "  Errors/Timeouts:  $ERRORS"
echo ""

if [ $ERRORS -gt 0 ]; then
    echo "Failed files:"
    tail -n +2 "$OUTPUT_CSV" | grep "ERROR\|TIMEOUT\|SOLVER_ERROR" | cut -d',' -f1 | sed 's/^/  - /'
    echo ""
fi

echo "Detailed Results:"
cat "$OUTPUT_CSV" | column -t -s ','
