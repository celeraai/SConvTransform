#!/bin/bash

# Verify if both $1 and $2 are set
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <base_mlir_file> <csv_file>"
    echo "  <base_mlir_file>: Path to the base MLIR template file."
    echo "  <csv_file>: Path to the CSV file containing convolution configurations."
    echo "  [--num-runs]: Num of runs. Default=1"
    exit 1
fi

# Paths
BASE_MLIR_FILE=$1  			# Base template MLIR file
CSV_FILE=$2                            	# CSV file with convolution parameters
RUNS=1

for arg in "$@"; do
  case $arg in
    --num-runs=*)
      RUNS="${arg#*=}"
    ;;
  esac
done

CSV_PREFIX=$(basename "${CSV_FILE%%_*}")          # Extract prefix
BASE_MLIR_FILENAME=$(basename "${BASE_MLIR_FILE%.*}")
BASE_MLIR_KIND="${BASE_MLIR_FILENAME%%_*}"
OUTPUT_DIR="generated_mlir_files_${CSV_PREFIX}_${BASE_MLIR_KIND}_${RUNS}"       # Output directory

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

IFS=, read -r -a HEADER < "$CSV_FILE"

# Read CSV file and skip the header
tail -n +2 "$CSV_FILE" | while IFS=, read -r "${HEADER[@]}"; do
    PHI=$((HI + HPBOTTOM + HPTOP))
    PWI=$((WI + WPLEFT + WPRIGHT))

    FLOPS=$((2 * HO * WO * DO * CI * HK * WK))
    
    # Generate the output file name
    OUTPUT_FILE="${OUTPUT_DIR}/conv_${ID}.mlir"

    # Copy the base template and replace placeholders using `sed`
    sed \
        -e "s/{{ID}}/${ID}/g" \
        -e "s/{{NI}}/${NI}/g" \
        -e "s/{{CI}}/${CI}/g" \
        -e "s/{{HI}}/${HI}/g" \
        -e "s/{{PHI}}/${PHI}/g" \
        -e "s/{{WI}}/${WI}/g" \
        -e "s/{{PWI}}/${PWI}/g" \
        -e "s/{{NO}}/${NO}/g" \
        -e "s/{{DO}}/${DO}/g" \
        -e "s/{{HO}}/${HO}/g" \
        -e "s/{{WO}}/${WO}/g" \
        -e "s/{{HK}}/${HK}/g" \
        -e "s/{{WK}}/${WK}/g" \
        -e "s/{{HPTOP}}/${HPTOP}/g" \
        -e "s/{{HPBOTTOM}}/${HPBOTTOM}/g" \
        -e "s/{{WPLEFT}}/${WPLEFT}/g" \
        -e "s/{{WPRIGHT}}/${WPRIGHT}/g" \
        -e "s/{{HS}}/${HS}/g" \
        -e "s/{{WS}}/${WS}/g" \
        -e "s/{{HD}}/${HD}/g" \
        -e "s/{{WD}}/${WD}/g" \
        -e "s/{{GROUP}}/${GROUP}/g" \
        -e "s/{{BIAS}}/${BIAS}/g" \
        -e "s/{{FLOPS}}/${FLOPS}/g" \
        -e "s/{{RUNS}}/${RUNS}/g" \
        "$BASE_MLIR_FILE" > "$OUTPUT_FILE"

    echo "Generated: $OUTPUT_FILE"
done

