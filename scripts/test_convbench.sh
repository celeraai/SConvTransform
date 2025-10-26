#!/usr/bin/env bash
#
# Script to test a transform sequence on ConvBench.
# Uses test_transform.sh script.

err() {
    echo "$*" >&2
    exit 1
}

# Check parameters
if [ $# -lt 2 ]; then
    err "Usage: ./test_convbench.sh <transform-ir>.mlir DIRPATH [-O]"
fi
readonly TRANSFORM=$1
readonly CONVBENCHDIR=$2
shift 2
OPTIMIZATION=""
while getopts 'O' flag; do
    case "${flag}" in
    O) OPTIMIZATION="-O" ;;
    *) error "Unexpected option ${flag}" ;;
    esac
done
readonly OPTIMIZATION

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
readonly SCRIPT

# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")
readonly SCRIPTPATH

# Cleanup
rm -f "$CONVBENCHDIR"/*.llvm.mlir "$CONVBENCHDIR"/*.opt.mlir

# Output
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
NC=$(tput sgr0)

success=()
failure=()
for file in "$CONVBENCHDIR"/*; do
    "$SCRIPTPATH"/test_transform.sh "$file" "$TRANSFORM" "$OPTIMIZATION" > /dev/null

    case $? in
    0)
        success=("${success[@]}" "$file")
        printf "%sSuccess:%s %s\n" "${GREEN}" "${NC}" "$file"
        ;;
    1)
        failure=("${failure[@]}" "$file")
        printf "%sFailure:%s %s\n" "${RED}" "${NC}" "$file"
        ;;
    esac

done

for successful in "${success[@]}"; do
    printf "%sSuccess:%s %s\n" "${GREEN}" "${NC}" "$successful"
done

for failed in "${failure[@]}"; do
    printf "%sFailure:%s %s\n" "${RED}" "${NC}" "$failed"
done

# Summary
printf "%sSuccessful Tests:%s %s\n" "${GREEN}" "${NC}" "${#success[@]}"
printf "%sFailed Tests:%s %s\n" "${RED}" "${NC}" "${#failure[@]}"

rm -f "$CONVBENCHDIR"/*.llvm.mlir "$CONVBENCHDIR"/*.opt.mlir
