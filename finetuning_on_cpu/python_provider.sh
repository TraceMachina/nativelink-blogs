#!/usr/bin/env bash
# Generic Python test wrapper that handles cross-architecture execution

# Get the directory where this script is located
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# First argument is the Python module or script to run
PYTHON_SCRIPT="$1"
# Remaining arguments are passed to the Python script
shift

echo "Running Python test: $PYTHON_SCRIPT"
echo "Python version:"
python3 --version

# Check if the argument is a file or a module
if [[ -f "$PYTHON_SCRIPT" ]]; then
    # Run as a script if it's a file
    echo "Running as script"
    python3 "$PYTHON_SCRIPT" "$@"
else
    # Run as a module if it's not a file
    echo "Running as module"
    python3 -m "$PYTHON_SCRIPT" "$@"
fi

# Capture and return the exit code
exit_code=$?
echo "Python test completed with exit code: $exit_code"
exit $exit_code
