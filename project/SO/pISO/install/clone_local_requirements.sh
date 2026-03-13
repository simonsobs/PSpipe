#!/bin/bash

# --- Configuration ---
# Set the path to your input file
INPUT_FILE="local_requirements.txt"

# Set the target directory where repositories will be cloned
# This directory will be created if it doesn't exist.
REPOS_DIR="../repos"

# --- Script Logic ---

# Check if the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found."
    echo "Please make sure the file exists and the path is correct."
    exit 1
fi

# Create the repositories directory if it doesn't exist
mkdir -p "$REPOS_DIR"

echo "Starting git cloning process from '$INPUT_FILE'..."
echo "Repositories will be cloned into '$REPOS_DIR/'"
echo "---"

# Read the input file line by line
while IFS= read -r line; do
    # Skip empty lines or lines starting with a '#' (for comments)
    if [[ -z "$line" || "$line" =~ ^# ]]; then
        continue
    fi

    # The full path is the line itself
    FULL_PATH="$line"

    # Extract the 'name' part, which is the last component after the last '/'
    # basename command extracts the filename from a path
    REPO_NAME=$(basename "$FULL_PATH")

    # Construct the git clone command
    CLONE_COMMAND="git clone git@github.com:${FULL_PATH}.git ${REPOS_DIR}/${REPO_NAME}"

    echo "Processing line: '$line'"
    echo "  Cloning: git@github.com:${FULL_PATH}.git"
    echo "  Into: ${REPOS_DIR}/${REPO_NAME}"
    echo "  Executing command: $CLONE_COMMAND"

    # Execute the git clone command
    if eval "$CLONE_COMMAND"; then
        echo "  Successfully cloned '$REPO_NAME'."
    else
        echo "  Error: Failed to clone '$REPO_NAME'."
        echo "  Check the repository path and your SSH keys/permissions."
    fi
    echo "---"

done < "$INPUT_FILE"

echo "Git cloning process completed."

