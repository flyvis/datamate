#!/bin/bash

# Default settings
PORT=8000
COMMANDS=()
DEFAULT_COMMANDS=("run_nbs" "clear_nbs" "serve_docs")
RUN_DEFAULTS=true

# Get the current conda environment name if one is active
CONDA_ENV=""
if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    CONDA_ENV="$CONDA_DEFAULT_ENV"
elif [[ -n "$CONDA_PREFIX" ]]; then
    CONDA_ENV=$(basename "$CONDA_PREFIX")
fi

# Initialize error collection
errors=()

# Function to handle errors
handle_error() {
    local error_msg="$1"
    echo -e "\n❌ Error: $error_msg"
    errors+=("$error_msg")
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [commands] [options]"
    echo "Commands:"
    echo "  run_nbs      - Execute notebooks in place"
    echo "  convert_nbs  - Convert notebooks to markdown"
    echo "  clear_nbs    - Clear notebooks"
    echo "  serve_docs   - Start local documentation server"
    echo ""
    echo "Options:"
    echo "  --port PORT  - Specify port for docs server (default: 8000)"
    echo ""
    echo "If no commands are specified, all commands will be run in sequence."
    exit 1
}

# Parse command line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        run_nbs|convert_nbs|clear_nbs|serve_docs)
            COMMANDS+=("$1")
            RUN_DEFAULTS=false
            ;;
        --port)
            PORT="$2"
            shift
            ;;
        --help|-h)
            show_usage
            ;;
        *)
            echo "Unknown parameter: $1"
            show_usage
            ;;
    esac
    shift
done

# If no commands specified, use defaults
if [ $RUN_DEFAULTS = true ]; then
    COMMANDS=("${DEFAULT_COMMANDS[@]}")
fi

# Create temporary directory for Jupyter config
export JUPYTER_CONFIG_DIR=$(mktemp -d)
export TESTING=true  # Optional: speeds up by not running animations

# Function to run notebooks
run_notebooks() {
    echo "🔄 Running notebooks..."
    mkdir -p docs/examples/
    for notebook in ../examples/*.ipynb; do
        filename=$(basename "$notebook")
        echo "  Running $notebook..."
        jupyter nbconvert --to notebook \
            --execute "$notebook" \
            --output-dir docs/examples/ \
            --output "$filename" \
            2>/tmp/nb_error || {
            error_content="Failed to execute notebook: $notebook\n$(cat /tmp/nb_error)"
            handle_error "$error_content"
        }
    done
}

# Function to convert notebooks
convert_notebooks() {
    echo -e "\n🔄 Converting notebooks to markdown..."
    jupyter nbconvert --to markdown ../examples/*.ipynb --output-dir docs/examples/ --TagRemovePreprocessor.remove_cell_tags hide 2>/tmp/nb_error || {
        error_content="Failed to convert notebooks to markdown:\n$(cat /tmp/nb_error)"
        handle_error "$error_content"
    }
}

# Add new function to clear notebooks
clear_notebooks() {
    echo "🔄 Clearing notebook outputs..."
    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace ../examples/*.ipynb 2>/tmp/nb_error || {
        error_content="Failed to clear notebook outputs:\n$(cat /tmp/nb_error)"
        handle_error "$error_content"
    }
}

# Function to serve docs
serve_docs() {
    echo -e "\n🔄 Starting local docs server on port $PORT..."
    if ! command -v tmux &> /dev/null; then
        echo "tmux is not installed. Starting server in background instead."
        mkdocs serve -a localhost:$PORT &
    else
        tmux new-session -d -s datamate-docs 2>/dev/null || true
        # Activate conda environment in tmux session if one is active
        if [[ -n "$CONDA_ENV" ]]; then
            tmux send-keys -t datamate-docs "eval \"\$(conda shell.bash hook)\" && conda activate $CONDA_ENV" C-m
        fi
        tmux send-keys -t datamate-docs "mkdocs serve -a localhost:$PORT" C-m
    fi
}

# Execute requested commands
for cmd in "${COMMANDS[@]}"; do
    case $cmd in
        run_nbs)
            run_notebooks
            ;;
        convert_nbs)
            convert_notebooks
            ;;
        clear_nbs)
            clear_notebooks
            ;;
        script_docs)
            build_script_docs
            ;;
        serve_docs)
            serve_docs
            ;;
    esac
done

# Print summary
echo -e "\n📋 Summary:"
if [ ${#errors[@]} -eq 0 ]; then
    echo "✅ All steps completed successfully!"
else
    echo "⚠️ The following errors occurred during execution:"
    echo "----------------------------------------"
    printf '%s\n\n' "${errors[@]}"
fi

# Only show server URL if serve_docs was run
if [[ " ${COMMANDS[@]} " =~ " serve_docs " ]]; then
    echo -e "🌐 Documentation is now available at: http://127.0.0.1:$PORT"
    echo "   (The server is running in a tmux session 'datamate-docs')"
fi
