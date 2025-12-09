# Essential Commands for UNO POMCP Project

## Development Commands
```bash
# Install dependencies
uv sync

# Run main simulation (Naive vs Naive)
uv run python main.py

# Run batch simulations
uv run python batch_run.py

# Analyze results
uv run python results.py

# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_game.py
```

## Git Commands
```bash
# Check status
git status

# Add changes
git add .

# Commit (concise messages)
git commit -m "add feature"

# Push to remote
git push
```

## File Operations
```bash
# List files
ls

# Find files
find . -name "*.py"

# Search in files
rg "pattern" src/
```