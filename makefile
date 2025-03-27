# Define Python interpreter and package manager
PYTHON = python3
PIP = pip3

# Define directory paths
SRC_DIR = src
TEST_DIR = tests

# Define color codes for better readability
GREEN = \033[0;32m
RED = \033[0;31m
YELLOW = \033[0;33m
NC = \033[0m  # No Color

.PHONY: help setup test format lint type-check coverage all pre-push clean

help:
	@echo "$(GREEN)OptivAI Training Makefile$(NC)"
	@echo "Available commands:"
	@echo "  $(YELLOW)setup$(NC)         - Install all dependencies"
	@echo "  $(YELLOW)format$(NC)        - Format code with black"
	@echo "  $(YELLOW)lint$(NC)          - Run flake8 linter"
	@echo "  $(YELLOW)type-check$(NC)    - Run mypy type checker"
	@echo "  $(YELLOW)test$(NC)          - Run tests"
	@echo "  $(YELLOW)coverage$(NC)      - Run tests with coverage"
	@echo "  $(YELLOW)all$(NC)           - Run all checks"
	@echo "  $(YELLOW)pre-push$(NC)      - Run all checks before pushing"
	@echo "  $(YELLOW)clean$(NC)         - Clean temporary files"

setup:
	@echo "$(GREEN)Installing dependencies...$(NC)"
	$(PIP) install flake8 black mypy pytest pytest-cov pre-commit
	$(PIP) install -e .
	pre-commit install

format:
	@echo "$(GREEN)Formatting code with black...$(NC)"
	black $(SRC_DIR) $(TEST_DIR)

lint:
	@echo "$(GREEN)Running flake8 linter...$(NC)"
	flake8 $(SRC_DIR) $(TEST_DIR) --count --max-complexity=10 --statistics

type-check:
	@echo "$(GREEN)Running mypy type checker...$(NC)"
	mypy $(SRC_DIR) $(TEST_DIR)

test:
	@echo "$(GREEN)Running tests...$(NC)"
	pytest $(TEST_DIR) -v

coverage:
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	pytest --cov=$(SRC_DIR) --cov-report=term $(TEST_DIR)
	pytest --cov=$(SRC_DIR) --cov-fail-under=90 $(TEST_DIR)

all: format lint type-check test coverage

pre-push: all
	@echo "$(GREEN)All checks completed!$(NC)"

clean:
	@echo "$(GREEN)Cleaning temporary files...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete