# Define Python interpreter
PYTHON = python
PIP = pip

# Define directory paths
PACKAGE_DIR = optivai_training
TEST_DIR = tests

# Define color codes for better readability
GREEN = \033[0;32m
RED = \033[0;31m
YELLOW = \033[0;33m
NC = \033[0m  # No Color

.PHONY: help setup-dev format check-format lint type-check test coverage security clean all pre-push

help:
	@echo "$(GREEN)OptivAI Training Makefile$(NC)"
	@echo "Available commands:"
	@echo "  $(YELLOW)setup-dev$(NC)     - Install development dependencies"
	@echo "  $(YELLOW)format$(NC)        - Format code using black"
	@echo "  $(YELLOW)check-format$(NC)  - Check code formatting without changing files"
	@echo "  $(YELLOW)lint$(NC)          - Run flake8 linter"
	@echo "  $(YELLOW)type-check$(NC)    - Run mypy type checker"
	@echo "  $(YELLOW)test$(NC)          - Run tests"
	@echo "  $(YELLOW)coverage$(NC)      - Run tests with coverage report"
	@echo "  $(YELLOW)security$(NC)      - Run security checks (requires Snyk CLI)"
	@echo "  $(YELLOW)clean$(NC)         - Remove build artifacts"
	@echo "  $(YELLOW)all$(NC)           - Run all checks (format, lint, type-check, test, coverage)"
	@echo "  $(YELLOW)pre-push$(NC)      - Run all checks before pushing to GitHub"

setup-dev:
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(PIP) install -e ".[dev]"
	$(PIP) install pre-commit
	pre-commit install

format:
	@echo "$(GREEN)Formatting code with black...$(NC)"
	black $(PACKAGE_DIR) $(TEST_DIR)

check-format:
	@echo "$(GREEN)Checking code formatting with black...$(NC)"
	black --check $(PACKAGE_DIR) $(TEST_DIR)

lint:
	@echo "$(GREEN)Running flake8 linter...$(NC)"
	flake8 $(PACKAGE_DIR) $(TEST_DIR) --count --max-complexity=10 --statistics

type-check:
	@echo "$(GREEN)Running mypy type checker...$(NC)"
	mypy $(PACKAGE_DIR) $(TEST_DIR)

test:
	@echo "$(GREEN)Running tests...$(NC)"
	pytest $(TEST_DIR)

coverage:
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	pytest --cov=$(PACKAGE_DIR) --cov-report=term --cov-report=html $(TEST_DIR)
	@echo "$(GREEN)Checking minimum coverage threshold (90%)...$(NC)"
	pytest --cov=$(PACKAGE_DIR) --cov-fail-under=90 $(TEST_DIR)

security:
	@echo "$(GREEN)Running security scan with Snyk...$(NC)"
	@if command -v snyk &> /dev/null; then \
		snyk test --severity-threshold=high; \
	else \
		echo "$(RED)Snyk CLI not found. Install with: npm install -g snyk$(NC)"; \
		exit 1; \
	fi

clean:
	@echo "$(GREEN)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

all: check-format lint type-check test coverage

pre-push: all
	@echo "$(GREEN)All checks passed! Ready to push to GitHub.$(NC)"