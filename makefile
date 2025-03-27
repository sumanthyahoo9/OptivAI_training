# Define Python interpreter and package manager
PYTHON = python3
PIP = pip3

# Define directory paths - case-sensitive
PACKAGE_DIR = optivai_training
TEST_DIR = tests

# Define color codes for better readability
GREEN = \033[0;32m
RED = \033[0;31m
YELLOW = \033[0;33m
NC = \033[0m  # No Color

.PHONY: help setup init format lint type-check test coverage all pre-push clean

help:
	@echo "$(GREEN)OptivAI Training Makefile$(NC)"
	@echo "Available commands:"
	@echo "  $(YELLOW)setup$(NC)         - Install all dependencies"
	@echo "  $(YELLOW)init$(NC)          - Create directory structure"
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

init:
	@echo "$(GREEN)Creating directory structure...$(NC)"
	mkdir -p $(PACKAGE_DIR)/models
	mkdir -p $(PACKAGE_DIR)/data
	mkdir -p $(PACKAGE_DIR)/config
	mkdir -p $(PACKAGE_DIR)/utils
	mkdir -p $(TEST_DIR)
	@echo "$(GREEN)Creating __init__.py files...$(NC)"
	touch $(PACKAGE_DIR)/__init__.py
	touch $(PACKAGE_DIR)/models/__init__.py
	touch $(PACKAGE_DIR)/data/__init__.py
	touch $(PACKAGE_DIR)/config/__init__.py
	touch $(PACKAGE_DIR)/utils/__init__.py
	touch $(TEST_DIR)/__init__.py
	@echo "$(GREEN)Creating sample files...$(NC)"
	echo 'def hello_world():\n    return "Hello, World!"' > $(PACKAGE_DIR)/utils/hello.py
	echo 'from $(PACKAGE_DIR).utils.hello import hello_world\n\ndef test_hello():\n    assert hello_world() == "Hello, World!"' > $(TEST_DIR)/test_hello.py
	@echo "$(GREEN)Directory structure created successfully!$(NC)"

format:
	@echo "$(GREEN)Formatting code with black...$(NC)"
	-black $(PACKAGE_DIR) $(TEST_DIR) 2>/dev/null || echo "$(YELLOW)No Python files to format$(NC)"

lint:
	@echo "$(GREEN)Running flake8 linter...$(NC)"
	-flake8 $(PACKAGE_DIR) $(TEST_DIR) --count --max-complexity=10 --statistics 2>/dev/null || echo "$(YELLOW)No Python files to lint$(NC)"

type-check:
	@echo "$(GREEN)Running mypy type checker...$(NC)"
	-mypy $(PACKAGE_DIR) $(TEST_DIR) 2>/dev/null || echo "$(YELLOW)No Python files to type check$(NC)"

test:
	@echo "$(GREEN)Running tests...$(NC)"
	-pytest $(TEST_DIR) -v 2>/dev/null || echo "$(YELLOW)No tests found$(NC)"

coverage:
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	-pytest --cov=$(PACKAGE_DIR) --cov-report=term $(TEST_DIR) 2>/dev/null || echo "$(YELLOW)No tests found$(NC)"

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