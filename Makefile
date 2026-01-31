# Batuta Ground Truth MLOps Corpus Makefile
# Pure Rust MLOps patterns for the Sovereign AI Stack
# Quality: PMAT A+ compliance, 95%+ coverage, zero tolerance for defects

.SUFFIXES:
.DELETE_ON_ERROR:
.PHONY: help build test test-fast lint fmt fmt-check quality-gates clean
.PHONY: cov coverage coverage-html coverage-summary coverage-clean coverage-check
.PHONY: tier1 tier2 tier3
.PHONY: mutants mutants-fast mutants-file
.DEFAULT_GOAL := help

# Color output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m

# Coverage threshold (PMAT A+ requires 95%)
COV_THRESHOLD ?= 95

# Coverage exclusions (test infrastructure, main.rs)
COV_EXCLUDE := --ignore-filename-regex='(/tests/|_tests|tests_|test_|main\.rs|/benches/|/examples/)'

help: ## Show this help message
	@echo "Batuta Ground Truth MLOps Corpus"
	@echo "================================="
	@echo "Pure Rust MLOps patterns using ONLY Sovereign AI Stack"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# === Build Targets ===

build: ## Build the project
	@echo "$(GREEN)Building...$(NC)"
	cargo build

build-release: ## Build in release mode
	@echo "$(GREEN)Building (release)...$(NC)"
	cargo build --release

# === Tiered Testing (Certeza methodology) ===

tier1: ## Tier 1: Sub-second feedback (ON-SAVE)
	@echo "$(BLUE)🚀 TIER 1: Sub-second feedback$(NC)"
	@cargo check --lib --quiet
	@cargo clippy --lib --quiet -- -D warnings
	@echo "$(GREEN)✅ Tier 1 complete$(NC)"

tier2: fmt-check lint test-lib ## Tier 2: Pre-commit checks (<30s)
	@echo "$(GREEN)✅ Tier 2 complete$(NC)"

tier3: quality-gates ## Tier 3: Full quality gates

# === Test Targets ===

test: ## Run all tests
	@echo "$(GREEN)Running tests...$(NC)"
	cargo test --lib

test-lib: ## Run library tests only (fast)
	@echo "$(GREEN)Running library tests...$(NC)"
	cargo test --lib

test-fast: ## Run fast tests (no property tests)
	@echo "$(GREEN)Running fast tests...$(NC)"
	cargo test --lib -- --skip prop_

# === Lint Targets ===

lint: ## Run clippy with strict warnings
	@echo "$(GREEN)Running clippy...$(NC)"
	cargo clippy -- -D warnings

fmt: ## Format code
	@echo "$(GREEN)Formatting code...$(NC)"
	cargo fmt

fmt-check: ## Check code formatting
	@echo "$(GREEN)Checking formatting...$(NC)"
	cargo fmt --check || (echo "$(RED)❌ Format check failed. Run 'make fmt'$(NC)" && exit 1)

# =============================================================================
# COVERAGE: Realizar-style single-command coverage (target: 95%)
# =============================================================================
# Pattern from realizar:
# - Single cargo llvm-cov command for all tests
# - No batching overhead, no multiple invocations
# - Threshold check with bc for floating point comparison
# =============================================================================

cov: coverage ## Alias for coverage

coverage: coverage-clean ## Run coverage and enforce 95% threshold (PMAT A+)
	@TOTAL_START=$$(date +%s); \
	echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"; \
	echo "$(BLUE)📊 COVERAGE: PMAT A+ Compliance (target: $(COV_THRESHOLD)%)$(NC)"; \
	echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)🧪 Running tests with coverage instrumentation...$(NC)"
	@cargo llvm-cov test --lib --html --output-dir target/coverage $(COV_EXCLUDE) 2>&1 | tail -3
	@echo ""
	@echo "$(GREEN)📊 Generating report...$(NC)"
	@mkdir -p target/coverage/html
	@cargo llvm-cov report --html --output-dir target/coverage/html $(COV_EXCLUDE) 2>&1 | tail -1
	@echo ""
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@cargo llvm-cov report --summary-only $(COV_EXCLUDE) 2>&1 | grep -E "^TOTAL"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@TOTAL_END=$$(date +%s); \
	echo "⏱️  Total: $$((TOTAL_END-TOTAL_START))s"; \
	echo ""; \
	echo "💡 HTML: target/coverage/html/index.html"; \
	COVERAGE=$$(cargo llvm-cov report --summary-only $(COV_EXCLUDE) 2>/dev/null | grep "TOTAL" | awk '{print $$10}' | sed 's/%//'); \
	if [ -n "$$COVERAGE" ]; then \
		RESULT=$$(echo "$$COVERAGE >= $(COV_THRESHOLD)" | bc -l 2>/dev/null || echo 0); \
		if [ "$$RESULT" = "1" ]; then \
			echo "$(GREEN)✅ CORROBORATED: $$COVERAGE% >= $(COV_THRESHOLD)%$(NC)"; \
		else \
			echo "$(RED)❌ FALSIFIED: $$COVERAGE% < $(COV_THRESHOLD)% (gap: $$(echo "$(COV_THRESHOLD) - $$COVERAGE" | bc)%)$(NC)"; \
			exit 1; \
		fi; \
	fi

coverage-html: coverage ## Generate HTML coverage report and open
	@if [ -f target/coverage/html/index.html ]; then \
		xdg-open target/coverage/html/index.html 2>/dev/null || \
		open target/coverage/html/index.html 2>/dev/null || \
		echo "Please open: target/coverage/html/index.html"; \
	fi

coverage-summary: ## Show coverage summary
	@cargo llvm-cov report --summary-only $(COV_EXCLUDE) 2>/dev/null || echo "Run 'make coverage' first"

coverage-check: ## Check coverage threshold without re-running tests
	@echo "🔒 Checking $(COV_THRESHOLD)% coverage threshold..."; \
	COVERAGE=$$(cargo llvm-cov report --summary-only $(COV_EXCLUDE) 2>/dev/null | grep "TOTAL" | awk '{print $$10}' | sed 's/%//'); \
	if [ -z "$$COVERAGE" ]; then echo "❌ No coverage data. Run 'make coverage' first."; exit 1; fi; \
	echo "Coverage: $$COVERAGE%"; \
	RESULT=$$(echo "$$COVERAGE >= $(COV_THRESHOLD)" | bc -l 2>/dev/null || echo 0); \
	if [ "$$RESULT" = "1" ]; then \
		echo "$(GREEN)✅ Coverage $$COVERAGE% >= $(COV_THRESHOLD)% threshold$(NC)"; \
	else \
		echo "$(RED)❌ FAIL: Coverage $$COVERAGE% < $(COV_THRESHOLD)% threshold$(NC)"; \
		exit 1; \
	fi

coverage-clean: ## Clean coverage artifacts
	@rm -rf target/coverage target/llvm-cov
	@find . -name "*.profraw" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Coverage artifacts cleaned$(NC)"

# === Mutation Testing ===

mutants: ## Run mutation testing (full)
	@echo "$(GREEN)🧬 Running mutation tests...$(NC)"
	cargo mutants --timeout 60

mutants-fast: ## Run mutation testing (quick sample)
	@echo "$(GREEN)🧬 Running mutation tests (sample)...$(NC)"
	cargo mutants --timeout 30 --jobs 4 -- --lib

mutants-file: ## Run mutation testing on a single file (FILE=path/to/file.rs)
	@if [ -z "$(FILE)" ]; then \
		echo "$(RED)❌ Error: FILE parameter required$(NC)"; \
		echo "Usage: make mutants-file FILE=src/path/to/file.rs"; \
		exit 1; \
	fi
	@echo "$(GREEN)🧬 Running mutation testing on $(FILE)...$(NC)"
	cargo mutants --file '$(FILE)' --no-times || true
	@echo "$(GREEN)📊 Mutation testing complete for $(FILE)$(NC)"

# === Quality Gates ===

quality-gates: fmt-check lint test coverage ## Run all quality gates (PMAT A+)
	@echo ""
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)✅ ALL QUALITY GATES PASSED (PMAT A+ Compliant)$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"

# === Examples ===

examples: ## Run all examples
	@echo "$(GREEN)Running examples...$(NC)"
	cargo run --example tokenization_demo
	cargo run --example random_forest_demo

# === Clean ===

clean: coverage-clean ## Clean build artifacts
	@echo "$(GREEN)Cleaning...$(NC)"
	cargo clean
	@rm -rf mutants.out mutants.out.old

# === Development ===

dev: ## Start development watch mode
	@echo "$(GREEN)Starting development mode...$(NC)"
	cargo watch -x 'test --lib' -x 'clippy'

# === Installation ===

install-tools: ## Install development tools
	@echo "$(GREEN)Installing development tools...$(NC)"
	cargo install cargo-llvm-cov --locked || true
	cargo install cargo-mutants || true
	cargo install cargo-watch || true
	@echo "$(GREEN)✅ Tools installed$(NC)"
