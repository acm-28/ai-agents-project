# Makefile para el proyecto AI Agents
# Comandos para desarrollo, testing y despliegue

.PHONY: help install install-dev test test-cov lint format clean build docs dev-setup

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy

# Directorios
SRC_DIR := ai_agents
TEST_DIR := tests
DOCS_DIR := docs

help:  ## Mostrar esta ayuda
	@echo "Comandos disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Instalar el paquete
	$(PIP) install -e .

install-dev:  ## Instalar el paquete con dependencias de desarrollo
	$(PIP) install -e ".[dev]"
	pre-commit install

test:  ## Ejecutar tests
	$(PYTEST) $(TEST_DIR)/ -v

test-unit:  ## Ejecutar solo tests unitarios
	$(PYTEST) $(TEST_DIR)/ -v -m "unit"

test-integration:  ## Ejecutar solo tests de integración
	$(PYTEST) $(TEST_DIR)/ -v -m "integration"

test-cov:  ## Ejecutar tests con coverage
	$(PYTEST) $(TEST_DIR)/ --cov=$(SRC_DIR) --cov-report=html --cov-report=term

test-watch:  ## Ejecutar tests en modo watch
	$(PYTEST) $(TEST_DIR)/ -f

lint:  ## Ejecutar linting
	$(FLAKE8) $(SRC_DIR)/ $(TEST_DIR)/
	$(MYPY) $(SRC_DIR)/

lint-fix:  ## Ejecutar linting y auto-fix lo que sea posible
	$(BLACK) $(SRC_DIR)/ $(TEST_DIR)/
	$(ISORT) $(SRC_DIR)/ $(TEST_DIR)/

format:  ## Formatear código
	$(BLACK) $(SRC_DIR)/ $(TEST_DIR)/
	$(ISORT) $(SRC_DIR)/ $(TEST_DIR)/

format-check:  ## Verificar formato sin cambiar archivos
	$(BLACK) --check $(SRC_DIR)/ $(TEST_DIR)/
	$(ISORT) --check-only $(SRC_DIR)/ $(TEST_DIR)/

clean:  ## Limpiar archivos temporales
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/

build:  ## Construir el paquete
	$(PYTHON) -m build

docs:  ## Generar documentación
	mkdocs build

docs-serve:  ## Servir documentación localmente
	mkdocs serve

dev-setup:  ## Configurar entorno de desarrollo completo
	@echo "Configurando entorno de desarrollo..."
	$(PIP) install -e ".[dev]"
	pre-commit install
	@echo "Creando directorios necesarios..."
	mkdir -p data/cache data/models logs
	@echo "Copiando archivo de configuración..."
	@if not exist .env (copy .env.example .env && echo "Archivo .env creado. Por favor configurar las variables necesarias.")
	@echo "Entorno de desarrollo configurado!"

check:  ## Ejecutar todas las verificaciones (lint + test)
	$(MAKE) lint
	$(MAKE) test

ci:  ## Ejecutar pipeline de CI completo
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) test-cov

release-patch:  ## Crear release patch (incrementar versión patch)
	bumpversion patch

release-minor:  ## Crear release minor (incrementar versión minor)
	bumpversion minor

release-major:  ## Crear release major (incrementar versión major)
	bumpversion major

run-example:  ## Ejecutar ejemplo básico
	$(PYTHON) examples/basic_chat.py

install-pre-commit:  ## Instalar pre-commit hooks
	pre-commit install

update-deps:  ## Actualizar dependencias
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r requirements.txt

# Comandos específicos para Windows
ifeq ($(OS),Windows_NT)
dev-setup-win:  ## Configurar entorno de desarrollo en Windows
	@echo "Configurando entorno de desarrollo en Windows..."
	$(PIP) install -e ".[dev]"
	pre-commit install
	@echo "Creando directorios necesarios..."
	if not exist data mkdir data
	if not exist data\cache mkdir data\cache
	if not exist data\models mkdir data\models
	if not exist logs mkdir logs
	@echo "Copiando archivo de configuración..."
	if not exist .env copy .env.example .env
	@echo "Entorno de desarrollo configurado para Windows!"
endif
