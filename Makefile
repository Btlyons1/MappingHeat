.PHONY: help install train build up down logs clean test lint

# Default target
help:
	@echo "Mapping Heat - Baseball Analytics Platform"
	@echo ""
	@echo "Available commands:"
	@echo "  make install    - Install Python dependencies"
	@echo "  make train      - Run data pipeline and train model"
	@echo "  make build      - Build Docker containers"
	@echo "  make up         - Start services"
	@echo "  make down       - Stop services"
	@echo "  make logs       - View service logs"
	@echo "  make clean      - Clean artifacts and cache"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run code linting"
	@echo "  make shell-backend  - Open shell in backend container"
	@echo "  make shell-frontend - Open shell in frontend container"

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	cd backend && pip install -r requirements.txt

# Train model
train:
	@echo "Running data pipeline..."
	cd backend && python pipeline.py

# Build containers
build:
	@echo "Building Docker containers..."
	docker-compose build

# Start services
up:
	@echo "Starting services..."
	docker-compose up -d
	@echo "Services started!"
	@echo "Frontend: http://localhost"
	@echo "Backend: http://localhost:5000"

# Start services with logs
up-logs:
	@echo "Starting services with logs..."
	docker-compose up

# Stop services
down:
	@echo "Stopping services..."
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# View backend logs only
logs-backend:
	docker-compose logs -f backend

# View frontend logs only
logs-frontend:
	docker-compose logs -f frontend

# Restart services
restart: down up

# Clean artifacts and cache
clean:
	@echo "Cleaning artifacts and cache..."
	rm -rf backend/artifacts/*.pkl
	rm -rf backend/artifacts/*.json
	rm -rf backend/__pycache__
	rm -rf .pybaseball_cache
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete!"

# Deep clean (including Docker)
clean-all: clean
	@echo "Removing Docker containers and images..."
	docker-compose down -v --rmi all
	@echo "Deep clean complete!"

# Run tests
test:
	@echo "Running tests..."
	cd backend && python -m pytest tests/ -v

# Lint code
lint:
	@echo "Linting Python code..."
	cd backend && python -m flake8 . --max-line-length=100
	cd backend && python -m black --check .

# Format code
format:
	@echo "Formatting Python code..."
	cd backend && python -m black .

# Shell into backend container
shell-backend:
	docker exec -it mapping-heat-backend /bin/bash

# Shell into frontend container
shell-frontend:
	docker exec -it mapping-heat-frontend /bin/sh

# Check service health
health:
	@echo "Checking backend health..."
	@curl -s http://localhost:5000/health | python -m json.tool || echo "Backend not responding"
	@echo ""
	@echo "Checking frontend health..."
	@curl -s -o /dev/null -w "%{http_code}" http://localhost/ || echo "Frontend not responding"

# Production build
prod-build:
	@echo "Building for production..."
	docker-compose -f docker-compose.yml build --no-cache

# Development setup
dev-setup: install
	@echo "Setting up development environment..."
	@echo "Creating artifacts directory..."
	mkdir -p backend/artifacts
	@echo "Development setup complete!"
	@echo "Run 'make train' to train the model"
	@echo "Run 'make up' to start services"

# Quick start (for first-time setup)
quickstart: dev-setup train build up
	@echo ""
	@echo "Mapping Heat is ready."
	@echo "Open http://localhost in your browser"

# Backup artifacts
backup:
	@echo "Backing up artifacts..."
	tar -czf artifacts_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz backend/artifacts/
	@echo "Backup created!"

# Monitor resource usage
stats:
	docker stats mapping-heat-backend mapping-heat-frontend

# Update dependencies
update-deps:
	@echo "Updating Python dependencies..."
	cd backend && pip list --outdated
	@echo "Run 'pip install --upgrade <package>' to update specific packages"