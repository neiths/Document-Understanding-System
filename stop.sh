#!/bin/bash

# Stop script for Document Understanding System

echo "Stopping Document Understanding System..."

# Stop all services
docker-compose down

# Optional: Clean up volumes
read -p "Do you want to remove volumes? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleaning up volumes..."
    docker-compose down -v
fi

echo "System stopped."