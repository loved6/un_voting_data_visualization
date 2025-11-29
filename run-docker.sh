#!/bin/bash

# UN Voting Data Visualization Docker Runner
# This script helps build and run the Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
COMMAND="up"
BUILD=false
DETACH=false

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS] [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  up      Start the application (default)"
    echo "  down    Stop the application"
    echo "  build   Build the Docker image"
    echo "  logs    Show application logs"
    echo "  shell   Open shell in running container"
    echo ""
    echo "Options:"
    echo "  -b, --build    Force rebuild the image"
    echo "  -d, --detach   Run in detached mode"
    echo "  -h, --help     Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                 # Start the application"
    echo "  $0 --build         # Rebuild and start"
    echo "  $0 -d up           # Start in detached mode"
    echo "  $0 down            # Stop the application"
    echo "  $0 logs            # View logs"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--build)
            BUILD=true
            shift
            ;;
        -d|--detach)
            DETACH=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        up|down|build|logs|shell)
            COMMAND=$1
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if dataset directory exists
if [ ! -d "dataset" ]; then
    echo -e "${YELLOW}Warning: dataset directory not found. Creating it...${NC}"
    mkdir -p dataset
    echo -e "${YELLOW}Please add your CSV files to the dataset/ directory.${NC}"
fi

echo -e "${GREEN}UN Voting Data Visualization - Docker Runner${NC}"
echo "=================================================="

case $COMMAND in
    up)
        echo -e "${GREEN}Starting the application...${NC}"
        if [ "$BUILD" = true ]; then
            echo -e "${YELLOW}Rebuilding Docker image...${NC}"
            docker-compose build --no-cache
        fi
        
        if [ "$DETACH" = true ]; then
            docker-compose up -d
            echo -e "${GREEN}Application started in detached mode.${NC}"
            echo -e "${GREEN}Access it at: http://localhost:8050${NC}"
            echo -e "${YELLOW}Run '$0 logs' to view logs${NC}"
        else
            echo -e "${GREEN}Starting application in foreground mode...${NC}"
            echo -e "${GREEN}Access it at: http://localhost:8050${NC}"
            echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
            docker-compose up
        fi
        ;;
    down)
        echo -e "${GREEN}Stopping the application...${NC}"
        docker-compose down
        echo -e "${GREEN}Application stopped.${NC}"
        ;;
    build)
        echo -e "${GREEN}Building Docker image...${NC}"
        docker-compose build --no-cache
        echo -e "${GREEN}Image built successfully.${NC}"
        ;;
    logs)
        echo -e "${GREEN}Showing application logs...${NC}"
        docker-compose logs -f un-voting-viz
        ;;
    shell)
        echo -e "${GREEN}Opening shell in container...${NC}"
        docker-compose exec un-voting-viz /bin/bash
        ;;
esac