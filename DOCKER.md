# Docker Setup for UN Voting Data Visualization

This guide explains how to run the UN Voting Data Visualization application using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (usually included with Docker Desktop)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Clone the repository and navigate to the project directory:**

   ```bash
   git clone https://github.com/loved6/un_voting_data_visualization.git
   cd un_voting_data_visualization
   ```

2. **Build and run the container:**

   ```bash
   docker-compose up --build
   ```

3. **Access the application:**

   Open your browser and go to: `http://localhost:8050`

4. **Stop the application:**

   ```bash
   docker-compose down
   ```

### Option 2: Using Docker directly

1. **Clone the repository and navigate to the project directory (if you haven't already):**

    ```bash
    git clone https://github.com/loved6/un_voting_data_visualization.git
    cd un_voting_data_visualization
    ```

    If you already have the repo, ensure you're in the project root where the Dockerfile and docker-compose.yml live.

2. **Build the Docker image:**

   ```bash
   docker build -t un-voting-viz .
   ```

3. **Run the container:**

   ```bash
   docker run -p 8050:8050 -v $(pwd)/dataset:/app/dataset un-voting-viz
   ```

4. **Access the application:**

   Open your browser and go to: `http://localhost:8050`

## Configuration

### Environment Variables

You can customize the application behavior using environment variables:

- `PYTHONPATH`: Set to `/app` (already configured)
- `PYTHONUNBUFFERED`: Set to `1` for immediate log output

### Volume Mounts

The Docker setup includes a volume mount for the dataset directory:

- Host path: `./dataset`
- Container path: `/app/dataset`

This ensures your dataset files persist between container restarts.

## Development

### Building for Development

If you want to run the container in development mode:

1. **Modify the Dockerfile** to enable debug mode:

   ```dockerfile
   CMD ["python", "-c", "import sys; sys.path.append('/app/src'); from app import main; main()"]
   ```

2. **Add volume mount for source code:**

   ```yaml
   volumes:
     - ./src:/app/src
     - ./dataset:/app/dataset
   ```

### Health Check

The docker-compose setup includes a health check that verifies the application is running correctly. The health check:

- Runs every 30 seconds
- Has a 10-second timeout
- Allows 3 retries
- Waits 40 seconds before starting checks

## Troubleshooting

### Container won't start

- Check if port 8050 is already in use: `lsof -i :8050`
- Verify dataset files exist in the `./dataset` directory
- Check Docker logs: `docker-compose logs un-voting-viz`

### Application not accessible

- Ensure you're connecting to `http://localhost:8050`, not `http://127.0.0.1:8050`
- Check if your firewall is blocking the connection
- Verify the container is running: `docker-compose ps`

### Performance issues

- Increase Docker memory allocation (Docker Desktop settings)
- Consider using multi-stage builds for smaller image size

## Production Deployment

For production deployment:

1. **Use production requirements:**
   The Dockerfile already uses `requirements-docker.txt` which excludes test dependencies.

2. **Set debug to False:**
   The application is configured to run with `debug=False` in Docker.

3. **Security considerations:**
   - The container runs as a non-root user (`appuser`)
   - Consider using Docker secrets for sensitive data
   - Use reverse proxy (nginx) for SSL/TLS termination

## Image Information

- **Base Image:** `python:3.12-slim`
- **Working Directory:** `/app`
- **Exposed Port:** `8050`
- **User:** `appuser` (non-root)
- **Healthcheck:** Enabled
