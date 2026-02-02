#!/bin/bash
# Quick setup script for the new configuration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect docker compose command (plugin vs standalone)
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    DOCKER_COMPOSE="docker compose"
fi

echo "========================================="
echo "RAG Chatbot - Network Access Setup"
echo "========================================="
echo ""

# Step 1: Stop current services
echo "1. Stopping current services..."
$DOCKER_COMPOSE down 2>/dev/null || true
echo "   ✓ Services stopped"
echo ""

# Step 2: Rebuild with new configuration
echo "2. Building services with new configuration..."
$DOCKER_COMPOSE build --no-cache nginx frontend
echo "   ✓ Services built"
echo ""

# Step 3: Start services
echo "3. Starting services..."
$DOCKER_COMPOSE up -d
echo "   ✓ Services starting..."
echo ""

# Step 4: Wait for services to be ready
echo "4. Waiting for services to be ready..."
sleep 30

# Check nginx
if $DOCKER_COMPOSE ps | grep -q "nginx.*Up\|nginx.*running"; then
    echo "   ✓ Nginx is running"
else
    echo "   ✗ Nginx failed to start"
fi

# Check frontend
if $DOCKER_COMPOSE ps | grep -q "frontend.*Up\|frontend.*running"; then
    echo "   ✓ Frontend is running"
else
    echo "   ✗ Frontend failed to start"
fi

# Check backend
if $DOCKER_COMPOSE ps | grep -q "backend.*Up\|backend.*running"; then
    echo "   ✓ Backend is running"
else
    echo "   ✗ Backend failed to start"
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""

# Get server IP
SERVER_IP=$(hostname -I | awk '{print $1}')

echo "Access the application:"
echo "  • From this server:  http://localhost/"
echo "  • From network:      http://$SERVER_IP/"
echo ""
echo "Backend API (localhost only):"
echo "  • http://localhost:8000/health"
echo ""
echo "Control external access:"
echo "  • Enable:  ./toggle-access.sh on"
echo "  • Disable: ./toggle-access.sh off"
echo "  • Status:  ./toggle-access.sh status"
echo ""
echo "View logs:"
echo "  • $DOCKER_COMPOSE logs -f"
echo ""

# Test backend locally
echo "Testing backend (localhost)..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "  ✓ Backend is responding on localhost"
else
    echo "  ✗ Backend is not responding (it may take a few more seconds to start)"
fi

# Test nginx
echo "Testing nginx proxy..."
sleep 2
if curl -s http://localhost/health > /dev/null 2>&1; then
    echo "  ✓ Nginx is responding"
else
    echo "  ⚠ Nginx may still be starting..."
fi

echo ""
echo "For detailed documentation, see NETWORK_ACCESS.md"
