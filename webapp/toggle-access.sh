#!/bin/bash
# Script to toggle external access to the web application

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yaml"

# Detect docker compose command (plugin vs standalone)
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    DOCKER_COMPOSE="docker compose"
fi

# Check if services are running
if ! $DOCKER_COMPOSE -f "$COMPOSE_FILE" ps | grep -q "Up\|running"; then
    echo "Error: Services are not running. Start them first with: $DOCKER_COMPOSE up -d"
    exit 1
fi

# Check current state
CURRENT_CONFIG=$($DOCKER_COMPOSE -f "$COMPOSE_FILE" exec -T nginx cat /etc/nginx/nginx.conf 2>/dev/null | head -n 20 | grep -q "Service temporarily unavailable" && echo "disabled" || echo "enabled")

if [ "$1" == "status" ]; then
    echo "External access is currently: $CURRENT_CONFIG"
    exit 0
fi

if [ "$1" == "on" ] || [ "$1" == "enable" ]; then
    echo "Enabling external access..."
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" exec -T nginx cp /etc/nginx/nginx.conf.enabled /etc/nginx/nginx.conf
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" exec -T nginx nginx -s reload
    echo "✓ External access enabled"
    echo "Frontend accessible at: http://<server-ip>"
    
elif [ "$1" == "off" ] || [ "$1" == "disable" ]; then
    echo "Disabling external access..."
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" exec -T nginx cp /etc/nginx/nginx.conf.disabled /etc/nginx/nginx.conf
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" exec -T nginx nginx -s reload
    echo "✓ External access disabled"
    echo "External users will see: Service temporarily unavailable"
    
else
    echo "Usage: $0 {on|off|enable|disable|status}"
    echo ""
    echo "Current status: $CURRENT_CONFIG"
    echo ""
    echo "Examples:"
    echo "  $0 on       # Enable external access"
    echo "  $0 off      # Disable external access"
    echo "  $0 status   # Check current status"
    exit 1
fi
