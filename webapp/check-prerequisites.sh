#!/bin/bash
# Pre-deployment checklist and validation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "Pre-Deployment Checklist"
echo "========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

checks_passed=0
checks_failed=0

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((checks_passed++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((checks_failed++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check 1: Docker installed
echo "1. Checking Docker..."
if command -v docker &> /dev/null; then
    check_pass "Docker is installed"
else
    check_fail "Docker is not installed"
fi

# Check 2: Docker Compose installed
echo "2. Checking Docker Compose..."
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    check_pass "Docker Compose is installed"
else
    check_fail "Docker Compose is not installed"
fi

# Check 3: Required files exist
echo "3. Checking configuration files..."
required_files=(
    "docker-compose.yaml"
    "nginx.conf"
    "nginx-disabled.conf"
    "Dockerfile.nginx"
    "Dockerfile.backend"
    "Dockerfile.frontend"
    "toggle-access.sh"
    "setup-network-access.sh"
)

all_files_exist=true
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        check_pass "$file exists"
    else
        check_fail "$file is missing"
        all_files_exist=false
    fi
done

# Check 4: Scripts are executable
echo "4. Checking script permissions..."
if [ -x "toggle-access.sh" ]; then
    check_pass "toggle-access.sh is executable"
else
    check_fail "toggle-access.sh is not executable (run: chmod +x toggle-access.sh)"
fi

if [ -x "setup-network-access.sh" ]; then
    check_pass "setup-network-access.sh is executable"
else
    check_fail "setup-network-access.sh is not executable (run: chmod +x setup-network-access.sh)"
fi

# Check 5: Port 80 availability
echo "5. Checking port availability..."
if ! netstat -tuln 2>/dev/null | grep -q ":80 "; then
    check_pass "Port 80 is available"
else
    check_warn "Port 80 is already in use (may need to stop current services)"
fi

if ! netstat -tuln 2>/dev/null | grep -q ":8000 "; then
    check_pass "Port 8000 is available"
else
    check_warn "Port 8000 is already in use (may need to stop current services)"
fi

# Check 6: Docker daemon running
echo "6. Checking Docker daemon..."
if docker info &> /dev/null; then
    check_pass "Docker daemon is running"
else
    check_fail "Docker daemon is not running"
fi

# Check 7: Required directories
echo "7. Checking directory structure..."
if [ -d "backend" ]; then
    check_pass "backend/ directory exists"
else
    check_fail "backend/ directory is missing"
fi

if [ -d "frontend" ]; then
    check_pass "frontend/ directory exists"
else
    check_fail "frontend/ directory is missing"
fi

if [ -f "backend/api.py" ]; then
    check_pass "backend/api.py exists"
else
    check_fail "backend/api.py is missing"
fi

# Check 8: GPU availability
echo "8. Checking GPU support "
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        check_pass "NVIDIA GPU is available"
    else
        check_warn "NVIDIA GPU tools installed but GPU not accessible"
    fi
else
    check_warn "NVIDIA GPU not detected (backend will use CPU)"
fi

# Summary
echo ""
echo "========================================="
echo "Summary"
echo "========================================="
echo -e "Passed: ${GREEN}$checks_passed${NC}"
echo -e "Failed: ${RED}$checks_failed${NC}"
echo ""

if [ $checks_failed -eq 0 ]; then
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    echo ""
    echo "You can now run:"
    echo "  ./setup-network-access.sh"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some checks failed. Please fix the issues above.${NC}"
    echo ""
    exit 1
fi
