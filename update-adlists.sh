#!/bin/bash
# update-adlists.sh - Automatic daily blocklist fetch for OmniBlock Layer 1

set -e  # Exit on any error

# User-writable working directory
WORK_DIR="$HOME/OmniBlock/blocklists"
mkdir -p "$WORK_DIR"

# Final output file (will be copied to system later)
AUTO_CONF="$WORK_DIR/adblock-auto.conf"

# System target directory (requires sudo to write)
SYSTEM_DIR="/etc/NetworkManager/dnsmasq.d"
SYSTEM_CONF="$SYSTEM_DIR/adblock-auto.conf"

echo "=== Updating OmniBlock blocklists ==="
echo "Date: $(date)"

# Fetch major blocklists (fail gracefully if one source is down)
echo "Fetching EasyList..."
curl -s -o "$WORK_DIR/easylist.txt" https://easylist.to/easylist/easylist.txt || echo "EasyList fetch failed"
echo "Fetching EasyPrivacy..."
curl -s -o "$WORK_DIR/easyprivacy.txt" https://easylist.to/easylist/easyprivacy.txt || echo "EasyPrivacy fetch failed"
echo "Fetching StevenBlack hosts..."
curl -s -o "$WORK_DIR/stevenblack.txt" https://raw.githubusercontent.com/StevenBlack/hosts/master/hosts || echo "StevenBlack fetch failed"
echo "Fetching OISD basic..."
curl -s -o "$WORK_DIR/oisd.txt" https://dbl.oisd.nl/basic/ || echo "OISD fetch failed"

# Combine, remove duplicates, convert to dnsmasq address= format
echo "Combining and processing lists..."
cat "$WORK_DIR"/*.txt 2>/dev/null | \
    grep -v '^#' | grep -v '^$' | \
    sort -u | \
    sed 's/^/address=\//;s/$/\/0.0.0.0/' > "$AUTO_CONF"

# Add IPv6 blocking (aaaa=)
cat "$AUTO_CONF" >> "$AUTO_CONF.ipv6"
sed -i 's/address=/aaaa=/g' "$AUTO_CONF.ipv6"
cat "$AUTO_CONF.ipv6" >> "$AUTO_CONF"
rm "$AUTO_CONF.ipv6"

# Count total unique blocked domains
COUNT=$(wc -l < "$AUTO_CONF")
echo "Success: $COUNT domains blocked"

# Copy to system directory (requires sudo)
echo "Copying to system directory..."
sudo cp "$AUTO_CONF" "$SYSTEM_CONF" || {
    echo "Error: Failed to copy to $SYSTEM_CONF"
    exit 1
}

# Reload NetworkManager to apply changes
echo "Reloading NetworkManager..."
sudo systemctl restart NetworkManager

# Log success
LOG_FILE="$HOME/OmniBlock/blocklog.txt"
echo "$(date '+%Y-%m-%d %H:%M:%S') - Updated: $COUNT domains" >> "$LOG_FILE"

echo "Update complete. Check logs: $LOG_FILE"
