import re
import json

# Path to your Layer 1 list
LIST_FILE = "/etc/NetworkManager/dnsmasq.d/adblock-auto.conf"

trackers = set()

try:
    with open(LIST_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            # If the line is an address rule, extract the domain
            if line.startswith("address=/") and line.endswith("/0.0.0.0"):
                match = re.match(r'address=/(.+?)/0\.0\.0\.0', line)
                if match:
                    trackers.add(match.group(1))
            # If the line is NOT a rule but just raw text/filter, add it as is
            elif line:
                trackers.add(line)
except FileNotFoundError:
    print(f"Error: Could not find {LIST_FILE}")

print(f"Extracted {len(trackers)} trackers.")

# Save with JSON formatting to prevent SyntaxErrors
with open("known_trackers_from_layer1.py", "w") as f:
    f.write("KNOWN_TRACKERS = {\n")
    for t in sorted(trackers):
        # json.dumps handles nested quotes and backslashes automatically
        safe_string = json.dumps(t)
        f.write(f'    {safe_string},\n')
    f.write("}\n")

print("Successfully updated known_trackers_from_layer1.py with safe formatting.")
