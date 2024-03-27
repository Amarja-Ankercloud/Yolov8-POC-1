import json
from datetime import datetime

# Path to your JSON file
file_path = 'data.json'

# Initialize a dictionary to hold the aggregated counts
counts_per_class_per_hour = {}

# Open and read the JSON file line by line
with open(file_path, 'r') as file:
    for line in file:
        # Parse each line as a JSON object
        entry = json.loads(line)
        
        # Parse the timestamp and round it down to the nearest hour
        timestamp = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
        hour_key = timestamp.strftime("%Y-%m-%d %H:00:00")
        
        # Aggregate counts per class per hour
        for object_class, count in entry["object_counts"].items():
            if hour_key not in counts_per_class_per_hour:
                counts_per_class_per_hour[hour_key] = {}
            if object_class not in counts_per_class_per_hour[hour_key]:
                counts_per_class_per_hour[hour_key][object_class] = 0
            counts_per_class_per_hour[hour_key][object_class] += count

# Print the aggregated counts
print(counts_per_class_per_hour)
