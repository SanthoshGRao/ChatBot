import json

# Load JSON data into a list
with open('intents.json', encoding='utf-8') as f:
    json_list = json.load(f)

result = []
items_set = set()

for js in json_list['intents']:
    print(js['tag'])
    # Only add unseen items (referring to 'title' as key)
    if js['tag'] not in items_set:
        # Mark as seen
        items_set.add(js['tag'])
        # Add to results
        result.append(js)

# Write to new JSON file
with open('intent1.json' ,'w') as nf:
    json.dump(result, nf)

print(result)
