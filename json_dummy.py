import json

class Model:
    def __init__(self, name=None, type=None, params=None):
        self.name = name
        self.type = type
        self.params = params
        
    @classmethod
    def from_json(cls, data):
        return json.loads(data)
    
    @classmethod
    def print_attributes(cls, data):
        if data["name"] is not None:
            print("name:", data["name"])
        if data["type"] is not None:
            print("type:", data["type"])
        if data["params"] is not None:
            print("params:", data["params"], "\n")
    

# Load JSON data from file
with open("models.json") as f:
    data = f.read()

# Convert JSON data into Model instances
models = Model.from_json(data)

# Print the attributes of each Model instance
for model in models:
    Model.print_attributes(model)


