import json

class Model:
    def __init__(self,name=None, supervised=None, target=None, n_clusters=None):
        self.name = name
        self.supervised = supervised
        self.target = target
        self.n_clusters = n_clusters
        
    @classmethod
    def from_json(cls, data):
        return [cls(**m) for m in json.loads(data)]
    
    @classmethod
    def print_attributes(cls, instance):
        if instance.name is not None:
            print("name:", instance.name)
        if instance.supervised is not None:
            print("supervised:", instance.supervised)
        if instance.target is not None:
            print("target:", instance.target)
        if instance.n_clusters is not None:
            print("n_clusters:", instance.n_clusters, "\n")


# Load JSON data from file
with open("models.json") as f:
    data = f.read()

# Convert JSON data into instances of the Model class
models = Model.from_json(data)

# Print the attributes of each instance
for model in models:
    Model.print_attributes(model)