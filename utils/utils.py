import yaml

def load_class_name(class_name_path):
    with open(class_name_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        return data['class_names']