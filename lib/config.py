import yaml

class Config(object):
    def __init__(self, path='config/settings.yaml'):
        with open(path, 'r') as stream:
            self.data = yaml.load(stream)
        self.model = self.data.get('model')
        self.agent = self.data.get('agent')
        self.env = self.data.get('env')