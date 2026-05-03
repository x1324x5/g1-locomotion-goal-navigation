class BaseAlg:
    def __init__(self, parameter, env):
        self.parameter = parameter
        self.env = env

    def train(self):
        raise NotImplementedError
