class TestStatistic:
    def __init__(self, config):
        self.config = config
        
    @property
    def name(self):
        return 'base_statistic'
        
    def calculate(self, matrix):
        raise NotImplementedError('not implemented')
        
    def visualize(self, statistic_data, output_paths):
        raise NotImplementedError('not implemented')
