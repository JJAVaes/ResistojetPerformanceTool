class Colors:
    black = 'k'
    blue = 'C9'
    red = 'r'
    orange = 'C1'
    green = 'C2'
    yellow = 'C8'
    purple = 'C4'
    brown = 'C5'
    pink = 'C6'
    dark_blue = 'C0'
    dark_red = 'C3'
    grey = '0.5'

    def __getitem__(self, item):
        return [
            self.dark_blue,
            self.orange,
            self.green,
            self.dark_red,
            self.purple,
            self.brown,
            self.pink
        ][item % 7]