class DynamicsModel:
    def __init__(self, f, G=None):
        self.f_ = f
        self.G_ = G

    def f(self, x, u):
        return self.f_(x, u)

    def G(self, x, u):
        if self.G_ is None:
            return None
        return self.G_(x, u)

    def hasG(self):
        return self.G_ is None
