import torch


class TimeFeatures:

    @classmethod
    def y_to_t(cls, y, exp_lambd):
        """
        Converts time features in [0,1) to time values based on the exponential CDF
        """
        return -torch.log(1 - y) / exp_lambd

    @classmethod
    def t_to_y(cls, t, exp_lambd):
        """
        Converts time values to time features in [0,1) based on the exponential CDF
        """
        return 1 - torch.exp(-exp_lambd * t)

    @classmethod
    def dy_dt(cls, t, exp_lambd):
        """
        Derivative of the function that converts time values to features
        """
        return exp_lambd * torch.exp(-exp_lambd * t)
