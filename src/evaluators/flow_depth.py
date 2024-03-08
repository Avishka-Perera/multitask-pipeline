
class MovingAverage:
    postfix = "avg"

    def __init__(self):
        self._sum = 0.0
        self._count = 0

    def add_value(self, sigma, addcount=1):
        self._sum += sigma
        self._count += addcount

    def add_average(self, avg, addcount):
        self._sum += avg*addcount
        self._count += addcount

    def mean(self):
        return self._sum / self._count

class FlowDepthEvaluator:
    def __init__(self, out_path=None):
        if out_path is not None:
            self.set_out_path(out_path)
        self.moving_averages_dict = None

    def tensor2float_dict(self,tensor_dict):
        return {key: tensor.item() for key, tensor in tensor_dict.items()}

    def process_batch(self, batch, info):
        loss_dict_per_step = self.tensor2float_dict(info)
        if self.moving_averages_dict is None:
                    self.moving_averages_dict = {
                        key: MovingAverage() for key in loss_dict_per_step.keys()
                    }
        for key, loss in loss_dict_per_step.items():
                    self.moving_averages_dict[key].add_average(loss, addcount=1)

    def get_results(self):
        return { key: ma.mean() for key, ma in self.moving_averages_dict.items() }