import numbers
from torch.utils.tensorboard import SummaryWriter
import logging
from .grad_analyzer import GradAnalyzer


def verbose_level_local2logging(local):
    assert local <= 3
    if local != 0:
        local += 1
    logging = local * 10
    return logging


class Logger:
    def __init__(self, level: int, rank: int = None) -> None:
        logging.basicConfig(
            level=verbose_level_local2logging(level),
            format="%(asctime)s %(name)s:%(levelname)s: %(message)s",
        )
        self.logger = logging.getLogger()
        self.rank = rank
        self.display_info = self.logger.level != 0
        self.plotter = None
        self.data = {}

    # logging
    def _format(self, txt: str) -> str:
        return txt if self.rank is None else f"R{self.rank}: {txt}"

    def info(self, txt: str = "") -> None:
        if self.display_info:
            self.logger.info(self._format(txt))

    def warn(self, txt: str = "") -> None:
        self.logger.warn(self._format(txt))

    def error(self, txt: str = "") -> None:
        self.logger.error(self._format(txt))

    # plotting
    def init_plotter(self, logdir, model, layers_per_plot=15):
        self.plotter = SummaryWriter(logdir)
        self.gradient_analyzer = GradAnalyzer(
            model, tb_writer=self.plotter, layers_per_plot=layers_per_plot
        )

    def _plot_to_tb(self, card: str, val: float, glob_step: int) -> None:
        self.plotter.add_scalar(card, val, glob_step)

    def step(self, epoch, analyze_grad: bool = False):
        for k, v in self.data.items():
            v = [val for val in v if val is not None]
            if len(v) != 0:
                self._plot_to_tb(k, sum(v) / len(v), epoch)
        self.data = dict()
        if analyze_grad:
            self.gradient_analyzer.epoch_step(epoch)

    def batch_step(self, analyze_grads: bool = False):
        if analyze_grads:
            if hasattr(self, "gradient_analyzer"):
                self.gradient_analyzer.batch_step()

    def _is_valid(self, val):
        return (
            isinstance(val, numbers.Number)
            and val != float("inf")
            and val != -float("inf")
        )

    def plot(self, category: str, card_name: str, val: float, glob_step: int) -> None:
        """Plots a scalar to the tensorboard"""
        category = category.replace("/", "_")
        card_name = card_name.replace("/", "_")
        card = f"{category}/{card_name}"
        self._plot_to_tb(card, val, glob_step)

    def _accumulate(self, card, y):
        if self._is_valid(y):
            if card in self.data.keys():
                self.data[card].append(y)
            else:
                self.data[card] = [y]

    def accumulate(self, category: str, card_name: str, val: float) -> None:
        """Accumulates a value under a specific card. The accumulated values will be plotted by calling step()"""
        category = category.replace("/", "_")
        card_name = card_name.replace("/", "_")
        card = f"{category}/{card_name}"
        self._accumulate(card, val)
