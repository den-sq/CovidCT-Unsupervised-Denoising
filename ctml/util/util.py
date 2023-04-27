import numpy as np
from ruamel.yaml import YAML, yaml_object


yaml = YAML()


@yaml_object(yaml)
class FloatRange:
    """ Range for Float Variables

        An internal array is used, but only created once an item is accessed.
        It is recreated once created whenever a parameter is updated,
        so for large ranges that usage could be sluggish.
    """

    _start: float
    _stop: float
    _step: float
    _space: np.array(float)
    yaml_tag = '!FloatRange'

    def __init__(self, start, stop, step):
        self._start = start
        self._stop = stop
        self._step = step
        self._space = None

    def _update_space(self):
        steps = int((self.start - self.stop) // self.step) + 1
        self._space = np.linspace(self.start, self.stop, steps)

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_scalar(cls.yaml_tag, str(node))

    @classmethod
    def from_yaml(cls, constructor, node):
        return cls(*node.value.split(","))

    def __str__(self):
        return f"{self.start},{self.stop},{self.step}"

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        self._start = value
        if self._space is not None:
            self._update_space()

    @property
    def stop(self):
        return self._stop

    @stop.setter
    def stop(self, value):
        self._stop = value
        if self._space is not None:
            self._update_space()

    @property
    def step(self):
        return self._step

    @stop.setter
    def stop(self, value):
        self._step = value
        if self._space is not None:
            self._update_space()

    def __getitem__(self, index):
        if self._space is None:
            self._update_space()
        return self.space[index]
