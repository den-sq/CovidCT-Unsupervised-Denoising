import click
import numpy as np
from ruamel.yaml import YAML, yaml_object

yaml = YAML()


@yaml_object(yaml)
class FloatRange:
	start: float
	stop: float
	step: float
	yaml_tag = '!FloatRange'

	def __init__(self, start, stop, step):
		self.start = start
		self.stop = stop
		self.step = step

	@classmethod
	def to_yaml(cls, representer, node):
		return representer.represent_scalar(cls.yaml_tag, str(node))

	@classmethod
	def from_yaml(cls, constructor, node):
		return cls(*node.value.split(","))

	def __str__(self):
		return f"{self.start},{self.stop},{self.step}"

	def as_array(self):
		steps = int((self.start - self.stop) // self.step) + 1
		return np.linspace(self.start, self.stop, steps)


# Click Parameter: Float Range (Imitated by linspace).
class Frange(click.ParamType):
	name = "Float Range"

	def convert(self, value, param, ctx):
		try:
			params = [float(x) for x in str(value).split(",")]
			start, stop, step = ([0.] if len(params) == 1 else []) + params + ([1.] if len(params) in [1, 2] else [])
			return FloatRange(start, stop, step)
		except ValueError:
			self.fail(f'{value} cannot be evaluated as a float range.')


FRANGE = Frange()
