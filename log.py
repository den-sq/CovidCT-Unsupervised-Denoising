from collections import namedtuple
from datetime import datetime
from enum import Enum
from sys import stdout
from typing import TextIO

import click
import numpy as np
import psutil

script_start = datetime.now()

Mem = namedtuple("Mem", ['shape', 'name', 'dtype'])

__attached_funcs = []


def mem_size(mem, swap=None):
	if swap is None:
		shape = mem.shape
	else:
		shape = [swap[x] if swap[x] is not None else mem.shape[x] for x in range(len(swap))]
	return int(np.prod(shape, dtype=np.float64) * np.dtype(mem.dtype).itemsize)


# More specific debug-level logging.
class DEBUG(Enum):
	SILENT = ("SILENT", 0, "black")
	ERROR = ("ERROR", 1, "red")
	STATUS = ("STATUS", 2, "green")
	TIME = ("TIME", 3, "cyan")
	WARN = ("WARN", 4, "yellow")
	INFO = ("INFO", 5, "white")

	def __str__(self):
		return str(self.value[0])

	def __le__(self, other):
		return self.value[1] <= other.value[1]

	@property
	def color(self):
		return self.value[2]


def __log_message(step: str, statement: str = '', log_level: DEBUG = DEBUG.TIME):
	styled_type = click.style(f'{log_level.name:6}', log_level.color)
	message = (f'{styled_type}'
			f'|{step[:20]:20}'
			f'|{str(datetime.now() - script_start).zfill(15)}'
			f'|{psutil.Process().memory_info().vms // 1024 ** 2:09.2f}MB'
			f'|{psutil.virtual_memory().available // 1024 ** 2:09.2f}MB'
			f'|"{statement}"')
	return message


def log(step: str, statement: str = '', log_level: DEBUG = DEBUG.TIME, out: TextIO = stdout,
		pid: int = psutil.Process().pid):
	for func in __attached_funcs:
		func(step, pid)
	click.echo(__log_message(step, statement, log_level), file=out, err=(log_level == DEBUG.ERROR))


def log_confirm(step: str, statement: str = '', log_level: DEBUG = DEBUG.TIME, out: TextIO = stdout,
				pid: int = psutil.Process().pid):
	for func in __attached_funcs:
		func(step, pid)
	return click.confirm(__log_message(step, statement, log_level), err=(log_level == DEBUG.ERROR))


def log_prompt(step: str, statement: str = '', log_level: DEBUG = DEBUG.TIME, out: TextIO = stdout,
				pid: int = psutil.Process().pid, default=None):
	for func in __attached_funcs:
		func(step, pid)
	return click.prompt(__log_message(step, statement, log_level), err=(log_level == DEBUG.ERROR), default=default)


def attach_func(func: callable):
	if func not in __attached_funcs:
		__attached_funcs.append(func)


def cleanup_mem(*shm_objects):
	""" Close and unlink shared memory objects.

		:param shm_objects: Shared memory objects to shut down.
	"""
	for shm in shm_objects:
		if shm is not None:
			shm.close()
			shm.unlink()


def exit_cleanly(step: str, *shm_objects, return_code: int = 0, statement: str = '', log_level: DEBUG = DEBUG.TIME,
					out: TextIO = stdout, throw: Exception = None):
	""" Exit while cleaning up shared memory.

		:param step: Step of reconstruction process we are exiting during.
		:param shm_objects: Shared memory objects to shut down.
		:param return_code: Process return code to send.
	"""
	log(step, statement, log_level, out)
	cleanup_mem(*shm_objects)

	if throw is not None:
		raise throw
	exit(return_code)