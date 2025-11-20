import time
from contextlib import contextmanager

class CumulativeTimer:
	def __init__(self):
		self.total_time = 0.0
		self._in_context = 0

	def __enter__(self):
		if self._in_context == 0:
			self._start_time = time.time()
		self._in_context += 1
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		if self._in_context == 0:
			return
		self._in_context -= 1
		if self._in_context == 0:
			end_time = time.time()
			self.total_time += (end_time - self._start_time)

	def reset(self):
		"""重置累计时间"""
		self.total_time = 0.0

	def get_total_time(self):
		"""获取累计时间（秒）"""
		return self.total_time

# 使用装饰器版本的上下文管理器
@contextmanager
def cumulative_timer():
	"""
	上下文管理器，记录在该上下文中的所有代码运行时间之和

	用法:
	with cumulative_timer() as timer:
		# 一些代码
		time.sleep(1)
		# 更多代码
		time.sleep(2)

	print(f"总运行时间: {timer.total_time:.6f} 秒")
	"""
	timer = CumulativeTimer()
	with timer:
		yield timer
