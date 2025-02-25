# chatbot.py
import time
from threading import Thread, Condition
import openvino_genai 

class Chatbot:
	def __init__(self, chatbot_id, device, model_path, precision, socketio):
		self.chatbot_id = chatbot_id
		self.device = device
		self.model_path = model_path
		self.precision = precision
		self.socketio = socketio
		self.running = False
		self.cv = Condition()
		self.message_buffer = ""  # Buffer to store streamed message
		self.load_model()

	def load_model(self):
		try:
			self.pipeline = openvino_genai.LLMPipeline(self.model_path, self.device)
			print(f"Chatbot {self.chatbot_id} loaded model on {self.device}")
			self.config = openvino_genai.GenerationConfig()
			self.config.max_new_tokens = 1000
		except Exception as e:
			print(f"Error loading model for chatbot {self.chatbot_id}: {e}")

	def streamer(self, subword):
		# Append subword to buffer and emit it via SocketIO
		self.message_buffer += subword
		self.socketio.emit('chatbot_stream', {
			'chatbot_id': self.chatbot_id,
			'subword': subword
		})
		print(subword, end='', flush=True)
		return openvino_genai.StreamingStatus.RUNNING

	def start(self):
		try:
			self.pipeline.start_chat()
			self.running = True
			return True
		except Exception as e:
			print(f"Starting chat failed for {self.chatbot_id}: {e}")
			return False

	def stop(self):  # Corrected method name from 'start' to 'stop'
		try:
			self.pipeline.stop_chat()
			self.running = False
			return True
		except Exception as e:
			print(f"Stopping chat failed for {self.chatbot_id}: {e}")
			return False

	def prompt(self, prompt):
		self.message_buffer = ""  # Reset buffer for new prompt
		try:
			self.pipeline.generate(prompt, self.config, self.streamer)
			return True
		except Exception as e:
			print(f"Generating answer failed for {self.chatbot_id}: {e}")
			return False

	def get_message(self):
		return self.message_buffer