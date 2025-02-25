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
		self.token_count = 0  # Track total tokens generated
		self.start_time = None  # Track start time of generation
		self.last_token_time = None  # Track time of the last token
		self.latency_per_token = 0  # Latency in ms per token
		self.throughput = 0  # Tokens per second
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
		# Record time for latency and throughput calculation
		current_time = time.time() * 1000  # Convert to milliseconds
		
		if self.start_time is None:
			self.start_time = current_time  # Set start time on first token
			self.last_token_time = current_time
		else:
			# Calculate latency for this token (time since last token)
			token_latency = current_time - self.last_token_time
			self.latency_per_token = token_latency  # Update latency per token
			self.last_token_time = current_time

			# Update token count and throughput
			self.token_count += 1
			elapsed_time = (current_time - self.start_time) / 1000  # Convert to seconds
			if elapsed_time > 0:
				self.throughput = self.token_count / elapsed_time  # Tokens per second

		# Append subword to buffer and emit it via SocketIO with metrics
		self.message_buffer += subword
		self.socketio.emit('chatbot_stream', {
			'chatbot_id': self.chatbot_id,
			'subword': subword,
			'latency_ms_per_token': round(self.latency_per_token, 2),  # Latency in ms per token
			'throughput_tokens_per_sec': round(self.throughput, 2)  # Throughput in tokens/sec
		})
		return openvino_genai.StreamingStatus.RUNNING

	def start(self):
		try:
			self.pipeline.start_chat()
			self.running = True
			return True
		except Exception as e:
			print(f"Starting chat failed for {self.chatbot_id}: {e}")
			return False

	def stop(self):
		try:
			self.pipeline.finish_chat()
			self.running = False
			# Reset metrics
			self.token_count = 0
			self.start_time = None
			self.last_token_time = None
			self.latency_per_token = 0
			self.throughput = 0
			return True
		except Exception as e:
			print(f"Stopping chat failed for {self.chatbot_id}: {e}")
			return False

	def prompt(self, prompt):
		self.message_buffer = ""  # Reset buffer for new prompt
		self.token_count = 0  # Reset token count
		self.start_time = None  # Reset start time
		self.last_token_time = None  # Reset last token time
		self.latency_per_token = 0  # Reset latency
		self.throughput = 0  # Reset throughput
		try:
			self.pipeline.generate(prompt, self.config, self.streamer)  # Fixed typo: selfStreamer -> self.streamer
			return True
		except Exception as e:
			print(f"Generating answer failed for {self.chatbot_id}: {e}")
			return False

	def get_message(self):
		return self.message_buffer

	def get_benchmark_data(self):
		"""Return latency and throughput for external use (e.g., metrics endpoint)."""
		return self.throughput, self.latency_per_token