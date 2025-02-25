import os
import psutil
import logging
import time
import fire
from flask import Flask, render_template, jsonify, request
from flask_bootstrap import Bootstrap
from flask_socketio import SocketIO
from threading import Thread, Lock, Condition
from collections import deque
from datetime import datetime, timedelta
from utils.system_utils import get_power_consumption, get_gpu_model, list_devices
from utils.chatbot import Chatbot

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class ChatbotApp:
	def __init__(self, port=80):
		self.app = Flask(__name__)
		Bootstrap(self.app)
		self.socketio = SocketIO(self.app)
		self.port = port
		self.chatbots = {}
		self.lock = Lock()
		self.cv = Condition()
		self.running = True
		self.cpu_loads = deque(maxlen=10)
		self.power_consumptions = deque(maxlen=10)
		self.metrics_thread = Thread(target=self.metrics_handler, daemon=True)
		
		# Populate models and precisions from /opt/models/
		self.models = []
		self.precisions = []
		self.load_models()
		
		self.init_routes()
		print(f"Available models: {self.models}")
		print(f"Available precisions: {self.precisions}")

	def load_models(self):
		"""Scan /opt/models/ to populate available models and precisions."""
		models_dir = '/opt/models/'
		if os.path.exists(models_dir):
			model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
			self.models = sorted(model_dirs)
			
			precision_set = set()
			for model in self.models:
				model_path = os.path.join(models_dir, model)
				precisions = [p for p in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, p))]
				precision_set.update(precisions)
			self.precisions = sorted(precision_set)
		

	def start_chatbot(self, chatbot_id, device, model_name, precision="FP16"):
		with self.cv:
			if chatbot_id in self.chatbots and self.chatbots[chatbot_id].running:
				return False
			try:
				model_path = os.path.join('/opt/models', model_name, precision, model_name)
				chatbot = Chatbot(chatbot_id, device, model_path, precision, self.socketio)
				self.chatbots[chatbot_id] = chatbot
				chatbot.start()
				self.cv.notify_all()
				return True
			except Exception as e:
				log.error(f"Error starting chatbot {chatbot_id}: {e}")
				return False

	def stop_chatbot(self, chatbot_id):
		with self.cv:
			if chatbot_id in self.chatbots:
				try:
					chatbot = self.chatbots.pop(chatbot_id)
					chatbot.stop()
					self.cv.notify_all()
					return True
				except Exception as e:
					log.error(f"Error stopping chatbot {chatbot_id}: {e}")
					return False
		return False

	def init_routes(self):
		app = self.app

		@app.route('/')
		def home():
			platform_name = get_gpu_model().split("[")[0].replace("Intel Corporation", "").strip()
			devices = list_devices()
			default_device = "CPU"
			default_model = self.models[0] if self.models else "TinyLlama-1.1B-Chat-v1.0"
			default_precision = "FP16" if "FP16" in self.precisions else self.precisions[0] if self.precisions else "FP16"

			return render_template(
				'index.html',
				devices=devices,
				default_device=default_device,
				default_precision=default_precision,
				default_model=default_model,
				platform_name=platform_name,
				models=self.models,
				precisions=self.precisions
			)

		@app.route('/start_chatbot', methods=['POST'])
		def start_chatbot():
			data = request.get_json()
			chatbot_id = data.get('chatbot_id')
			device = data.get('device', 'CPU')
			model = data.get('model', self.models[0] if self.models else "TinyLlama-1.1B-Chat-v1.0")
			precision = data.get('precision', "FP16")

			if self.start_chatbot(chatbot_id, device, model, precision):
				return jsonify({'message': f'Chatbot {chatbot_id} started successfully'})
			return jsonify({'error': f'Failed to start chatbot {chatbot_id}'}), 500

		@app.route('/prompt_chatbot', methods=['POST'])
		def prompt_chatbot():
			data = request.get_json()
			chatbot_id = data.get('chatbot_id')
			prompt = data.get('prompt')

			with self.cv:
				if chatbot_id in self.chatbots:
					chatbot = self.chatbots[chatbot_id]
					ret = chatbot.prompt(prompt)
					if ret:
						return jsonify({'message': f'Prompt successfully sent to {chatbot_id}'})
					else:
						return jsonify({'message': f'Prompt cannot be sent to {chatbot_id}'})

			return jsonify({'error': f'Chatbot {chatbot_id} not found'}), 404

		@app.route('/get_metrics', methods=['GET'])
		def get_metrics():
			metrics = {
				'cpu_percent': sum(self.cpu_loads) / len(self.cpu_loads) if self.cpu_loads else 0,
				'power_data': sum(self.power_consumptions) / len(self.power_consumptions) if self.power_consumptions else 0,
				'chatbots': {}
			}
			with self.lock:
				for chatbot_id, chatbot in self.chatbots.items():
					throughput, latency = chatbot.get_benchmark_data()  # Get real values from chatbot
					metrics['chatbots'][chatbot_id] = {
						'fps': round(throughput, 2),  # Throughput in tokens per second
						'latency': round(latency, 2),  # Latency in ms per token
						'model': chatbot.model_path.split('/')[-2],
						'precision': chatbot.precision,
						'device': chatbot.device
					}
			return jsonify(metrics)

	def metrics_handler(self):
		while self.running:
			with self.cv:
				try:
					cpu_percent = psutil.cpu_percent(interval=None)
					power_consumption = get_power_consumption()
					self.cpu_loads.append(cpu_percent)
					self.power_consumptions.append(power_consumption)
				except Exception as e:
					log.error(f"Metrics collection error: {e}")
				self.cv.notify_all()
			time.sleep(1)

	def run(self):
		self.running = True
		self.metrics_thread.start()
		self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=False)

def main(port=80):
	app = ChatbotApp(port=port)
	app.run()

if __name__ == "__main__":
	fire.Fire(main)