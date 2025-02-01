import os
import psutil
import logging
import time
from flask import Flask, render_template, jsonify, request
from flask_bootstrap import Bootstrap
from threading import Thread, Lock, Condition
from collections import deque
from datetime import datetime, timedelta
from utils.system_utils import get_power_consumption, get_gpu_model, list_devices, list_cameras
from utils.chatbot import Chatbot  # Updated import

# Disable Flask's default request logging
log = logging.getLogger('werkzeug')

# Global variables for connection management
connection_lock = Lock()
active_chatbots = {}  # Dictionary to track active chatbots: {chatbot_id: chatbot_instance}
connection_duration = timedelta(minutes=60)  # Limit duration to 60 minutes

# LLM Model definitions
MODELS = [
    "deepseek-ai/DeepSeek-V3-Base"
]

class ChatbotApp:
    def __init__(self, port=80):
        self.app = Flask(__name__)
        Bootstrap(self.app)
        self.port = port
        self.chatbots = {}  # Stores active chatbot instances
        self.lock = Lock()
        self.cv = Condition()
        self.running = True
        self.cpu_loads = deque(maxlen=4)
        self.power_consumptions = deque(maxlen=4)
        self.metrics_thread = Thread(target=self.metrics_handler)
        self.metrics_thread.daemon = True
        self.init_routes()

    def start_chatbot(self, chatbot_id, device, model_name, precision="FP16"):
        """Start a chatbot instance."""
        with self.cv:
            if chatbot_id in self.chatbots and self.chatbots[chatbot_id].running:
                return False
            try:
                chatbot = Chatbot(chatbot_id, device, model_name, precision)
                chatbot.start()
                self.chatbots[chatbot_id] = chatbot
                self.cv.notify_all()
                return True
            except Exception as e:
                print(f"Error starting chatbot {chatbot_id}: {e}")
                return False

    def stop_chatbot(self, chatbot_id):
        """Stop a chatbot instance."""
        with self.cv:
            if chatbot_id in self.chatbots:
                try:
                    chatbot = self.chatbots.pop(chatbot_id)
                    chatbot.stop()
                    self.cv.notify_all()
                    return True
                except Exception as e:
                    print(f"Error stopping chatbot {chatbot_id}: {e}")
                    return False
        return False

    def init_routes(self):
        """Define API routes."""
        app = self.app

        @app.route('/')
        def home():
            platform_name = get_gpu_model().split("[")[0].replace("Intel Corporation", "").strip()
            devices =   list_devices()
            default_device = "CPU"
            default_model = MODELS[0]
            default_precision = "FP16"
 
            return render_template(
                'index.html',
                devices=devices,
                default_device=default_device,
                default_precision=default_precision,
                default_model=default_model,
                platform_name=platform_name,
                models=MODELS
            )

        @app.route('/start_chatbot', methods=['POST'])
        def start_chatbot():
            data = request.get_json()  # ✅ Ensure 'data' is assigned first
            chatbot_id = data.get('chatbot_id')
            data = request.get_json()
            device = data.get('device')
            model = data.get('model')
            precision = data.get('precision', "FP16")

            if self.start_chatbot(chatbot_id, device, model, precision):
                return jsonify({'message': f'Chatbot {chatbot_id} started successfully'})
            return jsonify({'error': f'Failed to start chatbot {chatbot_id}'}), 500

        @app.route('/stop_chatbot', methods=['POST'])
        def stop_chatbot():
            data = request.get_json()
            chatbot_id = data.get('chatbot_id')
            if self.stop_chatbot(chatbot_id):
                return jsonify({'message': f'Chatbot {chatbot_id} stopped successfully'})
            return jsonify({'error': f'Failed to stop chatbot {chatbot_id}'}), 500

        @app.route('/send_prompt', methods=['POST'])
        def send_prompt():
            data = request.get_json()
            chatbot_id = data.get('chatbot_id')
            prompt = data.get('prompt')

            if chatbot_id not in app_obj.chatbots:
                return jsonify({'error': f'Chatbot {chatbot_id} not running'}), 400

            chatbot = self.chatbots[chatbot_id]
            chatbot.send_prompt(prompt)
            return jsonify({'message': 'Prompt sent successfully'}), 200

        @app.route('/get_response', methods=['GET'])
        def get_response():
            """Get the generated response from a chatbot."""
            chatbot_id = request.args.get('chatbot_id')

            if chatbot_id not in app_obj.chatbots:
                return jsonify({'error': f'Chatbot {chatbot_id} not running'}), 400

            chatbot = self.chatbots[chatbot_id]
            response = chatbot.get_response()

            if response is None:
                return jsonify({'error': 'No response available yet'}), 202

            return jsonify({'response': response}), 200

    def metrics_handler(self):
        """Periodically collect system metrics."""
        while self.running:
            with self.cv:
                try:
                    cpu_percent = psutil.cpu_percent(interval=None)
                    power_consumption = get_power_consumption()

                    self.cpu_loads.append(cpu_percent)
                    self.power_consumptions.append(power_consumption)
                    
                except Exception as e:
                    logging.error(f"Metrics collection error: {e}")  

                    self.cv.notify_all()

            time.sleep(1)


    def run(self):
        self.running = True
        self.metrics_thread.start()
        self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)

def main(port=80):
    app = ChatbotApp(port=port)
    app.run()

if __name__ == "__main__":
    main()