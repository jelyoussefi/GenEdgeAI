# chatbot.py
import time
from threading import Thread, Condition
from queue import Queue, Empty, Full
from collections import deque
import openvino_genai 
from openvino_genai import StreamingStatus 

class Chatbot:
    def __init__(self, chatbot_id, device, model_path, socketio):
        self.chatbot_id = chatbot_id
        self.device = device
        self.model_path = model_path
        self.socketio = socketio
        self.running = False
        self.cv = Condition()
        self.message_buffer = ""  # Buffer to store streamed message
        self.token_count = 0  # Track total tokens generated for current generation
        self.start_time = None  # Track start time of current generation
        self.last_token_time = None  # Track time of the last token
        self.latency_per_token = 0  # Current latency in ms per token
        self.throughput = 0  # Current throughput in tokens per second
        self.queue = Queue(maxsize=100)
        # History deques to store metrics for averaging
        self.latency_history = deque(maxlen=100)  # Latency in ms per token
        self.throughput_history = deque(maxlen=100)  # Throughput in tokens per second
        self.generating = False
        self.canceling = False 

    def load_model(self):
        try:
            print(f"Chatbot {self.chatbot_id} loading model on {self.device}")
            pipeline_config = { "CACHE_DIR": ".npucache" }
            self.pipeline = openvino_genai.LLMPipeline(self.model_path, self.device, pipeline_config)
            self.config = openvino_genai.GenerationConfig()
            self.config.max_new_tokens = 1000 # if self.device == "NPU" else 1000
            print(f"Chatbot {self.chatbot_id} loaded model on {self.device}")
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
            self.latency_per_token = token_latency  # Update current latency per token
            self.last_token_time = current_time

            # Update token count and throughput
            self.token_count += 1
            elapsed_time = (current_time - self.start_time) / 1000  # Convert to seconds
            if elapsed_time > 0:
                self.throughput = self.token_count / elapsed_time  # Tokens per second

            # Append current metrics to history
            self.latency_history.append(self.latency_per_token)
            self.throughput_history.append(self.throughput)

        if subword:  # Only process non-empty subwords
            # Emit the raw subword (including <think> and </think>) with metrics
            self.socketio.emit('chatbot_stream_message', {
                'chatbot_id': self.chatbot_id,
                'subword': subword,
                'latency_ms_per_token': round(sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0, 2),
                'throughput_tokens_per_sec': round(sum(self.throughput_history) / len(self.throughput_history) if self.throughput_history else 0, 2)
            })

        with self.cv:
            if self.generating:
                if self.canceling:
                    return StreamingStatus.STOP
                return StreamingStatus.RUNNING
            else:
                self.message_buffer = ''
                return StreamingStatus.STOP

    def start(self):
        with self.cv:
            if not self.running:
                self.running = True
                self.canceling = False
                self.thread = Thread(target=self.run)
                self.thread.daemon = True
                self.thread.start()
                return True
        return False

    def stop(self):
        with self.cv:
            if self.running:
                self.running = False
                self.generating = False
                self.canceling = False
                if self.thread and self.thread.is_alive():
                    self.thread.join()
                self.queue.queue.clear()
                self.cv.notify_all()
                return True
        return False

    def cancel(self):
        with self.cv:
            if self.running:
                self.canceling = True
                while self.canceling:
                    self.cv.wait(0.1)
                self.cv.notify_all()
                return True
        return False

    def run(self):
        try:
            with self.cv:
                self.load_model()
                self.socketio.emit('chatbot_ready', {'chatbot_id': self.chatbot_id})

            while self.running:
                try:
                    prompt = self.queue.get(timeout=0.5)
                except Empty:
                    continue

                self.message_buffer = "" 
                self.token_count = 0  
                self.start_time = None  
                self.last_token_time = None  
                self.latency_per_token = 0  
                self.throughput = 0  
                
                # Generate response and stream subwords
                with self.cv:
                    self.generating = True

                self.pipeline.start_chat()

                self.socketio.emit('chatbot_stream_start', {
                    'chatbot_id': self.chatbot_id
                })
                
                res = self.pipeline.generate(prompt, self.config, self.streamer)

                self.pipeline.finish_chat()

                self.socketio.emit('chatbot_stream_stop', {
                    'chatbot_id': self.chatbot_id
                })
                
                with self.cv:
                    self.generating = False
                    self.canceling = False
                
        finally:
            self.running = False

    def prompt(self, prompt):
        with self.cv:
            self.generating = False
        try:
            self.queue.put(prompt, timeout=1)
        except Full:
            return False
        return True
        
    def get_message(self):
        return self.message_buffer

    def get_benchmark_data(self):
        """Return average latency and throughput from historical data."""
        avg_throughput = sum(self.throughput_history) / len(self.throughput_history) if self.throughput_history else 0
        avg_latency = sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0
        return avg_throughput, avg_latency