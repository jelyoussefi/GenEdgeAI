import time
from threading import Thread, Condition
from queue import Queue, Empty, Full
from openvino_genai import LLMPipeline

class Chatbot:
    def __init__(self, chatbot_id, device, model_path, precision="FP16"):
        self.chatbot_id = chatbot_id
        self.device = device
        self.model_path = model_path
        self.precision = precision
        self.running = False
        self.paused = False
        self.thread = None
        self.queue = Queue(maxsize=100)
        self.cv = Condition()
        self.pipeline = None
        self.fps = 0.0
        self.latency = 0.0
        self._token_count = 0
        self._last_time = time.time()
        self.load_model()

    def load_model(self):
        """Load OpenVINO GenAI model."""
        try:
            self.pipeline = LLMPipeline(self.model_path, self.device)
            print(f"Chatbot {self.chatbot_id} loaded model on {self.device}")
        except Exception as e:
            print(f"Error loading model for chatbot {self.chatbot_id}: {e}")

    def start(self):
        """Start chatbot inference thread."""
        with self.cv:
            if not self.running:
                self.running = True
                self.paused = False
                self.thread = Thread(target=self.run)
                self.thread.daemon = True
                self.thread.start()
                return True
        return False

    def stop(self):
        """Stop chatbot inference thread."""
        with self.cv:
            if self.running:
                self.running = False
                self.paused = False
                if self.thread and self.thread.is_alive():
                    self.thread.join()
                self.queue.queue.clear()
                self.cv.notify_all()
                return True
        return False

    def run(self):
        """Chatbot processing loop with benchmarking."""
        try:
            while self.running:
                with self.cv:
                    if not self.running:
                        break
                    while self.paused:
                        self.cv.wait(timeout=0.1)

                try:
                    prompt = self.queue.get(timeout=0.5)
                    if prompt:
                        start_time = time.time()
                        response = self.generate_text(prompt)
                        end_time = time.time()
                        
                        # Calculate latency (in ms)
                        self.latency = (end_time - start_time) * 1000
                        # Calculate FPS (tokens per second)
                        token_count = len(response.split())  # Rough token count
                        self._token_count += token_count
                        elapsed_time = end_time - self._last_time
                        if elapsed_time > 1.0:  # Update FPS every second
                            self.fps = self._token_count / elapsed_time
                            self._token_count = 0
                            self._last_time = end_time
                        
                        self.queue.put(response, timeout=0.001)
                except Empty:
                    continue
        finally:
            self.running = False

    def generate_text(self, prompt):
        """Generate text response using OpenVINO GenAI."""
        try:
            response = self.pipeline.generate(prompt, max_new_tokens=100)
            return response
        except Exception as e:
            print(f"Error in chatbot inference for {self.chatbot_id}: {e}")
            return "Error generating response"

    def send_prompt(self, prompt):
        """Send a prompt to the chatbot."""
        try:
            self.queue.put(prompt, timeout=0.5)
        except Full:
            print("Chatbot queue is full. Dropping request.")

    def get_response(self):
        """Retrieve the chatbot's generated response."""
        try:
            return self.queue.get(timeout=0.5)
        except Empty:
            return None

    def get_benchmark_data(self):
        """Return FPS and latency for benchmarking."""
        return self.fps, self.latency