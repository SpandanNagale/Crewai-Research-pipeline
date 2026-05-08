import os
import sys
import threading
import queue
from flask import Flask, request, jsonify, render_template, Response
from crew.research_crew import ResearchCrew
from eval.scorer import score_report

app = Flask(__name__)

class ThreadSafeStdout:
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.local = threading.local()

    def write(self, data):
        # We capture output if a queue is configured for the current thread
        if hasattr(self.local, 'queue'):
            self.local.queue.put(data)
        # We also write to the original stdout for normal console viewing
        self.original_stdout.write(data)

    def flush(self):
        self.original_stdout.flush()

# Replace sys.stdout to capture CrewAI prints
original_stdout = sys.stdout
sys.stdout = ThreadSafeStdout(original_stdout)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/research/stream', methods=['POST'])
def stream_research():
    data = request.json
    topic = data.get('topic', '')
    evaluate = data.get('evaluate', False)
    
    if not topic.strip():
        return jsonify({"error": "Topic cannot be empty"}), 400

    def generate():
        # Setup queue for this thread's stdout capture
        q = queue.Queue()
        sys.stdout.local.queue = q
        
        yield f"data: --- STARTING RESEARCH PIPELINE FOR: {topic} ---\n\n"
        
        # We will run the crew.run(topic) in another thread so we don't block the generator
        result_box = {}
        error_box = {}
        
        def run_crew():
            # Setup queue in the crew thread as well
            sys.stdout.local.queue = q
            crew = ResearchCrew()
            try:
                report = crew.run(topic)
                result_box['report'] = report
            except Exception as e:
                error_box['error'] = str(e)
            finally:
                q.put(None) # Sentinel to signal completion
        
        thread = threading.Thread(target=run_crew)
        thread.start()
        
        buffer = ""
        while True:
            try:
                # Wait for data from stdout
                item = q.get(timeout=0.1)
                if item is None:
                    break
                
                # Yield it in SSE format
                lines = item.split('\n')
                for idx, line in enumerate(lines):
                    if idx < len(lines) - 1:
                        yield f"data: {buffer}{line}\n\n"
                        buffer = ""
                    else:
                        buffer += line
            except queue.Empty:
                # Keep alive
                pass
                
        if buffer:
            yield f"data: {buffer}\n\n"

        if 'error' in error_box:
            yield f"data: \n\ndata: --- ERROR ---\ndata: {error_box['error']}\n\n"
            yield "data: [DONE]\n\n"
            return
            
        report = result_box.get('report', '')
        yield f"data: \n\ndata: --- FINAL REPORT ---\n\n"
        
        for chunk in report.split("\n"):
            yield f"data: {chunk}\n\n"
            
        if evaluate:
            yield f"data: \n\ndata: --- EVALUATING REPORT ---\n\n"
            try:
                score = score_report(report, topic)
                for chunk in str(score).split("\n"):
                    yield f"data: {chunk}\n\n"
            except Exception as e:
                yield f"data: Evaluation error: {str(e)}\n\n"
                
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    # Make sure we run threaded so concurrent requests work (though sys.stdout will handle it gracefully)
    app.run(debug=True, threaded=True, port=5000)
