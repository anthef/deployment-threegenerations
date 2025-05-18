import http.server
import socketserver
import threading
import time
import os

PORT = int(os.environ.get('PORT', 8501))

class SimpleHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"<html><body><h1>Loading App...</h1><p>Please wait while the dashboard loads (this may take up to 60 seconds).</p><script>setTimeout(function(){window.location.reload();}, 30000);</script></body></html>")

def run_temporary_server():
    with socketserver.TCPServer(("", PORT), SimpleHandler) as httpd:
        print("Serving temporary page at port", PORT)
        # Run for 60 seconds
        httpd.serve_forever()

# Start temporary server in a thread
thread = threading.Thread(target=run_temporary_server)
thread.daemon = True
thread.start()

# Sleep to give Streamlit time to start
time.sleep(60)