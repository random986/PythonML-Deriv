"""Simple HTTP server for the frontend dashboard."""
import http.server
import socketserver
import os

PORT = 8000
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

os.chdir(FRONTEND_DIR)

handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), handler) as httpd:
    print(f"Serving frontend at http://127.0.0.1:{PORT}")
    httpd.serve_forever()
