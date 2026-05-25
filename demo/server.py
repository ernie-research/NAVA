"""
NAVA Demo Server - serves static files + /api/list?dir=xxx for video listing
"""
import os
import json
import urllib.parse
import http.server as _hs
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Some internal Python builds (Secure_SimpleHTTP) refuse all non-whitelisted
# clients with HTTP 403. We open the whitelist to the full IPv4 range so
# personal machines on the same internal network can access this demo.
if hasattr(_hs, "NET_WHITELIST"):
    _hs.NET_WHITELIST = ["0.0.0.0/0"]
if hasattr(_hs, "IP_WHITELIST"):
    _hs.IP_WHITELIST = []
if hasattr(_hs, "ACL_ON"):
    _hs.ACL_ON = True

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DEMO_DIR, **kwargs)

    def do_GET(self):
        if self.path.startswith("/api/list"):
            self.handle_list()
        else:
            # Support Range requests for video seeking
            range_header = self.headers.get('Range')
            if range_header and not self.path.startswith("/api/"):
                self.handle_range(range_header)
            else:
                super().do_GET()

    def handle_range(self, range_header):
        """Handle HTTP Range requests for video seeking."""
        path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        if path.startswith('/'):
            path = path[1:]
        fpath = os.path.join(DEMO_DIR, path)
        if not os.path.isfile(fpath):
            self.send_error(404)
            return
        file_size = os.path.getsize(fpath)
        # Parse range
        try:
            range_spec = range_header.replace('bytes=', '')
            start_str, end_str = range_spec.split('-')
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
        except Exception:
            start, end = 0, file_size - 1
        end = min(end, file_size - 1)
        length = end - start + 1
        # Guess content type
        import mimetypes
        ctype, _ = mimetypes.guess_type(fpath)
        if not ctype:
            ctype = 'application/octet-stream'
        self.send_response(206)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
        self.send_header('Content-Length', str(length))
        self.send_header('Accept-Ranges', 'bytes')
        self.end_headers()
        with open(fpath, 'rb') as f:
            f.seek(start)
            self.wfile.write(f.read(length))

    def handle_list(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        subdir = params.get("dir", [""])[0]
        target = os.path.join(DEMO_DIR, subdir)
        if not os.path.isdir(target):
            files = []
        else:
            import subprocess
            items = []
            for f in sorted(os.listdir(target)):
                if not f.endswith(".mp4"):
                    continue
                fpath = os.path.join(target, f)
                try:
                    r = subprocess.run(
                        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                         '-show_entries', 'stream=width,height', '-of', 'csv=p=0', fpath],
                        capture_output=True, text=True, timeout=5)
                    w, h = r.stdout.strip().split(',')
                    items.append({"name": f, "w": int(w), "h": int(h)})
                except Exception:
                    items.append({"name": f, "w": 1280, "h": 720})
            # Sort: portrait first
            items.sort(key=lambda x: (0 if x["h"] > x["w"] else 1, x["name"]))
            files = items
        body = json.dumps(files).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    import argparse
    import socket
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=8889)
    parser.add_argument("--host", default=None,
                        help="Bind host. Default: result of `hostname -i` (so personal machines on the same network can reach it).")
    args = parser.parse_args()

    if args.host is None:
        try:
            host = socket.gethostbyname(socket.gethostname())
        except Exception:
            host = "0.0.0.0"
    else:
        host = args.host

    server = HTTPServer((host, args.port), DemoHandler)
    print(f"NAVA Demo Server: http://{host}:{args.port}")
    server.serve_forever()
