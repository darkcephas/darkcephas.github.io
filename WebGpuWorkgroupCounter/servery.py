#Use to create local host
import http.server
import socketserver
import sys
import urllib.request
import socketserver
import webbrowser
import os
import argparse
import json

PORT = int(sys.argv[1]) if len(sys.argv)>1 else 6543

class GameEngineRequestHandler(http.server.SimpleHTTPRequestHandler):
      def end_headers(self):
        # https://gist.github.com/aallan/9416763d42534ae99f6f0228f54160c9
        self.send_my_headers()
        http.server.SimpleHTTPRequestHandler.end_headers(self)

      def send_my_headers(self):
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")

      def do_GET(self):
            http.server.SimpleHTTPRequestHandler.do_GET(self)

      def do_POST(self):
            print("Server post " + self.path)
            if(self.path.find(".js") == -1 ):
                  print("Warning only expected to save json files")
            # content is included from the header  
            content_length = int(self.headers['Content-Length'])
            # read the content from the file 
            post_data_bytes = self.rfile.read(content_length)
            # convert to ascii
            post_data_str = post_data_bytes.decode("UTF-8")
            data_as_python_json = json.loads(post_data_str)
            full_file_name_path = os.path.dirname(os.path.abspath(__file__)) +self.path
            print(full_file_name_path)

            text_file = open(full_file_name_path , "w")
            # pretty print so we can edit by hand
            text_file.write(json.dumps(data_as_python_json,indent=4))
            text_file.close()
            self.send_response(200)

Handler = GameEngineRequestHandler
Handler.extensions_map.update({
      ".js": "application/javascript",
});





if __name__ == '__main__':
      try:
            httpd = socketserver.TCPServer(("", PORT), Handler)
            httpd.serve_forever()
      except KeyboardInterrupt:
            tpc_server.server_close()
            sys.exit()