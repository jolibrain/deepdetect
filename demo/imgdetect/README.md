In order for this demo to work, you'll need to serve the static
index.html file in a webserver, and redirect a request to your
deepdetect server.

## nginx configuration

Here is an nginx configuration example you can use to serve the
index.html file and redirect api request to your deepdetect server:

    server {
      listen 80 default_server;
      listen [::]:80 default_server;
    
      root /home/alx/code/deepdetect/demo/imgdetect;
      index index.html
      server_name _;
    
      location / {
        try_files $uri $uri/ =404;
      }
    
      location /api {
        rewrite ^/api(.*)$  $1  break;
        proxy_pass         http://127.0.0.1:8080;
        proxy_set_header   Host                   $http_host;
        proxy_redirect off;
      }
    }

If you find a 'bad gateway' error after this modification, you can try
to bind deepdetect server to 127.0.0.1 host :

     ./dede -host 127.0.0.1

