To use the Python DeepDetect client, install `python-requests`.

Via pip:
```
pip install requests
```

or system-wide, e.g. on Ubuntu:
```
sudo apt-get install python-requests
```

Then install deepdetect python client from deepdetect source code:

```
cd deepdetect_src/clients/python
pip install .
```

or directly with:

```
pip install git+https://github.com/deepdetect.git#egg=dd_client&subdirectory=dd_client
```

The DD client may post images through they base64 representation if those are
not accessible to the deepdetect server. The base64 conversion is supported for
images on the client filesystem or online (http/https), shall the url be only
accessible to the client. A `base64` flag (default to `False`) is available in
the `post_predict` method.
