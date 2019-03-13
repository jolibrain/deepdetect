# -*- coding: utf-8 -*-
"""
DeepDetect Python client

Licence:
Copyright (c) 2015 Emmanuel Benazera, Evgeny BAZAROV <baz.evgenii@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import base64
import os
import re
import warnings

import requests

DD_TIMEOUT = 2000  # seconds, for long blocking training calls, as needed

API_METHODS_URL = {
    "0.1": {
        "info": "/info",
        "services": "/services",
        "train": "/train",
        "predict": "/predict"
    }
}

def _convert_base64(filename):  # return type: Optional[str]
    if os.path.isfile(filename):
        with open(filename, 'rb') as fh:
            data = fh.read()
            x = base64.encodebytes(data)
            return x.decode('ascii').replace('\n', '')
    if re.match('^http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+$', filename):
        result = requests.get(filename)
        if result.status_code != 200:
            warnings.warn("{} returned status {}".format(filename, status))
            return
        x = base64.encodebytes(result.content)
        return x.decode('ascii').replace('\n', '')
    warnings.warn("Unable to understand file type:"
                  " file not found or url not valid", RuntimeWarning)


class DD(object):
    """HTTP requests to the DeepDetect server
    """

    # return types
    RETURN_PYTHON = 0
    RETURN_JSON = 1
    RETURN_NONE = 2

    __HTTP = 0
    __HTTPS = 1

    def __init__(self, host="localhost", port=8080, proto=0, path='', apiversion="0.1"):
        """ DD class constructor
        Parameters:
        host -- the DeepDetect server host
        port -- the DeepDetect server port
        proto -- user http (0,default) or https connection
        """
        self.apiversion = apiversion
        self.__urls = API_METHODS_URL[apiversion]
        self.__host = host
        self.__port = port
        self.__path = path
        self.__proto = proto
        self.__returntype = self.RETURN_PYTHON
        if proto == self.__HTTP:
            self.__ddurl = 'http://%s:%d' % (host, port)
        else:
            self.__ddurl = 'https://%s:%d' % (host, port)
        if path:
            self.__ddurl += path
            
    def set_return_format(self, f):
        assert f == self.RETURN_PYTHON or f == self.RETURN_JSON or f == self.RETURN_NONE
        self.__returntype = f

    def __return_data(self, r):
        if self.__returntype == self.RETURN_PYTHON:
            return r.json()
        elif self.__returntype == self.RETURN_JSON:
            return r.text
        else:
            return None

    def get(self, method, json=None, params=None):
        """GET to DeepDetect server """
        url = self.__ddurl + method
        r = requests.get(url=url, json=json, params=params, timeout=DD_TIMEOUT)
        r.raise_for_status()
        return self.__return_data(r)

    def put(self, method, json=None, params=None):
        """PUT request to DeepDetect server"""
        url = self.__ddurl + method
        r = requests.put(url=url, json=json, params=params, timeout=DD_TIMEOUT)
        r.raise_for_status()
        return self.__return_data(r)

    def post(self, method, json=None, params=None):
        """POST request to DeepDetect server"""
        url = self.__ddurl + method
        r = requests.post(url=url, json=json, params=params, timeout=DD_TIMEOUT)
        r.raise_for_status()
        return self.__return_data(r)

    def delete(self, method, json=None, params=None):
        """DELETE request to DeepDetect server"""
        url = self.__ddurl + method
        r = requests.delete(url=url, json=json, params=params, timeout=DD_TIMEOUT)
        r.raise_for_status()
        return self.__return_data(r)

    # API methods
    def info(self):
        """Info on the DeepDetect server"""
        return self.get(self.__urls["info"])

    # API services
    def put_service(self, sname, model, description, mllib, parameters_input, parameters_mllib, parameters_output, mltype='supervised'):
        """
        Create a service
        Parameters:
        sname -- service name as a resource
        model -- dict with model location and optional templates
        description -- string describing the service
        mllib -- ML library name, e.g. caffe
        parameters_input -- dict of input parameters
        parameters_mllib -- dict ML library parameters
        parameters_output -- dict of output parameters
        """
        data = {"description": description,
                "mllib": mllib,
                "type": mltype,
                "parameters": {"input": parameters_input,
                               "mllib": parameters_mllib,
                               "output": parameters_output},
                "model": model}
        return self.put(self.__urls["services"] + '/%s' % sname, json=data)

    def get_service(self, sname):
        """
        Get information about a service
        Parameters:
        sname -- service name as a resource
        """
        return self.get(self.__urls["services"] + '/%s' % sname)

    def delete_service(self, sname, clear=None):
        """
        Delete a service
        Parameters:
        sname -- service name as a resource
        clear -- 'full','lib' or 'mem', optionally clears model repository data
        """
        lurl = '/%s' % sname
        if clear:
            lurl += '?clear=' + clear
        return self.delete(self.__urls["services"] + lurl)

    # API train
    def post_train(self, sname, data, parameters_input, parameters_mllib, parameters_output, async=True):
        """
        Creates a training job
        Parameters:
        sname -- service name as a resource
        async -- whether to run the job as non-blocking
        data -- array of input data / dataset for training
        parameters_input -- dict of input parameters
        parameters_mllib -- dict ML library parameters
        parameters_output -- dict of output parameters
        """
        data = {"service": sname,
                "async": async,
                "parameters": {"input": parameters_input,
                               "mllib": parameters_mllib,
                               "output": parameters_output},
                "data": data}
        return self.post(self.__urls["train"], json=data)

    def get_train(self, sname, job=1, timeout=0, measure_hist=False):
        """
        Get information on a non-blocking training job
        Parameters:
        sname -- service name as a resource
        job -- job number on the service
        timeout -- timeout before obtaining the job status
        measure_hist -- whether to return the full measure history (e.g. for plotting)
        """
        params = {"service": sname,
                  "job": str(job),
                  "timeout": str(timeout)}
        if measure_hist:
            params["parameters.output.measure_hist"] = measure_hist
        return self.get(self.__urls["train"], params=params)

    def delete_train(self, sname, job=1):
        """
        Kills a non-blocking training job
        Parameters:
        sname -- service name as a resource
        job -- job number on the service
        """
        params = {"service": sname,
                  "job": str(job)}
        return self.delete(self.__urls["train"], params=params)

    # API predict
    def post_predict(self, sname, data, parameters_input, parameters_mllib,
                     parameters_output, use_base64=False):
        """
        Makes prediction from data and model
        Parameters:
        sname -- service name as a resource
        data -- array of data URI to predict from
        parameters_input -- dict of input parameters
        parameters_mllib -- dict ML library parameters
        parameters_output -- dict of output parameters
        """

        if use_base64:
            data = [_convert_base64(d) for d in data]

        data = {"service": sname,
                "parameters": {"input": parameters_input,
                               "mllib": parameters_mllib,
                               "output": parameters_output},
                "data": data}
        return self.post(self.__urls["predict"], json=data)

# test
if __name__ == '__main__':
    dd = DD()
    dd.set_return_format(dd.RETURN_PYTHON)
    inf = dd.info()
    print(inf)
