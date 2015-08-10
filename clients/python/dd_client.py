"""
DeepDetect Python client

Provides:
TODO

Licence:


"""

import ConfigParser
import urllib2
import httplib
import os.path
import json
import uuid
import datetime

VERBOSE=False
DD_TIMEOUT = 2000 # seconds, for long blocking training alls, as needed

def LOG(msg):
    """Output a log message."""
    # XXX: may want to use python log manager classes instead of this stupid print
    if VERBOSE:
        msg = str(datetime.datetime.now()) + ' ' + msg
        print msg

### Exception classes :

class DDCommunicationError(Exception):
    def __init__(self, url, http_method,  headers, body, response=None):
        self.msg = """DeepDetect Communication Error"""
        self.http_method = http_method
        self.req_headers = headers
        self.req_body = body
        self.url = url
        self.res_headers = None
        if response is not None:
            self.res_headers = response.get_info()
            
    def __str__(self):
        msg = "%s %s\n"%(str(self.http_method),str(self.url))
        for h,v in self.req_headers.iteritems():
            msg += "%s:%s\n"%(h,v)
        msg += "\n"
        if self.req_body is not None:
            msg += str(self.req_body)[:100]
        msg += "\n"
        msg += "--\n"
        msg += str(self.res_headers)
        msg += "\n"
        return msg

    class DDDataError(Exception):
        def __init__(self, url, http_method,  headers, body, data=None):
            self.msg = "DeepDetect Data Error"
            self.http_method = http_method
            self.req_headers = headers
            self.req_body = body
            self.url = url
            self.data = data

        def __str__(self):
            msg = "%s %s\n"%(str(self.http_method),str(self.url))
            if self.data is not None:
                msg += str(self.data)[:100]
                msg += "\n"
            return msg
            for h,v in self.req_headers.iteritems():
                msg += "%s:%s\n"%(h,v)
            msg += "\n"
            if self.req_body is not None:
                msg += str(self.req_body)
            msg += "\n"
            msg += "--\n"
            msg += str(self.data)
            msg += "\n"
            return msg

# hack for wrongly encoded json
# input : s : str object 
# output ; unicode object
def hack_decode(s):
    if isinstance(s, unicode):
        return s
    while True:
        try:
            return s.decode('utf-8')
        except UnicodeDecodeError, e :
            s = s[:e.start]+'?'+s[e.end:]


API_METHODS_URL = {
    "0.1" : {
        "info":"/info",
        "services":"/services",
        "train":"/train",
        "predict":"/predict"
    }
}

class DD(object):
    """HTTP requests to the DeepDetect server

    """

    # return types
    RETURN_PYTHON=0
    RETURN_JSON=1
    RETURN_NONE=2

    __HTTP=0
    __HTTPS=1

    def __init__(self,host="localhost",port=8080,proto=0,apiversion="0.1"):
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
        self.__proto = proto
        self.__returntype=self.RETURN_PYTHON
        if proto == self.__HTTP:
            self.__ddurl='http://%s:%d'%(host,port)
        else:
            self.__ddurl='https://%s:%d'%(host,port)


    def set_return_format(self,f):
        assert f == self.RETURN_PYTHON or f == self.RETURN_JSON or f == self.RETURN_NONE
        self.__returntype = f

    def __return_format(self,js):
        if self.__returntype == self.RETURN_PYTHON:
            return json.loads(hack_decode(js))
        elif self.__returntype == self.RETURN_JSON:
            return js
        else:
            return None

    def get(self,method,args=None):
        """ GET to DeepDetect server """
        u = self.__ddurl
        u += method
        headers = {}
        if args is not None:
            sep = "?"
            for arg,argv in args.iteritems():
                u += sep
                sep = "&"
                u += urllib2.quote(arg)
                u += '='
                if argv is not None:
                    u += urllib2.quote(argv)
                    
        LOG("GET %s"%u)
        response = None
        try:
            req = urllib2.Request(u)
            response = urllib2.urlopen(req, timeout=DD_TIMEOUT)
            jsonresponse=response.read()
        except:
            raise DDCommunicationError(u,"GET",headers,None,response)
        LOG(jsonresponse)
        try:
            return self.__return_format(jsonresponse)
        except:
            raise DDDataError(u,"GET",headers,None,jsonresponse)

    def put(self, method, body):
        """PUT request to DeepDetect server"""

        LOG("PUT %s\n%s"%(method,body))
        r = None
        u = ""
        headers = {}
        try:
            u = self.__ddurl + method
            if self.__proto == self.__HTTP:
            #    u = "http://%s:%s%s"%(self.__host,self.__port,method)
                c=httplib.HTTPConnection(self.__host,self.__port, timeout=DD_TIMEOUT)
            else:
            #    u = "https://%s:%s%s"%(self.__host,self.__port,method)
                c=httplib.HTTPSConnection(self.__host,self.__port, timeout=DD_TIMEOUT)
            c.request('PUT',method,body,headers)
            r = c.getresponse()
            data = r.read()
        except:
            raise DDCommunicationError(u,"PUT",headers,body,r)
        LOG(data)
        try:
            return self.__return_format(data)
        except:
            raise DDDataError(u,"PUT",headers,body,data)

    def post(self,method,body):
        """POST request to DeepDetect server"""
        
        r = None
        u = ""
        headers = {}
        try:
            u = self.__ddurl + method
            if self.__proto == self.__HTTP:
                LOG("curl -X POST 'http://%s:%s%s' -d '%s'"%(self.__host,
                                                             self.__port,
                                                             method,
                                                             body))
                c=httplib.HTTPConnection(self.__host,self.__port,timeout=DD_TIMEOUT)
            else:
                LOG("curl -k -X POST 'https://%s:%s%s' -d '%s'"%(self.__host,
                                                                 self.__port,
                                                                 method,
                                                                 body))
                c=httplib.HTTPSConnection(self.__host,self.__port, timeout=DD_TIMEOUT)
            c.request('POST',method,body,headers)
            r = c.getresponse()
            data = r.read()
            
        except:
            raise DDCommunicationError(u,"POST",headers,body,r)

        # LOG(data)
        try:
            return self.__return_format(data)
        except:
            import traceback
            print traceback.format_exc()

            raise DDDataError(u,"POST",headers,body,data)
        
    def delete(self, method):
        """DELETE request to DeepDetect server"""

        LOG("DELETE %s"%(method))
        r = None
        u = ""
        body = ""
        headers = {}
        try:
            u = self.__ddurl + method
            if self.__proto == self.__HTTP:
                c=httplib.HTTPConnection(self.__host,self.__port, timeout=DD_TIMEOUT)
            else:
                c=httplib.HTTPSConnection(self.__host,self.__port, timeout=DD_TIMEOUT)
            c.request('DELETE',method,body,headers)
            r = c.getresponse()
            data = r.read()
        except:
            raise DDCommunicationError(u,"DELETE",headers,None,r)

        LOG(data)
        try:
            return self.__return_format(data)
        except:
            raise DDDataError(u,"DELETE",headers,None,data)


    # API methods

    def info(self):
        """Info on the DeepDetect server"""
        return self.get(self.__urls["info"])


    # - PUT services
    # - GET services
    # - DELETE services
    def put_service(self,sname,model,description,mllib,parameters_input,parameters_mllib,parameters_output):
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
        body={"description":description,"mllib":mllib,"type":"supervised",
              "parameters":{"input":parameters_input,"mllib":parameters_mllib,"output":parameters_output},
              "model":model}
        return self.put(self.__urls["services"] + '/%s'%sname,json.dumps(body))

    def get_services(self,sname):
        """
        Get information about a service
        Parameters:
        sname -- service name as a resource
        """
        return self.get(self.__urls["services"] + '/%s'%sname)

    def delete_service(self,sname,clear=None):
        """
        Delete a service
        Parameters:
        sname -- service name as a resource
        clear -- 'full','lib' or 'mem', optionally clears model repository data
        """
        qs = self.__urls["services"] + '/%s'%sname
        if clear:
            qs += '?clear=' + clear
        return self.delete(qs)


    # PUT/POST /train
    # GET /train
    # DELETE /train

    def post_train(self,sname,data,parameters_input,parameters_mllib,parameters_output,async=True):
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
        body={"service":sname,"async":async,
              "parameters":{"input":parameters_input,"mllib":parameters_mllib,"output":parameters_output},
              "data":data}
        return self.post(self.__urls["train"],json.dumps(body))

    def get_train(self,sname,job=1,timeout=0,measure_hist=False):
        """
        Get information on a non-blocking training job
        Parameters:
        sname -- service name as a resource
        job -- job number on the service
        timeout -- timeout before obtaining the job status
        measure_hist -- whether to return the full measure history (e.g. for plotting)
        """
        qs=self.__urls["train"] + "?service=" + sname + "&job=" + str(job) + "&timeout=" + str(timeout)
        if measure_hist:
            qs += "&parameters.output.measure_hist=true"
        return self.get(qs)

    def delete_train(self,sname,job=1):
        """
        Kills a non-blocking training job
        Parameters:
        sname -- service name as a resource
        job -- job number on the service
        """
        qs=self.__urls["train"] + "?service=" + sname + "&job=" + str(job)
        return self.delete(qs)


    # POST /predict

    def post_predict(self,sname,data,parameters_input,parameters_output):
        """
        Makes prediction from data and model
        Parameters:
        sname -- service name as a resource
        data -- array of data URI to predict from
        parameters_input -- dict of input parameters
        parameters_output -- dict of output parameters
        """
        body={"service":sname,
              "parameters":{"input":parameters_input,"output":parameters_output},
              "data":data}
        return self.post(self.__urls["predict"],json.dumps(body))
    
# test
if __name__ == '__main__':
    dd = DD()
    dd.set_return_format(dd.RETURN_PYTHON)
    inf = dd.info()
    print inf
