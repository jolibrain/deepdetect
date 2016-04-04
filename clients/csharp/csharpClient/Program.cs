using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.IO;
using System.Net;

using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;

using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace csharpClient
{
    class Program
    {
        public static string PostHttp(string url, string body, string contentType)
        {
            HttpWebRequest httpWebRequest = (HttpWebRequest)WebRequest.Create(url);

            httpWebRequest.ContentType = contentType;
            httpWebRequest.Method = "POST";
            httpWebRequest.Timeout = 80000;

            byte[] btBodys = Encoding.UTF8.GetBytes(body);
            httpWebRequest.ContentLength = btBodys.Length;
            httpWebRequest.GetRequestStream().Write(btBodys, 0, btBodys.Length);

            HttpWebResponse httpWebResponse = (HttpWebResponse)httpWebRequest.GetResponse();
            string htmlCharset = "GBK";
            Encoding htmlEncoding = Encoding.GetEncoding(htmlCharset);
            StreamReader streamReader = new StreamReader(httpWebResponse.GetResponseStream(), htmlEncoding);
            string responseContent = streamReader.ReadToEnd();

            httpWebResponse.Close();
            streamReader.Close();
            httpWebRequest.Abort();
            httpWebResponse.Close();

            return responseContent;
        }

        [DataContract]
        [Serializable]
        public class head
        {
            [DataMember]
            public string method { get; set; }
            [DataMember]
            public int time { get; set; }
            [DataMember]
            public string service { get; set; }

        }

        [DataContract]
        [Serializable]
        public class status
        {
            public status()
            {
                this.code = 0;
                this.msg = "";
            }
            [DataMember]
            public int code { get; set; }
            [DataMember]
            public string msg { get; set; }

        }

        [DataContract]
        [Serializable]
        public class probs
        {
            public probs()
            {
                this.prob = 0.0;
                this.cat = "";
            }
            [DataMember]
            public double prob { get; set; }
            [DataMember]
            public string cat { get; set; }

        }

        [DataContract]
        [Serializable]
        public class probslast
        {
            public probslast()
            {
                this.prob = 0.0;
                this.cat = "";
                this.last = true;
            }
            [DataMember]
            public bool last { get; set; }
            [DataMember]
            public double prob { get; set; }
            [DataMember]
            public string cat { get; set; }

        }

        [DataContract]
        [Serializable]
        public class classes
        {
            [DataMember]
            public probs _probs1 { get; set; }
            [DataMember]
            public probs _probs2 { get; set; }
            [DataMember]
            public probslast _probslast { get; set; }


        }

        [DataContract]
        [Serializable]
        public class predictions
        {
            public predictions()
            {
                this._classes = new classes();
                this.uri = "";
                this.loss = 0;
            }

            [DataMember]
            public string uri { get; set; }
            [DataMember]
            public int loss { get; set; }
            [DataMember]
            public classes _classes { get; set; }
        }

        [DataContract]
        [Serializable]
        public class body
        {
            public body()
            {
                this._predictions = new predictions();
            }
            [DataMember]
            public predictions _predictions { get; set; }

        }

        [DataContract]
        [Serializable]
        public class JSonPredictData
        {
            public JSonPredictData()
            {
                this._status = new status();
                this._head = new head();
                this._body = new body();
            }

            [DataMember]
            public body _body { get; set; }

            [DataMember]
            public head _head { get; set; }

            [DataMember]
            public status _status { get; set; }

        }

        public static T ParseFromJson<T>(string szJson)
        {
            T obj = Activator.CreateInstance<T>();
            using (MemoryStream ms = new MemoryStream(Encoding.UTF8.GetBytes(szJson)))
            {
                DataContractJsonSerializer serializer = new DataContractJsonSerializer(obj.GetType());
                return (T)serializer.ReadObject(ms);
            }
        }

        public static string GetJson<T>(T obj)
        {
            DataContractJsonSerializer json = new DataContractJsonSerializer(obj.GetType());
            using (MemoryStream stream = new MemoryStream())
            {
                json.WriteObject(stream, obj);
                string szJson = Encoding.UTF8.GetString(stream.ToArray());
                return szJson;
            }
        }

        private static T Serialization<T>(string obj) where T : class
        {
            using (var mStream = new MemoryStream(Encoding.UTF8.GetBytes(obj)))
            {
                DataContractJsonSerializer serializer = new DataContractJsonSerializer(typeof(T));
                T entity = serializer.ReadObject(mStream) as T;
                return entity;
            }
        }

        static void Main(string[] args)
        {
            string url = "http://123.56.191.136:8080/predict";
            string picUrl = "http://www.deepdetect.com/img/ambulance.jpg";
            string body = "{\"service\":\"imageserv\",\"parameters\":{\"input\":{\"width\":224,\"height\":224},\"output\":{\"best\":3}},\"data\":[\"" + picUrl + "\"]}";

            string result = PostHttp(url, body, "application/x-www-form-urlencoded");

            var jObject = JObject.Parse(result);

            string predictResult = jObject["body"]["predictions"]["classes"][0].ToString();
            //this is the best result, you can also get other prediction result: ...["classes"][1].ToString()...
            

        }
    }
}
