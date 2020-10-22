/**
 * DeepDetect
 * Copyright (c) 2020 Jolibrain
 *
 * This file is part of deepdetect.
 *
 * deepdetect is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * deepdetect is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with deepdetect.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "beasthttpjsonapi.h"
#include "utils/utils.hpp"
#include <algorithm>
#include <csignal>
#include <iostream>
#include <thread>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wterminate"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "ext/rmustache/mustache.h"
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/reader.h>
#include <rapidjson/writer.h>
#pragma GCC diagnostic pop
#include "dd_spdlog.h"
#include <gflags/gflags.h>
#include "utils/httpclient.hpp"
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/strand.hpp>
#include <boost/config.hpp>
#include <chrono>
#include <ctime>

DEFINE_string(host, "localhost", "host for running the server");
DEFINE_string(port, "8080", "server port");
DEFINE_int32(nthreads, 10, "number of HTTP server threads");
DEFINE_string(allow_origin, "", "Access-Control-Allow-Origin for the server");

namespace beast = boost::beast;   // from <boost/beast.hpp>
namespace http = beast::http;     // from <boost/beast/http.hpp>
using tcp = boost::asio::ip::tcp; // from <boost/asio/ip/tcp.hpp>
using namespace boost::iostreams;

namespace dd
{

  void mergeJObj(JVal &to, JVal &from, JDoc &jd)
  {
    for (auto fromIt = from.MemberBegin(); fromIt != from.MemberEnd();
         ++fromIt)
      {
        auto toIt = to.FindMember(fromIt->name);
        if (toIt == to.MemberEnd())
          to.AddMember(fromIt->name, fromIt->value, jd.GetAllocator());
        else
          {
            if (fromIt->value.IsArray())
              for (auto arrayIt = fromIt->value.Begin();
                   arrayIt != fromIt->value.End(); ++arrayIt)
                toIt->value.PushBack(*arrayIt, jd.GetAllocator());
            else if (fromIt->value.IsObject())
              mergeJObj(toIt->value, fromIt->value, jd);
            else
              toIt->value = fromIt->value;
          }
      }
  }

  std::string uri_query_to_json(const std::string &req_query)
  {
    if (!req_query.empty())
      {
        JDoc jd;
        JVal jsv(rapidjson::kObjectType);
        jd.SetObject();
        std::vector<std::string> voptions
            = dd::dd_utils::split(req_query, '&');
        for (const std::string o : voptions)
          {
            std::vector<std::string> vopt = dd::dd_utils::split(o, '=');
            if (vopt.size() == 2)
              {
                bool is_word = false;
                for (size_t i = 0; i < vopt.at(1).size(); i++)
                  {
                    if (isalpha(vopt.at(1)[i]))
                      {
                        is_word = true;
                        break;
                      }
                  }
                // we break '.' into JSON sub-objects
                std::vector<std::string> vpt
                    = dd::dd_utils::split(vopt.at(0), '.');
                JVal jobj(rapidjson::kObjectType);
                if (vpt.size() > 1)
                  {
                    bool bt = dd::dd_utils::iequals(vopt.at(1), "true");
                    bool bf = dd::dd_utils::iequals(vopt.at(1), "false");
                    if (is_word && !bt && !bf)
                      {
                        jobj.AddMember(JVal().SetString(vpt.back().c_str(),
                                                        jd.GetAllocator()),
                                       JVal().SetString(vopt.at(1).c_str(),
                                                        jd.GetAllocator()),
                                       jd.GetAllocator());
                      }
                    else if (bt || bf)
                      {
                        jobj.AddMember(JVal().SetString(vpt.back().c_str(),
                                                        jd.GetAllocator()),
                                       JVal(bt ? true : false),
                                       jd.GetAllocator());
                      }
                    else
                      jobj.AddMember(JVal().SetString(vpt.back().c_str(),
                                                      jd.GetAllocator()),
                                     JVal(atoi(vopt.at(1).c_str())),
                                     jd.GetAllocator());
                    for (int b = vpt.size() - 2; b > 0; b--)
                      {
                        JVal jnobj(rapidjson::kObjectType);
                        jobj = jnobj.AddMember(
                            JVal().SetString(vpt.at(b).c_str(),
                                             jd.GetAllocator()),
                            jobj, jd.GetAllocator());
                      }
                    JVal jsv2(rapidjson::kObjectType);
                    jsv2.AddMember(
                        JVal().SetString(vpt.at(0).c_str(), jd.GetAllocator()),
                        jobj, jd.GetAllocator());
                    mergeJObj(jsv, jsv2, jd);
                  }
                else
                  {
                    bool bt = dd::dd_utils::iequals(vopt.at(1), "true");
                    bool bf = dd::dd_utils::iequals(vopt.at(1), "false");
                    if (is_word && !bt && !bf)
                      {
                        jsv.AddMember(JVal().SetString(vopt.at(0).c_str(),
                                                       jd.GetAllocator()),
                                      JVal().SetString(vopt.at(1).c_str(),
                                                       jd.GetAllocator()),
                                      jd.GetAllocator());
                      }
                    else if (bt || bf)
                      {
                        jsv.AddMember(JVal().SetString(vopt.at(0).c_str(),
                                                       jd.GetAllocator()),
                                      JVal(bt ? true : false),
                                      jd.GetAllocator());
                      }
                    else
                      jsv.AddMember(JVal().SetString(vopt.at(0).c_str(),
                                                     jd.GetAllocator()),
                                    JVal(atoi(vopt.at(1).c_str())),
                                    jd.GetAllocator());
                  }
              }
          }
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        jsv.Accept(writer);
        return buffer.GetString();
      }
    else
      return "";
  }
}

template <class Body, class Allocator, class Send>
void handle_request(http::request<Body, http::basic_fields<Allocator>> &&req,
                    Send &&send)
{
  (void)send;
  (void)req;
  // std::string req_path{ req.target };
}

/*
template <class Body, class Allocator, class Send>
void handle_request_model(
    http::request<Body, http::basic_fields<Allocator>> &&req, Send &&send)
{
  // Returns a bad request response
  auto const bad_request = [&req](beast::string_view why) {
    http::response<http::string_body> res{ http::status::bad_request,
                                           req.version() };
    res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
    res.set(http::field::content_type, "application/json");
    res.keep_alive(req.keep_alive());
    res.body() = std::string(why);
    res.prepare_payload();
    return res;
  };

  // Returns a not found response
  auto const not_found = [&req](beast::string_view target) {
    http::response<http::string_body> res{ http::status::not_found,
                                           req.version() };
    res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
    res.set(http::field::content_type, "text/html");
    res.keep_alive(req.keep_alive());
    res.body() = "The resource '" + std::string(target) + "' was not found.";
    res.prepare_payload();
    return res;
  };

  // Returns a server error response
  auto const server_error = [&req](beast::string_view what) {
    http::response<http::string_body> res{ http::status::internal_server_error,
                                           req.version() };
    res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
    res.set(http::field::content_type, "text/html");
    res.keep_alive(req.keep_alive());
    res.body() = "An error occurred: '" + std::string(what) + "'";
    res.prepare_payload();
    return res;
  };

  // Make sure we can handle the method
  if (req.method() != http::verb::get && req.method() != http::verb::head)
    return send(bad_request("Unknown HTTP-method"));

  // Request path must be absolute and not contain "..".
  if (req.target().empty() || req.target()[0] != '/'
      || req.target().find("..") != beast::string_view::npos)
    return send(bad_request("Illegal request-target"));

  // Build the path to the requested file
  std::string path = "/";
  if (req.target().back() == '/')
    path.append("index.html");

  // Attempt to open the file
  beast::error_code ec;
  http::file_body::value_type body;
  body.open(path.c_str(), beast::file_mode::scan, ec);

  // Handle the case where the file doesn't exist
  if (ec == beast::errc::no_such_file_or_directory)
    return send(not_found(req.target()));

  // Handle an unknown error
  if (ec)
    return send(server_error(ec.message()));

  // Cache the size since we need it after the move
  auto const size = body.size();

  // Respond to HEAD request
  if (req.method() == http::verb::head)
    {
      http::response<http::empty_body> res{ http::status::ok, req.version() };
      res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
      res.set(http::field::content_type, mime_type(path));
      res.content_length(size);
      res.keep_alive(req.keep_alive());
      return send(std::move(res));
    }

  // Respond to GET request
  http::response<http::file_body> res{
    std::piecewise_construct, std::make_tuple(std::move(body)),
    std::make_tuple(http::status::ok, req.version())
  };
  res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
  res.set(http::field::content_type, mime_type(path));
  res.content_length(size);
  res.keep_alive(req.keep_alive());
  return send(std::move(res));
}

class APIHandler
{
public:
  APIHandler(dd::HttpJsonAPI *hja) : _hja(hja)
  {
    _logger = spdlog::get("api");
  }

  ~APIHandler()
  {
  }

  / *static boost::network::http::basic_response<std::string> cstock_reply(int
    status, std::string const& content) { using boost::lexical_cast;
    boost::network::http::basic_response<std::string> rep;
    rep.status = status;
    rep.content = content;
    rep.headers.resize(2);
    rep.headers[0].name = "Content-Length";
    rep.headers[0].value = lexical_cast<std::string>(rep.content.size());
    rep.headers[1].name = "Content-Type";
    rep.headers[1].value = "text/html";
    return rep;
    }* /

  void
  fillup_response(http_server::response &response, const JDoc &janswer,
                  std::string &access_log, int &code,
                  std::chrono::time_point<std::chrono::system_clock> tstart,
                  const std::string &encoding = "")
  {
    std::chrono::time_point<std::chrono::system_clock> tstop
        = std::chrono::system_clock::now();
    std::string service;
    if (janswer.HasMember("head"))
      {
        if (janswer["head"].HasMember("service"))
          service = janswer["head"]["service"].GetString();
      }
    if (!service.empty())
      access_log += " " + service;
    code = janswer["status"]["code"].GetInt();
    access_log += " " + std::to_string(code);
    int proctime
        = std::chrono::duration_cast<std::chrono::milliseconds>(tstop - tstart)
              .count();
    access_log += " " + std::to_string(proctime);
    int outcode = code;
    std::string stranswer;
    if (janswer.HasMember(
            "template")) // if output template, fillup with rendered template.
      {
        std::string tpl = janswer["template"].GetString();
        std::stringstream sg;
        mustache::RenderTemplate(tpl, " ", janswer, &sg);
        stranswer = sg.str();
      }
    else
      {
        stranswer = _hja->jrender(janswer);
      }
    if (janswer.HasMember("network"))
      {
        //- grab network call parameters
        std::string url, http_method = "POST",
                         content_type = "Content-Type: application/json";
        if (janswer["network"].HasMember("url"))
          url = janswer["network"]["url"].GetString();
        if (url.empty())
          {
            _logger->error("missing url in network output connector");
            stranswer = _hja->jrender(_hja->dd_bad_request_400());
            code = 400;
          }
        else
          {
            if (janswer["network"].HasMember("http_method"))
              http_method = janswer["network"]["http_method"].GetString();
            if (janswer["network"].HasMember("content_type"))
              content_type = janswer["network"]["content_type"].GetString();

            //- make call
            std::string outstr;
            try
              {
                dd::httpclient::post_call(url, stranswer, http_method, outcode,
                                          outstr, content_type);
                stranswer = outstr;
              }
            catch (std::runtime_error &e)
              {
                _logger->error(e.what());
                _logger->info(stranswer);
                stranswer = _hja->jrender(
                    _hja->dd_output_connector_network_error_1009());
              }
          }
      }
    bool has_gzip = (encoding.find("gzip") != std::string::npos);
    if (!encoding.empty() && has_gzip)
      {
        try
          {
            std::string gzstr;
            filtering_ostream gzout;
            gzout.push(gzip_compressor());
            gzout.push(boost::iostreams::back_inserter(gzstr));
            gzout << stranswer;
            boost::iostreams::close(gzout);
            stranswer = gzstr;
          }
        catch (const std::exception &e)
          {
            _logger->error(e.what());
            outcode = 400;
            stranswer = _hja->jrender(_hja->dd_bad_request_400());
          }
      }
    response = http_server::response::stock_reply(
        http_server::response::status_type(outcode), stranswer);
    response.headers[1].value = "application/json";
    if (!encoding.empty() && has_gzip)
      {
        response.headers.resize(3);
        response.headers[2].name = "Content-Encoding";
        response.headers[2].value = "gzip";
      }
    if (!FLAGS_allow_origin.empty())
      {
        int pos = response.headers.size() + 2;
        response.headers.resize(pos);
        response.headers[pos - 1].name = "Access-Control-Allow-Origin";
        response.headers[pos - 1].value = FLAGS_allow_origin;
      }
    response.status = static_cast<http_server::response::status_type>(code);
  }

  void operator()(http_server::request const &request2,
                  http_server::response &response2)
  {
    // debug
    / *std::cerr << "uri=" << request.destination << std::endl;
    std::cerr << "method=" << request.method << std::endl;
    std::cerr << "source=" << request.source << std::endl;
    std::cerr << "body=" << request.body << std::endl;* /
    // debug

    std::chrono::time_point<std::chrono::system_clock> tstart
        = std::chrono::system_clock::now();
    std::string access_log = request.source + " \"" + request.method + " "
                             + request.destination + "\"";
    int code;
    std::string source = request.source;
    if (source == "::1")
      source = "127.0.0.1";
    uri::uri ur;
    ur << uri::scheme("http") << uri::host(source)
       << uri::path(request.destination);

    std::string req_method = request.method;
    std::string req_path = uri::path(ur);
    std::string req_query = uri::query(ur);
    std::transform(req_path.begin(), req_path.end(), req_path.begin(),
                   ::tolower);
    std::transform(req_query.begin(), req_query.end(), req_query.begin(),
                   ::tolower);
    std::vector<std::string> rscs = dd::dd_utils::split(req_path, '/');
    if (rscs.empty())
      {
        _logger->error("empty resource");
        response = http_server::response::stock_reply(
            http_server::response::not_found,
            _hja->jrender(_hja->dd_not_found_404()));
        return;
      }
    std::string body = request.body;

    // debug
    / *std::cerr << "ur=" << ur << std::endl;
    std::cerr << "path=" << req_path << std::endl;
    std::cerr << "query=" << req_query << std::endl;
    std::cerr << "rscs size=" << rscs.size() << std::endl;
    std::cerr << "path1=" << rscs[1] << std::endl;
    _logger->info("HTTP {} / call / uri={}",req_method,ur);* /
    // debug

    std::string content_encoding;
    std::string accept_encoding;
    for (const auto &header : request.headers)
      {
        if (header.name == "Accept-Encoding")
          accept_encoding = header.value;
        else if (header.name == "Content-Encoding")
          content_encoding = header.value;
      }

    bool encoding_error = false;
    if (!content_encoding.empty())
      {
        if (content_encoding == "gzip")
          {
            if (!body.empty())
              {
                try
                  {
                    std::string gzstr;
                    filtering_ostream gzin;
                    gzin.push(gzip_decompressor());
                    gzin.push(boost::iostreams::back_inserter(gzstr));
                    gzin << body;
                    boost::iostreams::close(gzin);
                    body = gzstr;
                  }
                catch (const std::exception &e)
                  {
                    _logger->error(e.what());
                    fillup_response(response, _hja->dd_bad_request_400(),
                                    access_log, code, tstart);
                    code = 400;
                    encoding_error = true;
                  }
              }
          }
        else
          {
            _logger->error("Unsupported content-encoding: {}",
                           content_encoding);
            fillup_response(response, _hja->dd_bad_request_400(), access_log,
                            code, tstart);
            code = 400;
            encoding_error = true;
          }
      }

    if (!encoding_error)
      {
        if (rscs.at(0) == _rsc_info)
          {
            std::string jstr = dd::uri_query_to_json(req_query);
            fillup_response(response, _hja->info(jstr), access_log, code,
                            tstart, accept_encoding);
          }
        else if (rscs.at(0) == _rsc_services)
          {
            if (rscs.size() < 2)
              {
                fillup_response(response, _hja->dd_bad_request_400(),
                                access_log, code, tstart);
                _logger->error(access_log);
                return;
              }
            std::string sname = rscs.at(1);
            if (req_method == "GET")
              {
                fillup_response(response, _hja->service_status(sname),
                                access_log, code, tstart, accept_encoding);
              }
            else if (req_method == "PUT"
                     || req_method == "POST") // tolerance to using POST
              {
                fillup_response(response, _hja->service_create(sname, body),
                                access_log, code, tstart, accept_encoding);
              }
            else if (req_method == "DELETE")
              {
                // DELETE does not accept body so query options are turned into
                // JSON for internal processing
                std::string jstr = dd::uri_query_to_json(req_query);
                fillup_response(response, _hja->service_delete(sname, jstr),
                                access_log, code, tstart, accept_encoding);
              }
          }
        else if (rscs.at(0) == _rsc_predict)
          {
            if (req_method != "POST")
              {
                fillup_response(response, _hja->dd_bad_request_400(),
                                access_log, code, tstart);
                _logger->error(access_log);
                return;
              }
            fillup_response(response, _hja->service_predict(body), access_log,
                            code, tstart, accept_encoding);
          }
        else if (rscs.at(0) == _rsc_chain)
          {
            if (rscs.size() < 2)
              {
                fillup_response(response, _hja->dd_bad_request_400(),
                                access_log, code, tstart);
                _logger->error(access_log);
                return;
              }
            std::string cname = rscs.at(1);
            if (req_method == "PUT" || req_method == "POST")
              {
                fillup_response(response, _hja->service_chain(cname, body),
                                access_log, code, tstart, accept_encoding);
              }
          }
        else if (rscs.at(0) == _rsc_train)
          {
            if (req_method == "GET")
              {
                std::string jstr = dd::uri_query_to_json(req_query);
                fillup_response(response, _hja->service_train_status(jstr),
                                access_log, code, tstart, accept_encoding);
              }
            else if (req_method == "PUT" || req_method == "POST")
              {
                fillup_response(response, _hja->service_train(body),
                                access_log, code, tstart, accept_encoding);
              }
            else if (req_method == "DELETE")
              {
                // DELETE does not accept body so query options are turned into
                // JSON for internal processing
                std::string jstr = dd::uri_query_to_json(req_query);
                fillup_response(response, _hja->service_train_delete(jstr),
                                access_log, code, tstart);
              }
          }
        else
          {
            _logger->error("Unknown Service={}", rscs.at(0));
            response = http_server::response::stock_reply(
                http_server::response::not_found,
                _hja->jrender(_hja->dd_not_found_404()));
            code = 404;
          }
      }
    if (code == 200 || code == 201)
      _logger->info(access_log);
    else
      _logger->error(access_log);
  }

  void log(http_server::string_type const &info)
  {
    _logger->error(info);
  }

  dd::HttpJsonAPI *_hja;
  std::string _rsc_info = "info";
  std::string _rsc_services = "services";
  std::string _rsc_predict = "predict";
  std::string _rsc_train = "train";
  std::string _rsc_chain = "chain";
  std::shared_ptr<spdlog::logger> _logger;
};
*/

class session : public std::enable_shared_from_this<session>
{
  struct send_lambda
  {
    session &self_;

    explicit send_lambda(session &self) : self_(self)
    {
    }

    template <bool isRequest, class Body, class Fields>
    void operator()(http::message<isRequest, Body, Fields> &&msg) const
    {
      auto sp = std::make_shared<http::message<isRequest, Body, Fields>>(
          std::move(msg));
      // Store a type-erased version of the shared
      // pointer in the class to keep it alive.
      self_.res_ = sp;
      http::async_write(self_.stream_, *sp,
                        beast::bind_front_handler(&session::on_write,
                                                  self_.shared_from_this(),
                                                  sp->need_eof()));
    }
  };

  beast::tcp_stream stream_;
  beast::flat_buffer buffer_;
  http::request<http::string_body> req_;
  std::shared_ptr<void> res_;
  send_lambda lambda_;

public:
  session(tcp::socket &&socket) : stream_(std::move(socket)), lambda_(*this)
  {
  }

  void run()
  {
    boost::asio::dispatch(
        stream_.get_executor(),
        beast::bind_front_handler(&session::do_read, shared_from_this()));
  }

  void do_read()
  {
    req_ = {};
    stream_.expires_after(std::chrono::seconds(30));
    http::async_read(
        stream_, buffer_, req_,
        beast::bind_front_handler(&session::on_read, shared_from_this()));
  }

  void on_read(beast::error_code ec, std::size_t bytes_transferred)
  {
    boost::ignore_unused(bytes_transferred);
    if (ec)
      {
        if (ec == http::error::end_of_stream)
          do_close();
        else
          std::cerr << "read fail: " << ec.message() << std::endl;
        return;
      }
    handle_request(std::move(req_), lambda_);
  }

  void on_write(bool close, beast::error_code ec,
                std::size_t bytes_transferred)
  {
    boost::ignore_unused(bytes_transferred);

    if (ec)
      {
        std::cerr << "write fail: " << ec.message() << std::endl;
        return;
      }

    if (close)
      return do_close();

    res_ = nullptr;
    do_read();
  }

  void do_close()
  {
    beast::error_code ec;
    stream_.socket().shutdown(tcp::socket::shutdown_send, ec);
  }
};

class listener : public std::enable_shared_from_this<listener>
{
  boost::asio::io_context &ioc_;
  tcp::acceptor acceptor_;

public:
  listener(boost::asio::io_context &ioc, tcp::endpoint endpoint)
      : ioc_(ioc), acceptor_(boost::asio::make_strand(ioc))
  {
    acceptor_.open(endpoint.protocol());
    acceptor_.set_option(boost::asio::socket_base::reuse_address(true));
    acceptor_.set_option(boost::asio::socket_base::linger(false, 10));
    acceptor_.bind(endpoint);
    acceptor_.listen(boost::asio::socket_base::max_listen_connections);
  }

  void run()
  {
    do_accept();
  }

private:
  void do_accept()
  {
    acceptor_.async_accept(
        boost::asio::make_strand(ioc_),
        beast::bind_front_handler(&listener::on_accept, shared_from_this()));
  }

  void on_accept(beast::error_code ec, tcp::socket socket)
  {
    if (ec)
      std::cerr << "fail to accept new connection: " << ec.message()
                << std::endl;
    else
      std::make_shared<session>(std::move(socket))->run();
    do_accept();
  }
};

namespace dd
{
  volatile std::sig_atomic_t _sigstatus;

  boost::asio::io_context *_ioc = nullptr;
  std::thread *_main_thread = nullptr;

  BeastHttpJsonAPI::BeastHttpJsonAPI() : JsonAPI()
  {
  }

  BeastHttpJsonAPI::~BeastHttpJsonAPI()
  {
  }

  int BeastHttpJsonAPI::start_server(const std::string &shost,
                                     const std::string &sport,
                                     const int &nthreads)
  {
    try
      {
        _logger->info("Running DeepDetect HTTP server on {}:{}", shost, sport);
        if (!FLAGS_allow_origin.empty())
          _logger->info("Allowing origin from {}", FLAGS_allow_origin);
        auto const address = boost::asio::ip::make_address(shost);
        auto const port = static_cast<unsigned short>(std::stoi(sport));

        _ioc = new boost::asio::io_context(nthreads);
        std::make_shared<listener>(*_ioc, tcp::endpoint{ address, port })
            ->run();
        std::vector<std::thread> v;
        v.reserve(nthreads - 1);

        for (auto i = nthreads - 1; i > 0; --i)
          v.emplace_back([&] { _ioc->run(); });

        _ioc->run();
        return 0;
      }
    catch (const std::exception &e)
      {
        _logger->error(e.what());
        return 1;
      }
  }

  void BeastHttpJsonAPI::stop_server()
  {
    _logger->info("stopping HTTP server");
    if (_ioc != nullptr)
      {
        try
          {
            _ioc->stop();
            delete _ioc;
          }
        catch (std::exception &e)
          {
            _logger->error(e.what());
          }
      }
  }

  void BeastHttpJsonAPI::terminate(int param)
  {
    (void)param;
  }

  int BeastHttpJsonAPI::boot(int argc, char *argv[])
  {
    google::ParseCommandLineFlags(&argc, &argv, true);
    std::signal(SIGINT, terminate);
    JsonAPI::boot(argc, argv);
    return start_server(FLAGS_host, FLAGS_port, FLAGS_nthreads);
  }
}
