#include "deepdetect.h"
#include "jsonapi.h"
#include <gtest/gtest.h>
#include <stdio.h>
#include <iostream>

#include "graph.h"
#include "torch/torch.h"
#include <sstream>
#include <string>

using namespace dd;


static std::string ok_str = "{\"status\":{\"code\":200,\"msg\":\"OK\"}}";
static std::string created_str = "{\"status\":{\"code\":201,\"msg\":\"Created\"}}";
static std::string bad_param_str = "{\"status\":{\"code\":400,\"msg\":\"BadRequest\"}}";
static std::string not_found_str = "{\"status\":{\"code\":404,\"msg\":\"NotFound\"}}";

static std::string gpuid = "0"; // change as needed


static std::string sinus = "../examples/all/sinus/";

static std::string iterations_lstm_gpu = "1000";
static std::string iterations_lstm_cpu = "20";


TEST(graphapi, proto_read)
{
	CaffeToTorch ctt("../../examples/graph/recurrent.prototxt");
	std::stringstream ss;
	ctt.todot(ss);
	int cmp = ss.str().compare(R"DOT(digraph G {
0[label="data { 1 , 50 , 9 }", shape=ellipse];
1[label="LSTM0", shape=box];
2[label="LSTM_0 { 1 , 50 , 50 }", shape=ellipse];
3[label="LSTM1", shape=box];
4[label="LSTM_1 { 1 , 50 , 50 }", shape=ellipse];
5[label="affine_2", shape=box];
6[label="rnn_pred { 1 , 50 , 3 }", shape=ellipse];
0->1 ;
1->2 ;
2->3 ;
3->4 ;
4->5 ;
5->6 ;
}
)DOT");
	ASSERT_EQ(cmp,0);
}


TEST(graphapi, compute_lstm)
{
	CaffeToTorch ctt("../../examples/graph/recurrent.prototxt");
	torch::Tensor input = torch::randn({2,10,9});
	torch::Tensor output = ctt.forward(input);
	ASSERT_EQ(output.size(0),2);
	ASSERT_EQ(output.size(1),10);
	ASSERT_EQ(output.size(2),3);
}

TEST(graphapi, simple_cuda)
{
	CaffeToTorch ctt("../../examples/graph/recurrent.prototxt");
	ctt.to(torch::Device(c10::DeviceType::CUDA, 0));
	torch::Tensor input = torch::randn({2,50,9}).to(torch::Device(c10::DeviceType::CUDA, 0));
	torch::Tensor output = ctt.forward(input);
	ASSERT_EQ(output.size(0),2);
	ASSERT_EQ(output.size(1),50);
	ASSERT_EQ(output.size(2),3);
}

TEST(graphapi, simple_lstm_train)
{
	CaffeToTorch ctt("../../examples/graph/recurrent.prototxt");
	torch::Tensor input = torch::randn({2,50,9});
	torch::Tensor target = torch::randn({2,50,3});
	torch::Tensor output = ctt.forward(input);
	ASSERT_EQ(output.size(0),2);
	ASSERT_EQ(output.size(1),50);
	ASSERT_EQ(output.size(2),3);

	std::unique_ptr<torch::optim::Optimizer> optimizer;
	double base_lr = 0.0001;
	optimizer = std::unique_ptr<torch::optim::Optimizer>
	  (new torch::optim::RMSprop(ctt.parameters(), torch::optim::RMSpropOptions(base_lr)));

	optimizer->zero_grad();
	ctt.train();

	torch::Tensor loss;
	double l0;
	double l9;

	auto tstart = std::chrono::system_clock::now();
	for (unsigned int i = 0; i< std::stoi(iterations_lstm_cpu); ++i)
		{
		  output = ctt.forward(input);
			loss = torch::l1_loss(output,target);
			if (i==0)
				l0 = loss.item<double>();
			else if (i==9)
				l9 = loss.item<double>();
			std::cout << "loss: " << loss.item<double>() << std::endl;
			loss.backward();
			optimizer->step();
			optimizer->zero_grad();
		}
	auto tstop = std::chrono::system_clock::now();
	std::cout << "optimization duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(tstop - tstart).count() << std::endl;;
	ASSERT_TRUE(l9<l0);
}

#ifndef CPU_ONLY
TEST(graphapi, simple_lstm_train_gpu)
{
	// torch model allocated to cpu with size of 1,50,9 as in prototxt
	CaffeToTorch ctt("../../examples/graph/recurrent.prototxt");
	// moving previous torch model to gpu
	ctt.to(torch::Device(c10::DeviceType::CUDA, 0));

	auto tstart = std::chrono::system_clock::now();
	torch::Tensor input = torch::randn({2,10,9}).to(torch::Device(c10::DeviceType::CUDA, 0));
	torch::Tensor target = torch::randn({2,10,3}).to(torch::Device(c10::DeviceType::CUDA, 0));
	auto tstop = std::chrono::system_clock::now();
	std::cout << "data transfer duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(tstop - tstart).count() << std::endl;
	//below forward is forced to reallocate all model as datasize has changed from 50 to 10
	tstart = std::chrono::system_clock::now();
	torch::Tensor output = ctt.forward(input);
	tstop = std::chrono::system_clock::now();
	std::cout << "realloc + push model on GPU + 1 forward  duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(tstop - tstart).count() << std::endl;
	// it is reallocated and moved to gpu automatically :)
	ASSERT_EQ(output.size(0),2);
	ASSERT_EQ(output.size(1),10);
	ASSERT_EQ(output.size(2),3);

	std::unique_ptr<torch::optim::Optimizer> optimizer;
	double base_lr = 0.0001;
	optimizer = std::unique_ptr<torch::optim::Optimizer>
		(new torch::optim::RMSprop(ctt.parameters(), torch::optim::RMSpropOptions(base_lr)));
	optimizer->zero_grad();
	ctt.train();

	torch::Tensor loss;
	//	loss.to(torch::Device(c10::DeviceType::CUDA, 0));
	double l0;
	double l9;

	tstart = std::chrono::system_clock::now();
	for (unsigned int i = 0; i< std::stoi(iterations_lstm_gpu); ++i)
		{
			output = ctt.forward(input);
			loss = torch::l1_loss(output,target);
			if (i==0)
				l0 = loss.item<double>();
			else if (i==9)
				l9 = loss.item<double>();
			std::cout << "loss: " << loss.item<double>() << std::endl;
			loss.backward();
			optimizer->step();
			optimizer->zero_grad();
		}
	tstop = std::chrono::system_clock::now();
	std::cout << "optimizaton duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(tstop - tstart).count() << std::endl;
	ASSERT_TRUE(l9<l0);
}
#endif

TEST(graphapi, intermediate_lstm_train)
{
	CaffeToTorch ctt("../../examples/graph/recurrent.prototxt");
	CSVTSTorchInputFileConn inputc;
	inputc._train = true;
	inputc._logger = spdlog::stdout_logger_mt("ut-graphapi");

	APIData ad;
	APIData ad_param;
	APIData ad_input;
	std::vector<std::string> data;
	data.push_back("../examples/all/sinus/train");
	data.push_back("../examples/all/sinus/test");
	ad.add("data",data);
	ad_input.add("connector",std::string("csvts"));
	std::vector<std::string> outputs;
	outputs.push_back("output");
	ad_input.add("label",outputs);
	ad_input.add("separator",std::string(","));
	ad_input.add("timesteps",200);
	ad_param.add("input", ad_input);
	ad.add("parameters",ad_param);

	inputc.transform(ad);

	ASSERT_EQ(inputc._dataset.cache_size(), 485);
	ASSERT_EQ(inputc._test_dataset.cache_size(), 10);

	auto batchoptional = inputc._dataset.get_batch({inputc._dataset.cache_size()});
	TorchBatch batch = batchoptional.value();
	torch::Tensor td = batch.data[0];
	torch::Tensor tt = batch.target[0];

	ctt.finalize(td.sizes());

	ASSERT_TRUE(td.sizes() ==  std::vector<int64_t>({485,200,1}));
	ASSERT_TRUE(tt.sizes() == std::vector<int64_t>({485,200,1}));

	std::unique_ptr<torch::optim::Optimizer> optimizer;
	double base_lr = 0.1;
	optimizer = std::unique_ptr<torch::optim::Optimizer>
	  (new torch::optim::RMSprop(ctt.parameters(), torch::optim::RMSpropOptions(base_lr)));
	optimizer->zero_grad();
	ctt.train();

	torch::Tensor loss;
	double l0;
	double l9;

	inputc._dataset.reset();
	auto dataloader =
	  torch::data::make_data_loader(std::move(inputc._dataset),
									torch::data::DataLoaderOptions(100));

	auto tstart = std::chrono::system_clock::now();
	for (unsigned int i = 0; i< std::stoi(iterations_lstm_cpu); ++i)
	  {
		int nbatches = 0;
		double train_loss = 0;
		for (TorchBatch batch : *dataloader)
		  {
			torch::Tensor input = batch.data[0]; //input is only one tensor, hence [0]
			ASSERT_TRUE(input.size(0) <= 100);
			ASSERT_EQ(input.size(1), 200);
			ASSERT_EQ(input.size(2), 1);
			torch::Tensor target = batch.target[0];
			torch::Tensor output = ctt.forward(input);
			loss = torch::l1_loss(output,target);
			train_loss += loss.item<double>();
			nbatches++;
			loss.backward();
			std::cout << "minibatch loss : " << loss.item<double>() << std::endl;
			optimizer->step();
			optimizer->zero_grad();
		  }
		std::cout << "epoch loss: " << train_loss/nbatches << std::endl;
		if (i == 0)
		  l0 = train_loss/nbatches;
		else if (i==9)
		  l9 = train_loss/nbatches;
	 	}
	 auto tstop = std::chrono::system_clock::now();
	 std::cout << "optimization duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(tstop - tstart).count() << std::endl;;
	 ASSERT_TRUE(l9<l0);
}


TEST(graphapi, complete_lstm_train)
{
  // create service
  JsonAPI japi;
  std::string csvts_data = sinus + "train";
  std::string csvts_test = sinus +"test";
  std::string csvts_predict = sinus +"predict";
  std::string csvts_repo = "csvts";
  mkdir(csvts_repo.c_str(),0777);
  std::string sname = "my_service_csvts";
  std::string jstr = "{\"mllib\":\"torch\",\"description\":\"my ts regressor\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  csvts_repo+"\"},\"parameters\":{\"input\":{\"connector\":\"csvts\",\"label\":[\"output\"]},\"mllib\":{\"template\":\"recurrent\",\"layers\":[\"L10\",\"L10\"],\"dropout\":[0.0,0.0,0.0],\"regression\":true,\"sl1sigma\":100.0,\"loss\":\"L1\"}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"shuffle\":true,\"separator\":\",\",\"scale\":true,\"timesteps\":200,\"label\":[\"output\"]},\"mllib\":{\"gpu\":false,\"solver\":{\"iterations\":" + iterations_lstm_cpu + ",\"test_interval\":10,\"base_lr\":0.1,\"snapshot\":500,\"test_initialization\":false},\"net\":{\"batch_size\":100,\"test_batch_size\":10}},\"output\":{\"measure\":[\"L1\",\"L2\"]}},\"data\":[\"" + csvts_data+"\",\""+csvts_test+"\"]}";
  std::cerr << "jtrainstr=" << jtrainstr << std::endl;
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(201,jd["status"]["code"].GetInt());
  ASSERT_EQ("Created",jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("head"));
  ASSERT_EQ("/train",jd["head"]["method"]);
  ASSERT_TRUE(jd["head"]["time"].GetDouble() >= 0);
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"]["measure"].HasMember("train_loss"));
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0);
  ASSERT_TRUE(jd["body"]["measure"].HasMember("L1_mean_error"));
  ASSERT_TRUE(jd["body"]["measure"]["L1_max_error_0"].GetDouble() > 0.0);
  ASSERT_TRUE(jd["body"]["parameters"]["input"].HasMember("max_vals"));
  ASSERT_TRUE(jd["body"]["parameters"]["input"].HasMember("min_vals"));

  std::string str_min_vals = japi.jrender(jd["body"]["parameters"]["input"]["min_vals"]);
  std::string str_max_vals = japi.jrender(jd["body"]["parameters"]["input"]["max_vals"]);

  //  predict
  std::string jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"timesteps\":20,\"connector\":\"csvts\",\"scale\":true,\"continuation\":true,\"label\":[\"output\"],\"min_vals\":" + str_min_vals + ",\"max_vals\":" + str_max_vals + "},\"output\":{}},\"data\":[\"" + csvts_predict + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  std::string uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("../examples/all/sinus/predict/seq_2.csv #0_19",uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"][0]["out"][0].GetDouble() >= -1.0);

  //  remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
  rmdir(csvts_repo.c_str());

}

#ifndef CPU_ONLY
TEST(graphapi, complete_lstm_train_gpu)
{
  // create service
  JsonAPI japi;
  std::string csvts_data = sinus + "train";
  std::string csvts_test = sinus +"test";
  std::string csvts_predict = sinus +"predict";
  std::string csvts_repo = "csvts";
  mkdir(csvts_repo.c_str(),0777);
  std::string sname = "my_service_csvts";
  std::string jstr = "{\"mllib\":\"torch\",\"description\":\"my ts regressor\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  csvts_repo+"\"},\"parameters\":{\"input\":{\"connector\":\"csvts\",\"label\":[\"output\"]},\"mllib\":{\"template\":\"recurrent\",\"layers\":[\"L10\",\"L10\"],\"dropout\":[0.0,0.0,0.0],\"regression\":true,\"sl1sigma\":100.0,\"loss\":\"L1\",\"gpu\":true,\"gpuid\":"+gpuid+"}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"shuffle\":true,\"separator\":\",\",\"scale\":true,\"timesteps\":200,\"label\":[\"output\"]},\"mllib\":{\"gpu\":false,\"solver\":{\"iterations\":" + iterations_lstm_gpu + ",\"test_interval\":500,\"base_lr\":0.1,\"snapshot\":500,\"test_initialization\":false},\"net\":{\"batch_size\":100,\"test_batch_size\":10}},\"output\":{\"measure\":[\"L1\",\"L2\"]}},\"data\":[\"" + csvts_data+"\",\""+csvts_test+"\"]}";
  std::cerr << "jtrainstr=" << jtrainstr << std::endl;
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(201,jd["status"]["code"].GetInt());
  ASSERT_EQ("Created",jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("head"));
  ASSERT_EQ("/train",jd["head"]["method"]);
  ASSERT_TRUE(jd["head"]["time"].GetDouble() >= 0);
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"]["measure"].HasMember("train_loss"));
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0);
  ASSERT_TRUE(jd["body"]["measure"].HasMember("L1_mean_error"));
  ASSERT_TRUE(jd["body"]["measure"]["L1_max_error_0"].GetDouble() > 0.0);
  ASSERT_TRUE(jd["body"]["parameters"]["input"].HasMember("max_vals"));
  ASSERT_TRUE(jd["body"]["parameters"]["input"].HasMember("min_vals"));

  std::string str_min_vals = japi.jrender(jd["body"]["parameters"]["input"]["min_vals"]);
  std::string str_max_vals = japi.jrender(jd["body"]["parameters"]["input"]["max_vals"]);

  //  predict
  std::string jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"timesteps\":20,\"connector\":\"csvts\",\"scale\":true,\"label\":[\"output\"],\"continuation\":true,\"min_vals\":" + str_min_vals + ",\"max_vals\":" + str_max_vals + "},\"output\":{}},\"data\":[\"" + csvts_predict + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  std::string uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("../examples/all/sinus/predict/seq_2.csv #0_19",uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"][0]["out"][0].GetDouble() >= -1.0);

  //  remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
  rmdir(csvts_repo.c_str());

}
#endif
