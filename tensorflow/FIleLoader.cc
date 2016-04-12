#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
# include <iostream>

using namespace tensorflow;


int main(int argc,char* argv[]){
   // initializing a tensorflow session
	Session* session;
	
	// trying to extract the repository name otherwise using default naeme
	try{
	std::string  Repository = argv[1];
	}
	catch (int e){
		std:: string Repository = "./model/"
	}

	try{
	std::string InputName = argv[2];
	}
	catch{
		std::string InputName = "input";
	} 

	try{
	std::string OutputName = argv[3];
	}
	catch{
		std::string OutputName = "output";
	}

   // Linitializing the session input and output

	Status  status = NewSession(SessionOption(),& session );

	if (!status.ok()){
		std::cout << status.ToString() << std::endl;

		return 1;
	}

	// initializing the graph 
	GraphDef graph_def;

	status = ReadBinaryProto(Env::Default(),Repository,&graph_def)
	if (!status.ok()){
		std::cout << status.ToString()<<std::endl;
		return 1;
	}

	// connecting the input and the output nodes of the graph.
	std:: string file_name = "ext/tensorflow/tensorflow/examples/label_image/data/grace_hopper.jpg";



	// parameter for input image

   int inputHeight = 299;
   int InputWidth = 299;
   float ImageMean = 128;
   float ImageStd = 128;

   // creating a new NOde to fit the requirment of the defined tensorflow graph


    std::string input_name = "file_reader";
    std::string output_name = "normalized";
    Node* file_reader =tensorflow::ops::ReadFile(tensorflow::ops::Const(file_name, b.opts()),
                                b.opts().WithName(input_name));
}