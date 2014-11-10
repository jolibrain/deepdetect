/**
 * DeepDetect
 * Copyright (c) 2014 Emmanuel Benazera
 * Author: Emmanuel Benazera <beniz@droidnik.fr>
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

#include "deepdetect.h"
#include "commandlineapi.h"
#include "caffelib.h"
#include "imginputfileconn.h"
#include "outputconnectorstrategy.h"

using namespace dd;

int main(int argc, char *argv[])
{
  DeepDetect<CommandLineAPI> dd;
  std::string def = "/home/beniz/projects/deepdetect/datasets/imagenet/models/nin/deploy.prototxt";
  std::string weights = "/home/beniz/projects/deepdetect/datasets/imagenet/models/nin/ilsvrc2011_mine_train_iter_370000.caffemodel";
  //std::string mean = "/home/beniz/projects/deepdetect/datasets/imagenet/ilsvrc2011_mine_mean.binaryproto";
  std::string corresp = "/home/beniz/projects/deepdetect/datasets/imagenet/corresp.txt";
  CaffeModel cmodel(def,weights,corresp);
  //cmodel._mean = mean;
  
  dd.add_service(std::move(MLService<CaffeLib,ImgInputFileConn,SupervisedOutput,CaffeModel>(cmodel)));
  std::cout << "booting\n";
  dd.boot(argc,argv);
}
