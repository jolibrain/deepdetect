/**
 * DeepDetect
 * Copyright (c) 2017 Emmanuel Benazera
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

#include "simsearch.h"
#include <gtest/gtest.h>
#include <iostream>

using namespace dd;

TEST(faissse, index_search)
{
  std::vector<double> vec1 = { 1.0, 0.0, 0.0, 0.0 };
  std::vector<double> vec2 = { 0.0, 1.0, 0.0, 0.0 };
  std::vector<double> vec3 = { 1.0, 0.0, 1.0, 0.0 };

  int t = 4;
  std::string model_repo = "simsearch";
  mkdir(model_repo.c_str(), 0770);
  FaissSE fse(t, model_repo);
  fse.create_index();                // index creation
  fse.index(URIData("test1"), vec1); // indexing data
  fse.index(URIData("test2"), vec2);
  fse.index(URIData("test3"), vec3);
  std::vector<URIData> uris;
  std::vector<double> distances;
  fse.update_index();
  fse.search(vec1, 3, uris, distances); // searching nearest neighbors
  std::cerr << "search uris size=" << uris.size() << std::endl;
  for (size_t i = 0; i < uris.size(); i++)
    std::cout << uris.at(i)._uri << " / distances=" << distances.at(i)
              << std::endl;
  fse.remove_index();
  rmdir(model_repo.c_str());
}

TEST(faissse, index_search_incr)
{
  std::vector<double> vec1 = { 1.0, 0.0, 0.0, 0.0 };
  std::vector<double> vec2 = { 0.0, 1.0, 0.0, 0.0 };
  std::vector<double> vec3 = { 1.0, 0.0, 1.0, 0.0 };
  std::vector<double> vec4 = { 0.0, 0.0, 5.0, 5.0 };

  int t = 4;
  std::string model_repo = "simsearch";
  mkdir(model_repo.c_str(), 0770);
  FaissSE fse(t, model_repo);
  fse.create_index();
  fse.index(URIData("test1"), vec1);
  fse.index(URIData("test2"), vec2);
  fse.index(URIData("test3"), vec3);
  std::vector<URIData> uris;
  std::vector<double> distances;
  fse.update_index();
  fse.search(vec1, 3, uris, distances);
  std::cerr << "search uris size=" << uris.size() << std::endl;
  for (size_t i = 0; i < uris.size(); i++)
    std::cout << uris.at(i)._uri << " / distances=" << distances.at(i)
              << std::endl;
  uris.clear();
  distances.clear();
  fse.index(URIData("test4"), vec4);
  fse.update_index();
  fse.search(vec4, 3, uris, distances);
  std::cerr << "search uris size=" << uris.size() << std::endl;
  for (size_t i = 0; i < uris.size(); i++)
    std::cout << uris.at(i)._uri << " / distances=" << distances.at(i)
              << std::endl;
  fse.remove_index();
  rmdir(model_repo.c_str());
}
