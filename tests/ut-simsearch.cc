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

TEST(annoyse, index_search)
{
  std::vector<double> vec1 = { 1.0, 0.0, 0.0, 0.0 };
  std::vector<double> vec2 = { 0.0, 1.0, 0.0, 0.0 };
  std::vector<double> vec3 = { 1.0, 0.0, 1.0, 0.0 };

  int t = 4;
  std::string model_repo = "simsearch";
  mkdir(model_repo.c_str(), 0770);
  AnnoySE ase(t, model_repo);
  ase.create_index();                // index creation
  ase.index(URIData("test1"), vec1); // indexing data
  ase.index(URIData("test2"), vec2);
  ase.index(URIData("test3"), vec3);
  std::vector<URIData> uris;
  std::vector<double> distances;
  ase.build_tree();                     // tree building
  ase.save_tree();                      // tree saving
  ase.search(vec1, 3, uris, distances); // searching nearest neighbors
  std::cerr << "search uris size=" << uris.size() << std::endl;
  for (size_t i = 0; i < uris.size(); i++)
    std::cout << uris.at(i)._uri << " / distances=" << distances.at(i)
              << std::endl;
  ase.remove_index();
  rmdir(model_repo.c_str());
}

TEST(annoyse, index_search_incr)
{
  std::vector<double> vec1 = { 1.0, 0.0, 0.0, 0.0 };
  std::vector<double> vec2 = { 0.0, 1.0, 0.0, 0.0 };
  std::vector<double> vec3 = { 1.0, 0.0, 1.0, 0.0 };
  std::vector<double> vec4 = { 0.0, 0.0, 5.0, 5.0 };

  int t = 4;
  std::string model_repo = "simsearch";
  mkdir(model_repo.c_str(), 0770);
  AnnoySE ase(t, model_repo);
  ase.create_index();
  ase.index(URIData("test1"), vec1);
  ase.index(URIData("test2"), vec2);
  ase.index(URIData("test3"), vec3);
  std::vector<URIData> uris;
  std::vector<double> distances;
  ase.build_tree();
  ase.search(vec1, 3, uris, distances);
  std::cerr << "search uris size=" << uris.size() << std::endl;
  for (size_t i = 0; i < uris.size(); i++)
    std::cout << uris.at(i)._uri << " / distances=" << distances.at(i)
              << std::endl;
  uris.clear();
  distances.clear();
  ase.unbuild_tree();
  ase.index(URIData("test4"), vec4);
  ase.build_tree();
  ase.search(vec4, 3, uris, distances);
  std::cerr << "search uris size=" << uris.size() << std::endl;
  for (size_t i = 0; i < uris.size(); i++)
    std::cout << uris.at(i)._uri << " / distances=" << distances.at(i)
              << std::endl;
  ase.remove_index();
  rmdir(model_repo.c_str());
}
