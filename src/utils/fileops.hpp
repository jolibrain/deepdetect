/**
 * DeepDetect
 * Copyright (c) 2014-2015 Emmanuel Benazera
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

#ifndef DD_FILEOPS_H
#define DD_FILEOPS_H

#include <dirent.h>
#include <fstream>
#include <unordered_set>

namespace dd
{
  class fileops
  {
  public:

    static int list_directory_files(const std::string &repo,
				    std::unordered_set<std::string> &lfiles)
    {
      DIR *dir;
      struct dirent *ent;
      if ((dir = opendir(repo.c_str())) != NULL) {
	/* print all the files and directories within directory */
	while ((ent = readdir(dir)) != NULL) {
	  lfiles.insert(std::string(repo) + "/" + std::string(ent->d_name));
	}
	closedir(dir);
	return 0;
      } 
      else 
	{
	  return 1;
	}
    }
    
    static int copy_file(const std::string &fin,
			 const std::string &fout)
    {
      std::ifstream src(fin,std::ios::binary);
      if (!src.is_open())
	return 1;
      std::ofstream dst(fout,std::ios::binary);
      dst << src.rdbuf();
      src.close();
      dst.close();
      return 0;
    }
  };
  
}

#endif
