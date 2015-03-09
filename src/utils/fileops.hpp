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
#include <sys/stat.h>

namespace dd
{
  class fileops
  {
  public:

    static bool file_exists(const std::string &fname)
    {
      struct stat bstat;
      return (stat(fname.c_str(),&bstat)==0);
    }

    static int list_directory(const std::string &repo,
			      const bool &files,
			      const bool &dirs,
			      std::unordered_set<std::string> &lfiles)
    {
      DIR *dir;
      struct dirent *ent;
      if ((dir = opendir(repo.c_str())) != NULL) {
	/* print all the files and directories within directory */
	while ((ent = readdir(dir)) != NULL) {
	  if ((files && (ent->d_type == DT_REG || ent->d_type == DT_LNK))
	      || (dirs && ent->d_type == DT_DIR && ent->d_name[0] != '.'))
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
