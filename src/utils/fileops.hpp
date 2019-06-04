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

#include <fstream>
#include <iostream>
#include <unordered_set>
#include <sys/stat.h>
#include <stdio.h>
#include <boost/filesystem.hpp>

#if !defined(WIN32)
#include <dirent.h>
#include <unistd.h>
#include <archive.h>
#include <archive_entry.h>
#endif

namespace dd
{
  class fileops
  {
  public:
#if defined(WIN32)
    static bool create_dir(const std::string &dirName, int mode)
    {
      assert(false);
      return false;
    }

    static bool file_exists(const std::string &fname)
    {
      boost::filesystem::path p(fname);

      return boost::filesystem::exists(p);
    }

    static bool file_exists(const std::string &fname, bool &directory)
    {
      boost::filesystem::path p(fname);

      directory = boost::filesystem::is_directory(p);
      return boost::filesystem::exists(p);
    }

    static bool is_db(const std::string &fname)
    {
      const std::vector<std::string> db_exts = { ".lmdb" }; // add more here
      for (auto e : db_exts)
        if (fname.find(e) != std::string::npos)
          return true;
      return false;
    }

    static long int file_last_modif(const std::string &fname)
    {
      boost::filesystem::path p(fname);

      return boost::filesystem::last_write_time(p);
    }

    static int clear_directory(const std::string &repo)
    {
      assert(false);
      return -1;
    }

    static int remove_dir(const std::string &d)
    {
      assert(false);
      return -1;
    }

    static int uncompress(const std::string &fc, const std::string &repo)
    {
      assert(false);
      return -1;
    }

#else
    static bool create_dir(const std::string &dirName, mode_t mode)
    {
      std::string s = dirName;
      size_t pre = 0, pos;
      std::string dir;
      int mdret = 0;

      if (s[s.size() - 1] != '/')
        {
          // force trailing / so we can handle everything in loop
          s += '/';
        }

      while ((pos = s.find_first_of('/', pre)) != std::string::npos)
        {
          dir = s.substr(0, pos++);
          pre = pos;
          if (dir.size() == 0)
            continue; // if leading / first time is 0 length
          if ((mdret = mkdir(dir.c_str(), mode)) && errno != EEXIST)
            {
              return mdret;
            }
        }
      return mdret;
    }

    static bool file_exists(const std::string &fname)
    {
      struct stat bstat;
      return (stat(fname.c_str(), &bstat) == 0);
    }

    static bool dir_exists(const std::string &fname)
    {
      bool dir;
      bool exists = file_exists(fname, dir);
      return exists && dir;
    }

    static bool file_exists(const std::string &fname, bool &directory)
    {
      struct stat bstat;
      int r = stat(fname.c_str(), &bstat);
      if (r != 0)
        {
          directory = false;
          return false;
        }
      if (S_ISDIR(bstat.st_mode))
        directory = true;
      else
        directory = false;
      return r == 0;
    }

    static bool is_db(const std::string &fname)
    {
      const std::vector<std::string> db_exts = { ".lmdb" }; // add more here
      for (auto e : db_exts)
        if (fname.find(e) != std::string::npos)
          return true;
      return false;
    }

    static long int file_last_modif(const std::string &fname)
    {
      struct stat bstat;
      if (stat(fname.c_str(), &bstat) == 0)
        return bstat.st_mtim.tv_sec;
      else
        return -1;
    }

    static int list_directory(const std::string &repo, const bool &files,
                              const bool &dirs, const bool &sub_files,
                              std::unordered_set<std::string> &lfiles)
    {
      boost::filesystem::path p(repo);

      if (!boost::filesystem::is_directory(p))
        return 1;

      boost::filesystem::directory_iterator dir_iter(p);
      boost::filesystem::directory_iterator end_iter;
      for (; dir_iter != end_iter; ++dir_iter)
        {
          if ((dirs && boost::filesystem::is_directory(dir_iter->status()))
              || (files
                  && boost::filesystem::is_regular_file(dir_iter->status()))
              || ((dirs || files)
                  && boost::filesystem::is_symlink(dir_iter->status())))
            lfiles.insert(dir_iter->path().string());

          if (sub_files
              && (boost::filesystem::is_directory(dir_iter->status())
                  || boost::filesystem::is_symlink(dir_iter->status())))
            list_directory(dir_iter->path().string(), files, dirs, sub_files,
                           lfiles);
        }

      return 0;
    }

    // remove everything, including first level directories within directory
    static int clear_directory(const std::string &repo)
    {
      boost::system::error_code ercode;
      boost::filesystem::directory_iterator dir_iter(repo, ercode);
      boost::filesystem::directory_iterator end_iter;
      int err = ercode.value();
      if (ercode.value() != 0)
        return ercode.value();

      err = 0;
      for (; dir_iter != end_iter; ++dir_iter)
        {
          boost::filesystem::remove_all(*dir_iter, ercode);
          if (ercode.value() == ENOENT || ercode.value() == EACCES)
            err++;
        }
      return err;
    }

    // empty extensions means a wildcard
    static int
    remove_directory_files(const std::string &repo,
                           const std::vector<std::string> &extensions)
    {
      boost::system::error_code ercode;
      boost::filesystem::directory_iterator dir_iter(repo, ercode);
      boost::filesystem::directory_iterator end_iter;
      int err = ercode.value();
      if (ercode.value() != 0)
        return ercode.value();

      err = 0;
      for (; dir_iter != end_iter; ++dir_iter)
        {
          if (extensions.empty())
            {
              boost::filesystem::remove_all(*dir_iter, ercode);
              if (ercode.value() == ENOENT || ercode.value() == EACCES)
                err++;
            }
          else
            {
              for (std::string s : extensions)
                if (dir_iter->path().native().find(s) != std::string::npos)
                  {
                    boost::filesystem::remove(*dir_iter, ercode);
                    if (ercode.value() == ENOENT || ercode.value() == EACCES)
                      err++;
                  }
            }
        }
      return err;
    }

    static int copy_file(const std::string &fin, const std::string &fout)
    {
      std::ifstream src(fin, std::ios::binary);
      if (!src.is_open())
        return 1;
      std::ofstream dst(fout, std::ios::binary);
      if (!dst.is_open())
        return 2;
      dst << src.rdbuf();
      src.close();
      dst.close();
      return 0;
    }

    static int remove_file(const std::string &repo, const std::string &f)
    {
      std::string fn = repo + "/" + f;
      if (remove(fn.c_str()))
        return -1; // error.
      return 0;
    }

    static int remove_dir(const std::string &d)
    {
      if (remove(d.c_str()))
        return -1; // error.
      return 0;
    }

    static int replace_string_in_file(const std::string &filename,
                                      const std::string &search,
                                      const std::string &replace)
    {
      int nr = 0;
      std::ifstream ifs(filename.c_str(),
                        std::ios::in | std::ios::binary | std::ios::ate);
      std::stringstream sstr;
      sstr << ifs.rdbuf();
      ifs.close();
      std::string filestring = sstr.str();

      size_t pos = 0;
      while ((pos = filestring.find(search, pos)) != std::string::npos)
        {
          filestring.replace(pos, search.length(), replace);
          pos += replace.length();
          nr++;
        }
      std::ofstream out(filename.c_str(),
                        std::ios::out | std::ios::binary | std::ios::trunc);
      out << filestring;
      out.close();
      return nr;
    }

    static int copy_uncompressed_data(struct archive *ar, struct archive *aw)
    {
      int r;
      const void *buff;
      size_t size;
      int64_t offset;

      for (;;)
        {
          r = archive_read_data_block(ar, &buff, &size, &offset);
          if (r == ARCHIVE_EOF)
            return (ARCHIVE_OK);
          if (r < ARCHIVE_OK)
            return (r);
          r = archive_write_data_block(aw, buff, size, offset);
          if (r < ARCHIVE_OK)
            {
              std::cerr << "archive error: " << archive_error_string(aw)
                        << std::endl;
              return r;
            }
        }
    }

    static int uncompress(const std::string &fc, const std::string &repo)
    {
      struct archive *a;
      struct archive *ext;
      struct archive_entry *entry;
      int flags;
      int r;

      flags = ARCHIVE_EXTRACT_TIME;
      // flags |= ARCHIVE_EXTRACT_PERM;
      // flags |= ARCHIVE_EXTRACT_ACL;
      flags |= ARCHIVE_EXTRACT_FFLAGS;

      a = archive_read_new();
      archive_read_support_format_all(a);
      archive_read_support_filter_all(a);
      ext = archive_write_disk_new();
      archive_write_disk_set_options(ext, flags);
      archive_write_disk_set_standard_lookup(ext);
      if ((r = archive_read_open_filename(a, fc.c_str(),
                                          10240))) // 10240 is block_size
        return r;
      for (;;)
        {
          r = archive_read_next_header(a, &entry);
          if (r == ARCHIVE_EOF)
            break;
          if (r < ARCHIVE_OK)
            std::cerr << "archive error: " << archive_error_string(a)
                      << std::endl;
          if (r < ARCHIVE_WARN)
            {
              std::cerr << "archive error, aborting\n";
              return 1;
            }
          const char *compressed_head = archive_entry_pathname(entry);
          const std::string full_outpath = repo + "/" + compressed_head;
          archive_entry_set_pathname(entry, full_outpath.c_str());
          r = archive_write_header(ext, entry);
          if (r < ARCHIVE_OK)
            std::cerr << "archive error: " << archive_error_string(ext)
                      << std::endl;
          else if (archive_entry_size(entry) > 0)
            {
              r = copy_uncompressed_data(a, ext);
              if (r < ARCHIVE_OK)
                std::cerr << "error writing uncompressed data: "
                          << archive_error_string(ext) << std::endl;
              if (r < ARCHIVE_WARN)
                {
                  std::cerr << "archive error, aborting\n";
                  return 2;
                }
            }
          r = archive_write_finish_entry(ext);
          if (r < ARCHIVE_OK)
            std::cerr << "error finishing uncompressed data entry: "
                      << archive_error_string(ext) << std::endl;
          if (r < ARCHIVE_WARN)
            {
              std::cerr << "archive error, aborting\n";
              return 3;
            }
        }
      archive_read_close(a);
      archive_read_free(a);
      archive_write_close(ext);
      archive_write_free(ext);

      return 0;
    }
#endif
  };
}

#endif
