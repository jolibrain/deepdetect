#ifndef DD_DB_HPP
#define DD_DB_HPP

// Disable the copy and assignment operator for a class.
#ifndef DISABLE_COPY_AND_ASSIGN
#define DISABLE_COPY_AND_ASSIGN(classname)      \
  private:                                      \
  classname(const classname&);                  \
  classname& operator=(const classname&)
#endif


#include <string>
#include "backends/torch/llogging.h"


namespace dd { namespace db {

    enum Mode { READ, WRITE, NEW };

    class Cursor {
    public:
      Cursor() { }
      virtual ~Cursor() { }
      virtual void SeekToFirst() = 0;
      virtual void Next() = 0;
      virtual std::string key() = 0;
      virtual std::string value() = 0;
      virtual bool valid() = 0;
  
      DISABLE_COPY_AND_ASSIGN(Cursor);
    };

    class Transaction {
    public:
      Transaction() { }
      virtual ~Transaction() { }
      virtual void Put(const std::string& key, const std::string& value) = 0;
      virtual void Commit() = 0;

      DISABLE_COPY_AND_ASSIGN(Transaction);
    };

    class DB {
    public:
      DB() { }
      virtual ~DB() { }
      virtual void Open(const std::string& source, Mode mode) = 0;
      virtual void Close() = 0;
      virtual Cursor* NewCursor() = 0;
      virtual Transaction* NewTransaction() = 0;
      virtual int Count() = 0;
      virtual void Get(const std::string &key,
                       std::string &data_val) = 0;
      virtual void Remove(const std::string &key) = 0;
  
      DISABLE_COPY_AND_ASSIGN(DB);
    };

    DB* GetDB(const std::string& backend);

  }  // namespace db
}  // namespace dd

#endif  // DD_DB_HPP
