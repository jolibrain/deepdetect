#include "db.hpp"
#include "db_lmdb.hpp"

#include <string>

namespace dd { namespace db {

    DB* GetDB(const std::string& backend) {
      // #ifdef USE_LEVELDB
      //   if (backend == "leveldb") {
      //     return new LevelDB();
      //   }
      // #endif  // USE_LEVELDB
      // #ifdef USE_LMDB
      if (backend == "lmdb") {
        return new LMDB();
      }
      // #endif  // USE_LMDB
      LOG(ERROR) << "Unknown database backend";
      LOG(FATAL) << "fatal error";
        return NULL;
    }

  }  // namespace db
}  // namespace dd
