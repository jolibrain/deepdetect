#pragma once

#include <staticjson/primitive_types.hpp>
#include <string>
#include <utility>
#include <vector>

namespace staticjson
{
template <class Enum, class Derived>
class EnumHandler : public BaseHandler
{
private:
    Enum* m_value;

    static const std::vector<std::pair<std::string, Enum>>& get_mapping()
    {
        return Derived::get_mapping();
    };

public:
    explicit EnumHandler(Enum* value) : m_value(value) {}

    bool String(const char* str, SizeType sz, bool) override
    {
        const auto& mapping = get_mapping();
        for (const std::pair<std::string, Enum>& pair : mapping)
        {
            if (pair.first.size() == sz && memcmp(pair.first.data(), str, sz) == 0)
            {
                *m_value = pair.second;
                this->parsed = true;
                return true;
            }
        }
        the_error.reset(new error::InvalidEnumError(std::string(str, str + sz)));
        return false;
    }

    bool write(IHandler* output) const override
    {
        const auto& mapping = get_mapping();
        for (const std::pair<std::string, Enum>& pair : mapping)
        {
            if (*m_value == pair.second)
            {
                output->String(pair.first.data(), static_cast<SizeType>(pair.first.size()), false);
                return true;
            }
        }
        return false;
    }

    void generate_schema(Value& output, MemoryPoolAllocator& alloc) const override
    {
        output.SetObject();
        output.AddMember(rapidjson::StringRef("type"), rapidjson::StringRef("string"), alloc);
        Value enumerations(rapidjson::kArrayType);
        const auto& mapping = get_mapping();
        for (const std::pair<std::string, Enum>& pair : mapping)
        {
            enumerations.PushBack(rapidjson::StringRef(pair.first.data(), pair.first.size()),
                                  alloc);
        }
        output.AddMember(rapidjson::StringRef("enum"), enumerations, alloc);
    }
};
}

#define STATICJSON_DECLARE_ENUM(type, ...)                                                         \
    namespace staticjson                                                                           \
    {                                                                                              \
        template <>                                                                                \
        class Handler<type> : public EnumHandler<type, Handler<type>>                              \
        {                                                                                          \
        public:                                                                                    \
            explicit Handler(type* value) : EnumHandler<type, Handler<type>>(value) {}             \
            std::string type_name() const override { return #type; }                               \
            static const std::vector<std::pair<std::string, type>>& get_mapping()                  \
            {                                                                                      \
                static std::vector<std::pair<std::string, type>> mapping{__VA_ARGS__};             \
                return mapping;                                                                    \
            }                                                                                      \
        };                                                                                         \
    }
