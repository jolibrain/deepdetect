#pragma once

#include "stl_types.hpp"

#ifdef __has_include
#if __has_include(<optional>)
#include <optional>

namespace staticjson
{
    template <typename T>
    using optional = std::optional<T>;

    using std::nullopt;
}

#elif __has_include(<experimental/optional>)
#include <experimental/optional>

namespace staticjson
{
    template <typename T>
    using optional = std::experimental::optional<T>;

    using std::experimental::nullopt;
}

#else
#error "Missing <optional>"
#endif
#else
#error "Missing <optional>"
#endif

namespace staticjson
{

template <class T>
class Handler<optional<T>> : public BaseHandler
{
public:
    using ElementType = T;

protected:
    mutable optional<T>* m_value;
    mutable optional<Handler<ElementType>> internal_handler;
    int depth = 0;

public:
    explicit Handler(optional<T>* value) : m_value(value) {}

protected:
    void initialize()
    {
        if (!internal_handler)
        {
            m_value->emplace();
            internal_handler.emplace(&(**m_value));
        }
    }

    void reset() override
    {
        depth = 0;
        internal_handler = nullopt;
        *m_value = nullopt;
    }

    bool postcheck(bool success)
    {
        if (success)
            this->parsed = internal_handler->is_parsed();
        return success;
    }

public:
    bool Null() override
    {
        if (depth == 0)
        {
            *m_value = nullopt;
            this->parsed = true;
            return true;
        }
        else
        {
            initialize();
            return postcheck(internal_handler->Null());
        }
    }

    bool write(IHandler* out) const override
    {
        if (!m_value || !(*m_value))
        {
            return out->Null();
        }
        if (!internal_handler)
        {
            internal_handler.emplace(&(**m_value));
        }
        return internal_handler->write(out);
    }

    void generate_schema(Value& output, MemoryPoolAllocator& alloc) const override
    {
        const_cast<Handler<optional<T>>*>(this)->initialize();
        output.SetObject();
        Value anyOf(rapidjson::kArrayType);
        Value nullDescriptor(rapidjson::kObjectType);
        nullDescriptor.AddMember(rapidjson::StringRef("type"), rapidjson::StringRef("null"), alloc);
        Value descriptor;
        internal_handler->generate_schema(descriptor, alloc);
        anyOf.PushBack(nullDescriptor, alloc);
        anyOf.PushBack(descriptor, alloc);
        output.AddMember(rapidjson::StringRef("anyOf"), anyOf, alloc);
    }

    bool Bool(bool b) override
    {
        initialize();
        return postcheck(internal_handler->Bool(b));
    }

    bool Int(int i) override
    {
        initialize();
        return postcheck(internal_handler->Int(i));
    }

    bool Uint(unsigned i) override
    {
        initialize();
        return postcheck(internal_handler->Uint(i));
    }

    bool Int64(std::int64_t i) override
    {
        initialize();
        return postcheck(internal_handler->Int64(i));
    }

    bool Uint64(std::uint64_t i) override
    {
        initialize();
        return postcheck(internal_handler->Uint64(i));
    }

    bool Double(double i) override
    {
        initialize();
        return postcheck(internal_handler->Double(i));
    }

    bool String(const char* str, SizeType len, bool copy) override
    {
        initialize();
        return postcheck(internal_handler->String(str, len, copy));
    }

    bool Key(const char* str, SizeType len, bool copy) override
    {
        initialize();
        return postcheck(internal_handler->Key(str, len, copy));
    }

    bool StartObject() override
    {
        initialize();
        ++depth;
        return internal_handler->StartObject();
    }

    bool EndObject(SizeType len) override
    {
        initialize();
        --depth;
        return postcheck(internal_handler->EndObject(len));
    }

    bool StartArray() override
    {
        initialize();
        ++depth;
        return postcheck(internal_handler->StartArray());
    }

    bool EndArray(SizeType len) override
    {
        initialize();
        --depth;
        return postcheck(internal_handler->EndArray(len));
    }

    bool has_error() const override { return internal_handler && internal_handler->has_error(); }

    bool reap_error(ErrorStack& stk) override
    {
        return internal_handler && internal_handler->reap_error(stk);
    }

    std::string type_name() const override
    {
        if (this->internal_handler)
        {
            return "std::optional<" + this->internal_handler->type_name() + ">";
        }
        return "std::optional";
    }
};
}
