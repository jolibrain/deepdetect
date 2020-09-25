#pragma once

#include <rapidjson/document.h>
#include <staticjson/error.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <type_traits>

namespace staticjson
{
struct NonMobile
{
    NonMobile() {}
    ~NonMobile() {}
    NonMobile(const NonMobile&) = delete;
    NonMobile(NonMobile&&) = delete;
    NonMobile& operator=(const NonMobile&) = delete;
    NonMobile& operator=(NonMobile&&) = delete;
};

typedef unsigned int SizeType;

class IHandler
{
public:
    IHandler() {}

    virtual ~IHandler();

    virtual bool Null() = 0;

    virtual bool Bool(bool) = 0;

    virtual bool Int(int) = 0;

    virtual bool Uint(unsigned) = 0;

    virtual bool Int64(std::int64_t) = 0;

    virtual bool Uint64(std::uint64_t) = 0;

    virtual bool Double(double) = 0;

    virtual bool String(const char*, SizeType, bool) = 0;

    virtual bool StartObject() = 0;

    virtual bool Key(const char*, SizeType, bool) = 0;

    virtual bool EndObject(SizeType) = 0;

    virtual bool StartArray() = 0;

    virtual bool EndArray(SizeType) = 0;

    virtual bool RawNumber(const char*, SizeType, bool);

    virtual void prepare_for_reuse() = 0;
};

using rapidjson::Document;
using rapidjson::Value;

typedef rapidjson::MemoryPoolAllocator<> MemoryPoolAllocator;

class BaseHandler : public IHandler, private NonMobile
{
    friend class NullableHandler;

protected:
    std::unique_ptr<ErrorBase> the_error;
    bool parsed = false;

protected:
    bool set_out_of_range(const char* actual_type);
    bool set_type_mismatch(const char* actual_type);

    virtual void reset() {}

public:
    BaseHandler() {}

    virtual ~BaseHandler();

    virtual std::string type_name() const = 0;

    virtual bool Null() override { return set_type_mismatch("null"); }

    virtual bool Bool(bool) override { return set_type_mismatch("bool"); }

    virtual bool Int(int) override { return set_type_mismatch("int"); }

    virtual bool Uint(unsigned) override { return set_type_mismatch("unsigned"); }

    virtual bool Int64(std::int64_t) override { return set_type_mismatch("int64_t"); }

    virtual bool Uint64(std::uint64_t) override { return set_type_mismatch("uint64_t"); }

    virtual bool Double(double) override { return set_type_mismatch("double"); }

    virtual bool String(const char*, SizeType, bool) override
    {
        return set_type_mismatch("string");
    }

    virtual bool StartObject() override { return set_type_mismatch("object"); }

    virtual bool Key(const char*, SizeType, bool) override { return set_type_mismatch("object"); }

    virtual bool EndObject(SizeType) override { return set_type_mismatch("object"); }

    virtual bool StartArray() override { return set_type_mismatch("array"); }

    virtual bool EndArray(SizeType) override { return set_type_mismatch("array"); }

    virtual bool has_error() const { return bool(the_error); }

    virtual bool reap_error(ErrorStack& errs)
    {
        if (!the_error)
            return false;
        errs.push(the_error.release());
        return true;
    }

    bool is_parsed() const { return parsed; }

    void prepare_for_reuse() override
    {
        the_error.reset();
        parsed = false;
        reset();
    }

    virtual bool write(IHandler* output) const = 0;

    virtual void generate_schema(Value& output, MemoryPoolAllocator& alloc) const = 0;
};

struct Flags
{
    static const unsigned Default = 0x0, AllowDuplicateKey = 0x1, Optional = 0x2, IgnoreRead = 0x4,
                          IgnoreWrite = 0x8, DisallowUnknownKey = 0x10;
};

// Forward declaration
template <class T>
class Handler;

class ObjectHandler : public BaseHandler
{
protected:
    struct FlaggedHandler
    {
        std::unique_ptr<BaseHandler> handler;
        unsigned flags;
    };

protected:
    std::map<std::string, FlaggedHandler> internals;
    FlaggedHandler* current = nullptr;
    std::string current_name;
    int depth = 0;
    unsigned flags = Flags::Default;

protected:
    bool precheck(const char* type);
    bool postcheck(bool success);
    void set_missing_required(const std::string& name);
    void add_handler(std::string&&, FlaggedHandler&&);
    void reset() override;

public:
    ObjectHandler();

    ~ObjectHandler();

    std::string type_name() const override;

    virtual bool Null() override;

    virtual bool Bool(bool) override;

    virtual bool Int(int) override;

    virtual bool Uint(unsigned) override;

    virtual bool Int64(std::int64_t) override;

    virtual bool Uint64(std::uint64_t) override;

    virtual bool Double(double) override;

    virtual bool String(const char*, SizeType, bool) override;

    virtual bool StartObject() override;

    virtual bool Key(const char*, SizeType, bool) override;

    virtual bool EndObject(SizeType) override;

    virtual bool StartArray() override;

    virtual bool EndArray(SizeType) override;

    virtual bool reap_error(ErrorStack&) override;

    virtual bool write(IHandler* output) const override;

    virtual void generate_schema(Value& output, MemoryPoolAllocator& alloc) const override;

    unsigned get_flags() const { return flags; }

    void set_flags(unsigned f) { flags = f; }

    template <class T>
    void add_property(std::string name, T* pointer, unsigned flags_ = Flags::Default)
    {
        FlaggedHandler fh;
        fh.handler.reset(new Handler<T>(pointer));
        fh.flags = flags_;
        add_handler(std::move(name), std::move(fh));
    }
};

template <class T>
struct Converter
{
    typedef T shadow_type;

    static std::unique_ptr<ErrorBase> from_shadow(const shadow_type& shadow, T& value)
    {
        (void)shadow;
        (void)value;
        return nullptr;
    }

    static void to_shadow(const T& value, shadow_type& shadow)
    {
        (void)shadow;
        (void)value;
    }

    static std::string type_name() { return "T"; }

    static constexpr bool has_specialized_type_name = false;
};

template <class T>
void init(T* t, ObjectHandler* h)
{
    t->staticjson_init(h);
}

template <class T>
class ObjectTypeHandler : public ObjectHandler
{
public:
    explicit ObjectTypeHandler(T* t) { init(t, this); }
};

template <class T>
class ConversionHandler : public BaseHandler
{
private:
    typedef typename Converter<T>::shadow_type shadow_type;
    typedef Handler<shadow_type> internal_type;

private:
    shadow_type shadow;
    internal_type internal;
    T* m_value;

protected:
    bool postprocess(bool success)
    {
        if (!success)
        {
            return false;
        }
        if (!internal.is_parsed())
            return true;
        this->parsed = true;
        auto err = Converter<T>::from_shadow(shadow, *m_value);
        if (err)
        {
            this->the_error.swap(err);
            return false;
        }
        return true;
    }

    void reset() override
    {
        shadow = shadow_type();
        internal.prepare_for_reuse();
    }

public:
    explicit ConversionHandler(T* t) : shadow(), internal(&shadow), m_value(t) {}

    std::string type_name() const override
    {
        // if (Converter<T>::has_specialized_type_name)
        //  return Converter<T>::type_name();
        return internal.type_name();
    }

    virtual bool Null() override { return postprocess(internal.Null()); }

    virtual bool Bool(bool b) override { return postprocess(internal.Bool(b)); }

    virtual bool Int(int i) override { return postprocess(internal.Int(i)); }

    virtual bool Uint(unsigned u) override { return postprocess(internal.Uint(u)); }

    virtual bool Int64(std::int64_t i) override { return postprocess(internal.Int64(i)); }

    virtual bool Uint64(std::uint64_t u) override { return postprocess(internal.Uint64(u)); }

    virtual bool Double(double d) override { return postprocess(internal.Double(d)); }

    virtual bool String(const char* str, SizeType size, bool copy) override
    {
        return postprocess(internal.String(str, size, copy));
    }

    virtual bool StartObject() override { return postprocess(internal.StartObject()); }

    virtual bool Key(const char* str, SizeType size, bool copy) override
    {
        return postprocess(internal.Key(str, size, copy));
    }

    virtual bool EndObject(SizeType sz) override { return postprocess(internal.EndObject(sz)); }

    virtual bool StartArray() override { return postprocess(internal.StartArray()); }

    virtual bool EndArray(SizeType sz) override { return postprocess(internal.EndArray(sz)); }

    virtual bool has_error() const override
    {
        return BaseHandler::has_error() || internal.has_error();
    }

    bool reap_error(ErrorStack& errs) override
    {
        return BaseHandler::reap_error(errs) || internal.reap_error(errs);
    }

    virtual bool write(IHandler* output) const override
    {
        Converter<T>::to_shadow(*m_value, const_cast<shadow_type&>(shadow));
        return internal.write(output);
    }

    void generate_schema(Value& output, MemoryPoolAllocator& alloc) const override
    {
        return internal.generate_schema(output, alloc);
    }
};

namespace helper
{
    template <class T, bool no_conversion>
    class DispatchHandler;
    template <class T>
    class DispatchHandler<T, true> : public ::staticjson::ObjectTypeHandler<T>
    {
    public:
        explicit DispatchHandler(T* t) : ::staticjson::ObjectTypeHandler<T>(t) {}
    };

    template <class T>
    class DispatchHandler<T, false> : public ::staticjson::ConversionHandler<T>
    {
    public:
        explicit DispatchHandler(T* t) : ::staticjson::ConversionHandler<T>(t) {}
    };
}

template <class T>
class Handler
    : public helper::DispatchHandler<T, std::is_same<typename Converter<T>::shadow_type, T>::value>
{
public:
    typedef helper::DispatchHandler<T, std::is_same<typename Converter<T>::shadow_type, T>::value>
        base_type;
    explicit Handler(T* t) : base_type(t) {}
};
}
