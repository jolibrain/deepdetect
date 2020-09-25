#pragma once
#include <staticjson/basic.hpp>

#include <limits>
#include <string>
#include <type_traits>

namespace staticjson
{

template <class IntType>
class IntegerHandler : public BaseHandler
{
    static_assert(std::is_arithmetic<IntType>::value, "Only arithmetic types are allowed");

protected:
    IntType* m_value;

    template <class AnotherIntType>
    static constexpr typename std::enable_if<std::is_integral<AnotherIntType>::value, bool>::type
    is_out_of_range(AnotherIntType a)
    {
        typedef typename std::common_type<IntType, AnotherIntType>::type CommonType;
        typedef typename std::numeric_limits<IntType> this_limits;
        typedef typename std::numeric_limits<AnotherIntType> that_limits;

        // The extra logic related to this_limits::min/max allows the compiler to
        // short circuit this check at compile time. For instance, a `uint32_t`
        // will NEVER be out of range for an `int64_t`
        return ((this_limits::is_signed == that_limits::is_signed)
                    ? ((CommonType(this_limits::min()) > CommonType(a)
                        || CommonType(this_limits::max()) < CommonType(a)))
                    : (this_limits::is_signed)
                        ? (CommonType(this_limits::max()) < CommonType(a))
                        : (a < 0 || CommonType(a) > CommonType(this_limits::max())));
    }

    template <class FloatType>
    static constexpr typename std::enable_if<std::is_floating_point<FloatType>::value, bool>::type
    is_out_of_range(FloatType f)
    {
        return static_cast<FloatType>(static_cast<IntType>(f)) != f;
    }

    template <class ReceiveNumType>
    bool receive(ReceiveNumType r, const char* actual_type)
    {
        if (is_out_of_range(r))
            return set_out_of_range(actual_type);
        *m_value = static_cast<IntType>(r);
        this->parsed = true;
        return true;
    }

public:
    explicit IntegerHandler(IntType* value) : m_value(value) {}

    bool Int(int i) override { return receive(i, "int"); }

    bool Uint(unsigned i) override { return receive(i, "unsigned int"); }

    bool Int64(std::int64_t i) override { return receive(i, "std::int64_t"); }

    bool Uint64(std::uint64_t i) override { return receive(i, "std::uint64_t"); }

    bool Double(double d) override { return receive(d, "double"); }

    bool write(IHandler* output) const override
    {
        if (std::numeric_limits<IntType>::is_signed)
        {
            return output->Int64(*m_value);
        }
        else
        {
            return output->Uint64(*m_value);
        }
    }

    void generate_schema(Value& output, MemoryPoolAllocator& alloc) const override
    {
        output.SetObject();
        output.AddMember(rapidjson::StringRef("type"), rapidjson::StringRef("integer"), alloc);
        Value minimum, maximum;
        if (std::numeric_limits<IntType>::is_signed)
        {
            minimum.SetInt64(std::numeric_limits<IntType>::min());
            maximum.SetInt64(std::numeric_limits<IntType>::max());
        }
        else
        {
            minimum.SetUint64(std::numeric_limits<IntType>::min());
            maximum.SetUint64(std::numeric_limits<IntType>::max());
        }
        output.AddMember(rapidjson::StringRef("minimum"), minimum, alloc);
        output.AddMember(rapidjson::StringRef("maximum"), maximum, alloc);
    }
};

template <>
class Handler<std::nullptr_t> : public BaseHandler
{
public:
    explicit Handler(std::nullptr_t*) {}

    bool Null() override
    {
        this->parsed = true;
        return true;
    }

    std::string type_name() const override { return "null"; }

    bool write(IHandler* output) const override { return output->Null(); }

    void generate_schema(Value& output, MemoryPoolAllocator& alloc) const override
    {
        output.SetObject();
        output.AddMember(rapidjson::StringRef("type"), rapidjson::StringRef("null"), alloc);
    }
};

template <>
class Handler<bool> : public BaseHandler
{
private:
    bool* m_value;

public:
    explicit Handler(bool* value) : m_value(value) {}

    bool Bool(bool v) override
    {
        *m_value = v;
        this->parsed = true;
        return true;
    }

    std::string type_name() const override { return "bool"; }

    bool write(IHandler* output) const override { return output->Bool(*m_value); }

    void generate_schema(Value& output, MemoryPoolAllocator& alloc) const override
    {
        output.SetObject();
        output.AddMember(rapidjson::StringRef("type"), rapidjson::StringRef("boolean"), alloc);
    }
};

template <>
class Handler<int> : public IntegerHandler<int>
{
public:
    explicit Handler(int* i) : IntegerHandler<int>(i) {}

    std::string type_name() const override { return "int"; }

    bool write(IHandler* output) const override { return output->Int(*m_value); }
};

template <>
class Handler<unsigned int> : public IntegerHandler<unsigned int>
{
public:
    explicit Handler(unsigned* i) : IntegerHandler<unsigned int>(i) {}

    std::string type_name() const override { return "unsigned int"; }

    bool write(IHandler* output) const override { return output->Uint(*m_value); }
};

template <>
class Handler<long> : public IntegerHandler<long>
{
public:
    explicit Handler(long* i) : IntegerHandler<long>(i) {}

    std::string type_name() const override { return "long"; }
};

template <>
class Handler<unsigned long> : public IntegerHandler<unsigned long>
{
public:
    explicit Handler(unsigned long* i) : IntegerHandler<unsigned long>(i) {}

    std::string type_name() const override { return "unsigned long"; }
};

template <>
class Handler<long long> : public IntegerHandler<long long>
{
public:
    explicit Handler(long long* i) : IntegerHandler<long long>(i) {}

    std::string type_name() const override { return "long long"; }
};

template <>
class Handler<unsigned long long> : public IntegerHandler<unsigned long long>
{
public:
    explicit Handler(unsigned long long* i) : IntegerHandler<unsigned long long>(i) {}

    std::string type_name() const override { return "unsigned long long"; }
};

// char is an alias for bool to work around the stupid `std::vector<bool>`
template <>
class Handler<char> : public BaseHandler
{
private:
    char* m_value;

public:
    explicit Handler(char* i) : m_value(i) {}

    std::string type_name() const override { return "bool"; }

    bool Bool(bool v) override
    {
        *this->m_value = v;
        this->parsed = true;
        return true;
    }

    bool write(IHandler* out) const override { return out->Bool(*m_value != 0); }

    void generate_schema(Value& output, MemoryPoolAllocator& alloc) const override
    {
        output.SetObject();
        output.AddMember(rapidjson::StringRef("type"), rapidjson::StringRef("boolean"), alloc);
    }
};

template <>
class Handler<double> : public BaseHandler
{
private:
    double* m_value;

public:
    explicit Handler(double* v) : m_value(v) {}

    bool Int(int i) override
    {
        *m_value = i;
        this->parsed = true;
        return true;
    }

    bool Uint(unsigned i) override
    {
        *m_value = i;
        this->parsed = true;
        return true;
    }

    bool Int64(std::int64_t i) override
    {
        *m_value = static_cast<double>(i);
        if (static_cast<decltype(i)>(*m_value) != i)
            return set_out_of_range("std::int64_t");
        this->parsed = true;
        return true;
    }

    bool Uint64(std::uint64_t i) override
    {
        *m_value = static_cast<double>(i);
        if (static_cast<decltype(i)>(*m_value) != i)
            return set_out_of_range("std::uint64_t");
        this->parsed = true;
        return true;
    }

    bool Double(double d) override
    {
        *m_value = d;
        this->parsed = true;
        return true;
    }

    std::string type_name() const override { return "double"; }

    bool write(IHandler* out) const override { return out->Double(*m_value); }

    void generate_schema(Value& output, MemoryPoolAllocator& alloc) const override
    {
        output.SetObject();
        output.AddMember(rapidjson::StringRef("type"), rapidjson::StringRef("number"), alloc);
    }
};

template <>
class Handler<float> : public BaseHandler
{
private:
    float* m_value;

public:
    explicit Handler(float* v) : m_value(v) {}

    bool Int(int i) override
    {
        *m_value = static_cast<float>(i);
        if (static_cast<decltype(i)>(*m_value) != i)
            return set_out_of_range("int");
        this->parsed = true;
        return true;
    }

    bool Uint(unsigned i) override
    {
        *m_value = static_cast<float>(i);
        if (static_cast<decltype(i)>(*m_value) != i)
            return set_out_of_range("unsigned int");
        this->parsed = true;
        return true;
    }

    bool Int64(std::int64_t i) override
    {
        *m_value = static_cast<float>(i);
        if (static_cast<decltype(i)>(*m_value) != i)
            return set_out_of_range("std::int64_t");
        this->parsed = true;
        return true;
    }

    bool Uint64(std::uint64_t i) override
    {
        *m_value = static_cast<float>(i);
        if (static_cast<decltype(i)>(*m_value) != i)
            return set_out_of_range("std::uint64_t");
        this->parsed = true;
        return true;
    }

    bool Double(double d) override
    {
        *m_value = static_cast<float>(d);
        this->parsed = true;
        return true;
    }

    std::string type_name() const override { return "float"; }

    bool write(IHandler* out) const override { return out->Double(*m_value); }

    void generate_schema(Value& output, MemoryPoolAllocator& alloc) const override
    {
        output.SetObject();
        output.AddMember(rapidjson::StringRef("type"), rapidjson::StringRef("number"), alloc);
    }
};

template <>
class Handler<std::string> : public BaseHandler
{
private:
    std::string* m_value;

public:
    explicit Handler(std::string* v) : m_value(v) {}

    bool String(const char* str, SizeType length, bool) override
    {
        m_value->assign(str, length);
        this->parsed = true;
        return true;
    }

    std::string type_name() const override { return "string"; }

    bool write(IHandler* out) const override
    {
        return out->String(m_value->data(), SizeType(m_value->size()), true);
    }

    void generate_schema(Value& output, MemoryPoolAllocator& alloc) const override
    {
        output.SetObject();
        output.AddMember(rapidjson::StringRef("type"), rapidjson::StringRef("string"), alloc);
    }
};
}
