#pragma once
#include <staticjson/basic.hpp>

#include <rapidjson/document.h>

#include <vector>

namespace staticjson
{
class JSONHandler : public BaseHandler
{
protected:
    static const int MAX_DEPTH = 32;

    std::vector<Value> m_stack;
    Value* m_value;
    MemoryPoolAllocator* m_alloc;

private:
    bool stack_push();
    void stack_pop();
    Value& stack_top();
    bool postprocess();
    bool set_corrupted_dom();

public:
    explicit JSONHandler(Value* v, MemoryPoolAllocator* a);

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

    virtual bool write(IHandler* output) const override;

    virtual void reset() override;

    void reset(MemoryPoolAllocator* a);

    void generate_schema(Value& output, MemoryPoolAllocator&) const override { output.SetObject(); }
};

template <>
class Handler<Document> : public JSONHandler
{
public:
    explicit Handler(Document* h) : JSONHandler(h, &h->GetAllocator()) {}
    virtual void reset() override
    {
        JSONHandler::reset(&(static_cast<Document*>(this->m_value)->GetAllocator()));
    }
};

namespace nonpublic
{
    bool write_value(const Value& v, BaseHandler* out, ParseStatus* status);
    bool
    read_value(Value* v, MemoryPoolAllocator* alloc, const BaseHandler* input, ParseStatus* status);
}

template <class T>
bool from_json_value(const Value& v, T* t, ParseStatus* status)
{
    Handler<T> h(t);
    return nonpublic::write_value(v, &h, status);
}

template <class T>
bool from_json_document(const Document& d,
                        T* t,
                        ParseStatus* status)    // for consistency in API
{
    return from_json_value(d, t, status);
}

template <class T>
bool to_json_value(Value* v, MemoryPoolAllocator* alloc, const T& t, ParseStatus* status)
{
    Handler<T> h(const_cast<T*>(&t));
    return nonpublic::read_value(v, alloc, &h, status);
}

template <class T>
bool to_json_document(Document* d, const T& t, ParseStatus* status)
{
    return to_json_value(d, &d->GetAllocator(), t, status);
}
}
