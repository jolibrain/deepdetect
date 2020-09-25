#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace staticjson
{
std::string quote(const std::string& str);

class ErrorStack;
class ErrorBase;

namespace error
{
    namespace internal
    {
        class error_stack_const_iterator;
    }
    using staticjson::ErrorBase;
    using staticjson::ErrorStack;
}

class ErrorBase
{
protected:
    explicit ErrorBase() : next(0) {}

private:
    ErrorBase* next;

    friend class ErrorStack;
    friend class error::internal::error_stack_const_iterator;

public:
    virtual int type() const = 0;
    virtual bool is_intermediate() const { return false; }
    virtual ~ErrorBase() {}
    virtual std::string description() const = 0;
};

namespace error
{
    typedef int error_type;

    static const error_type SUCCESS = 0, OBJECT_MEMBER = 1, ARRAY_ELEMENT = 2, MISSING_REQUIRED = 3,
                            TYPE_MISMATCH = 4, NUMBER_OUT_OF_RANGE = 5, ARRAY_LENGTH_MISMATCH = 6,
                            UNKNOWN_FIELD = 7, DUPLICATE_KEYS = 8, CORRUPTED_DOM = 9,
                            TOO_DEEP_RECURSION = 10, INVALID_ENUM = 11, CUSTOM = -1;

    class Success : public ErrorBase
    {
    public:
        explicit Success() {}
        std::string description() const;
        error_type type() const { return SUCCESS; }
    };

    class IntermediateError : public ErrorBase
    {
    public:
        bool is_intermediate() const { return true; }
    };

    class ObjectMemberError : public IntermediateError
    {
    private:
        std::string m_member_name;

    public:
        explicit ObjectMemberError(std::string memberName) { m_member_name.swap(memberName); }

        const std::string& member_name() const { return m_member_name; }

        std::string description() const;

        error_type type() const { return OBJECT_MEMBER; }
    };

    class ArrayElementError : public IntermediateError
    {
    private:
        std::size_t m_index;

    public:
        explicit ArrayElementError(std::size_t idx) : m_index(idx) {}

        std::size_t index() const { return m_index; }

        std::string description() const;

        error_type type() const { return ARRAY_ELEMENT; }
    };

    class RequiredFieldMissingError : public ErrorBase
    {
    private:
        std::vector<std::string> m_missing_members;

    public:
        explicit RequiredFieldMissingError() {}

        std::vector<std::string>& missing_members() { return m_missing_members; };

        const std::vector<std::string>& missing_members() const { return m_missing_members; }

        std::string description() const;

        error_type type() const { return MISSING_REQUIRED; }
    };

    class TypeMismatchError : public ErrorBase
    {
    private:
        std::string m_expected_type;
        std::string m_actual_type;

    public:
        explicit TypeMismatchError(std::string expectedType, std::string actualType)
        {
            m_expected_type.swap(expectedType);
            m_actual_type.swap(actualType);
        }

        const std::string& expected_type() const { return m_expected_type; }

        const std::string& actual_type() const { return m_actual_type; }

        std::string description() const;

        error_type type() const { return TYPE_MISMATCH; }
    };

    class RecursionTooDeepError : public ErrorBase
    {
        std::string description() const override;
        error_type type() const override { return TOO_DEEP_RECURSION; }
    };

    class NumberOutOfRangeError : public ErrorBase
    {
        std::string m_expected_type;
        std::string m_actual_type;

    public:
        explicit NumberOutOfRangeError(std::string expectedType, std::string actualType)
        {
            m_expected_type.swap(expectedType);
            m_actual_type.swap(actualType);
        }

        const std::string& expected_type() const { return m_expected_type; }

        const std::string& actual_type() const { return m_actual_type; }

        std::string description() const;

        error_type type() const { return NUMBER_OUT_OF_RANGE; }
    };

    class DuplicateKeyError : public ErrorBase
    {
    private:
        std::string key_name;

    public:
        explicit DuplicateKeyError(std::string name) { key_name.swap(name); }

        const std::string& key() const { return key_name; }

        error_type type() const { return DUPLICATE_KEYS; }

        std::string description() const;
    };

    class UnknownFieldError : public ErrorBase
    {
    private:
        std::string m_name;

    public:
        explicit UnknownFieldError(const char* name, std::size_t length) : m_name(name, length) {}

        const std::string& field_name() const { return m_name; }

        error_type type() const { return UNKNOWN_FIELD; }

        std::string description() const;
    };

    class CorruptedDOMError : public ErrorBase
    {
    public:
        std::string description() const;

        error_type type() const { return CORRUPTED_DOM; }
    };

    class ArrayLengthMismatchError : public ErrorBase
    {
    public:
        std::string description() const;

        error_type type() const { return ARRAY_LENGTH_MISMATCH; }
    };

    class InvalidEnumError : public ErrorBase
    {
    private:
        std::string m_name;

    public:
        explicit InvalidEnumError(std::string name) { m_name.swap(name); }
        std::string description() const;
        error_type type() const { return INVALID_ENUM; }
    };

    class CustomError : public ErrorBase
    {
    private:
        std::string m_message;

    public:
        explicit CustomError(std::string message) { m_message.swap(message); }
        std::string description() const;
        error_type type() const { return CUSTOM; }
    };

    namespace internal
    {

        class error_stack_const_iterator
            : public std::iterator<std::forward_iterator_tag, const ErrorBase>
        {
        private:
            const ErrorBase* e;

            typedef std::iterator<std::forward_iterator_tag, const ErrorBase> base_type;

        public:
            explicit error_stack_const_iterator(const ErrorBase* p) : e(p) {}
            reference operator*() const { return *e; }

            pointer operator->() const { return e; }

            error_stack_const_iterator& operator++()
            {
                e = e->next;
                return *this;
            }

            bool operator==(error_stack_const_iterator that) const { return e == that.e; }

            bool operator!=(error_stack_const_iterator that) const { return e != that.e; }
        };
    }
}

class ErrorStack
{
private:
    ErrorBase* head;
    std::size_t m_size;

    ErrorStack(const ErrorStack&);
    ErrorStack& operator=(const ErrorStack&);

public:
    typedef error::internal::error_stack_const_iterator const_iterator;

    explicit ErrorStack() : head(0), m_size(0) {}

    const_iterator begin() const { return const_iterator(head); }

    const_iterator end() const { return const_iterator(0); }

    // This will take the ownership of e
    // Requires it to be dynamically allocated
    void push(ErrorBase* e)
    {
        if (e)
        {
            e->next = head;
            head = e;
            ++m_size;
        }
    }

    // The caller will take the responsibility of deleting the returned pointer
    // Returns NULL when empty
    ErrorBase* pop()
    {
        if (head)
        {
            ErrorBase* result = head;
            head = head->next;
            --m_size;
            return result;
        }
        return 0;
    }

    bool empty() const { return head == 0; }

    explicit operator bool() const { return !empty(); }

    bool operator!() const { return empty(); }

    std::size_t size() const { return m_size; }

    ~ErrorStack()
    {
        while (head)
        {
            ErrorBase* next = head->next;
            delete head;
            head = next;
        }
    }

    void swap(ErrorStack& other) noexcept
    {
        std::swap(head, other.head);
        std::swap(m_size, other.m_size);
    }

    ErrorStack(ErrorStack&& other) : head(other.head), m_size(other.m_size)
    {
        other.head = 0;
        other.m_size = 0;
    }

    ErrorStack& operator==(ErrorStack&& other)
    {
        swap(other);
        return *this;
    }
};

// For argument dependent lookup
inline void swap(ErrorStack& s1, ErrorStack& s2) { s1.swap(s2); }

class ParseStatus
{
private:
    ErrorStack m_stack;
    std::size_t m_offset;
    int m_code;

public:
    explicit ParseStatus() : m_stack(), m_offset(), m_code() {}

    void set_result(int err, std::size_t off)
    {
        m_code = err;
        m_offset = off;
    }

    int error_code() const { return m_code; }

    std::size_t offset() const { return m_offset; }

    std::string short_description() const;

    ErrorStack& error_stack() { return m_stack; }

    const ErrorStack& error_stack() const { return m_stack; }

    typedef ErrorStack::const_iterator const_iterator;

    const_iterator begin() const { return m_stack.begin(); }

    const_iterator end() const { return m_stack.end(); }

    bool has_error() const { return m_code != 0 || !m_stack.empty(); }

    std::string description() const;

    void swap(ParseStatus& other) noexcept
    {
        std::swap(m_code, other.m_code);
        std::swap(m_offset, other.m_offset);
        m_stack.swap(other.m_stack);
    }

    bool operator!() const { return has_error(); }

    explicit operator bool() const { return !has_error(); }

    ParseStatus(ParseStatus&& other) noexcept : m_stack(), m_offset(), m_code() { swap(other); }

    ParseStatus& operator==(ParseStatus&& other) noexcept
    {
        swap(other);
        return *this;
    }
};

// For argument dependent lookup
inline void swap(ParseStatus& r1, ParseStatus& r2) { r1.swap(r2); }
}
