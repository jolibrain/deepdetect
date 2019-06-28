// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mustache.h"

#include "ext/rapidjson/stringbuffer.h"
#include "ext/rapidjson/writer.h"

#include <iostream>
#include <fstream>
#include <vector>

#ifdef WIN32
#define strcasecmp _stricmp
#endif

#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>

using namespace rapidjson;
using namespace std;
using namespace boost::algorithm;

namespace mustache {

// TODO:
// # Handle malformed templates better
// # Better support for reading templates from files

enum TagOperator {
  SUBSTITUTION,
  SECTION_START,
  NEGATED_SECTION_START,
  PREDICATE_SECTION_START,
  SECTION_END,
  PARTIAL,
  COMMENT,
  LENGTH,
  EQUALITY,
  INEQUALITY,
  NONE
};

TagOperator GetOperator(const string& tag) {
  if (tag.size() == 0) return SUBSTITUTION;
  switch (tag[0]) {
    case '#': return SECTION_START;
    case '^': return NEGATED_SECTION_START;
    case '?': return PREDICATE_SECTION_START;
    case '/': return SECTION_END;
    case '>': return PARTIAL;
    case '!':
      if (tag.size() == 1 || tag[1] != '=') return COMMENT;
      return INEQUALITY;
    case '%': return LENGTH;
    case '=': return EQUALITY;
    default: return SUBSTITUTION;
  }
}

int EvaluateTag(const string& document, const string& document_root, int idx,
    const Value* context, TagOperator tag, const string& tag_name, const string& tag_arg,
    bool is_triple, stringstream* out);

void EscapeHtml(const string& in, stringstream *out) {
  BOOST_FOREACH(const char& c, in) {
    switch (c) {
      case '&': (*out) << "&amp;";
        break;
      case '"': (*out) << "&quot;";
        break;
      case '\'': (*out) << "&apos;";
        break;
      case '<': (*out) << "&lt;";
        break;
      case '>': (*out) << "&gt;";
        break;
      default: (*out) << c;
        break;
    }
  }
}

// Breaks a dotted path into individual components. One wrinkle, which stops this from
// being a simple split() is that we allow path components to be quoted, e.g.: "foo".bar,
// and any '.' characters inside those quoted sections aren't considered to be
// delimiters. This is to allow Json keys that contain periods.
void FindJsonPathComponents(const string& path, vector<string>* components) {
  bool in_quote = false;
  bool escape_this_char = false;
  int start = 0;
  for (int i = start; i < path.size(); ++i) {
    if (path[i] == '"' && !escape_this_char) in_quote = !in_quote;
    if (path[i] == '.' && !escape_this_char && !in_quote) {
      // Current char == delimiter and not escaped and not in a quote pair => found a
      // component
      if (i - start > 0) {
        if (path[start] == '"' && path[(i - 1) - start] == '"') {
          if (i - start > 3) {
            components->push_back(path.substr(start + 1, i - (start + 2)));
          }
        } else {
          components->push_back(path.substr(start, i - start));
        }
        start = i + 1;
      }
    }

    escape_this_char = (path[i] == '\\' && !escape_this_char);
  }

  if (path.size() - start > 0) {
    if (path[start] == '"' && path[(path.size() - 1) - start] == '"') {
      if (path.size() - start > 3) {
        components->push_back(path.substr(start + 1, path.size() - (start + 2)));
      }
    } else {
      components->push_back(path.substr(start, path.size() - start));
    }
  }
}

// Looks up the json entity at 'path' in 'parent_context', and places it in 'resolved'. If
// the entity does not exist (i.e. the path is invalid), 'resolved' will be set to NULL.
void ResolveJsonContext(const string& path, const Value& parent_context,
    const Value** resolved) {
  if (path == ".") {
    *resolved = &parent_context;
    return;
  }
  vector<string> components;
  FindJsonPathComponents(path, &components);
  const Value* cur = &parent_context;
  BOOST_FOREACH(const string& c, components) {
    if (cur->IsObject() && cur->HasMember(c.c_str())) {
      cur = &(*cur)[c.c_str()];
    } else {
      *resolved = NULL;
      return;
    }
  }

  *resolved = cur;
}

int FindNextTag(const string& document, int idx, TagOperator* tag_op, string* tag_name,
    string* tag_arg, bool* is_triple, stringstream* out) {
  *tag_op = NONE;
  while (idx < document.size()) {
    if (document[idx] == '{' && idx < (document.size() - 3) && document[idx + 1] == '{') {
      if (document[idx + 2] == '{') {
        idx += 3;
        *is_triple = true;
      } else {
        *is_triple = false;
        idx += 2; // Now at start of template expression
      }
      stringstream expr;
      while (idx < document.size()) {
        if (document[idx] != '}') {
          expr << document[idx];
          ++idx;
        } else {
          if (!*is_triple && idx < document.size() - 1 && document[idx + 1] == '}') {
            ++idx;
            break;
          } else if (*is_triple && idx < document.size() - 2 && document[idx + 1] == '}'
              && document[idx + 2] == '}') {
            idx += 2;
            break;
          } else {
            expr << '}'; // XXX: potential infinite loop here
	  }
        }
      }

      string key = expr.str();
      trim(key);
      if (key != ".") trim_if(key, is_any_of("."));
      if (key.size() == 0) continue;
      *tag_op = GetOperator(key);
      if (*tag_op != SUBSTITUTION) {
        int len = *tag_op == INEQUALITY ? 2 : 1;
        key = key.substr(len);
        trim(key);
      }
      if (key.size() == 0) continue;

      if (*tag_op == EQUALITY || *tag_op == INEQUALITY) {
        // Find an argument
        vector<string> components;
        split(components, key, is_any_of(" "));
        key = components[0];
        components.erase(components.begin());
        *tag_arg = join(components, " ");
      }

      *tag_name = key;
      return ++idx;
    } else {
      if (out != NULL) (*out) << document[idx];
    }
    ++idx;
  }
  return idx;
}

// Evaluates a [PREDICATE_|NEGATED_]SECTION_START / SECTION_END pair by evaluating the tag
// in 'parent_context'. False or non-existant values cause the entire section to be
// skipped. True values cause the section to be evaluated as though it were a normal
// section, but with the parent context being the root context for that section.
//
// If 'is_negation' is true, the behaviour is the opposite of the above: false values
// cause the section to be normally evaluated etc.
int EvaluateSection(const string& document, const string& document_root, int idx,
    const Value* parent_context, TagOperator op, const string& tag_name,
    const string& tag_arg, stringstream* out) {
  // Precondition: idx is the immediate next character after an opening {{ #tag_name }}
  const Value* context;
  ResolveJsonContext(tag_name, *parent_context, &context);

  // If we a) cannot resolve the context from the tag name or b) the context evaluates to
  // false, we should skip the contents of the template until a closing {{/tag_name}}.
  bool skip_contents = false;

  if (op == NEGATED_SECTION_START || op == PREDICATE_SECTION_START ||
      op == SECTION_START) {
    skip_contents = (context == NULL || context->IsFalse());

    // If the tag is a negative block (i.e. {{^tag_name}}), do the opposite: if the
    // context exists and is true, skip the contents, else echo them.
    if (op == NEGATED_SECTION_START) {
      context = parent_context;
      skip_contents = !skip_contents;
    } else if (op == PREDICATE_SECTION_START) {
      context = parent_context;
    }
  } else if (op == INEQUALITY || op == EQUALITY) {
    skip_contents = (context == NULL || !context->IsString() ||
        strcasecmp(context->GetString(), tag_arg.c_str()) != 0);
    if (op == INEQUALITY) skip_contents = !skip_contents;
    context = parent_context;
  }

  vector<const Value*> values;
  if (!skip_contents && context != NULL && context->IsArray()) {
    for (int i = 0; i < context->Size(); ++i) {
      values.push_back(&(*context)[i]);
    }
  } else {
    values.push_back(skip_contents ? NULL : context);
  }
  if (values.size() == 0) {
    skip_contents = true;
    values.push_back(NULL);
  }

  int start_idx = idx;
  BOOST_FOREACH(const Value* v, values) {
    idx = start_idx;
    while (idx < document.size()) {
      TagOperator tag_op;
      string tag_arg;
      string next_tag_name;
      bool is_triple;
      idx = FindNextTag(document, idx, &tag_op, &next_tag_name, &tag_arg, &is_triple,
          skip_contents ? NULL : out);

      if (idx > document.size()) return idx;
      if (tag_op == SECTION_END && next_tag_name == tag_name) {
        break;
      }

      // Don't need to evaluate any templates if we're skipping the contents
      if (!skip_contents) {
        idx = EvaluateTag(document, document_root, idx, v, tag_op, next_tag_name,
            tag_arg, is_triple, out);
      }
    }
  }
  return idx;
}

// Evaluates a SUBSTITUTION tag, by replacing its contents with the value of the tag's
// name in 'parent_context'.
int EvaluateSubstitution(const string& document, const int idx,
    const Value* parent_context, const string& tag_name, bool is_triple,
    stringstream* out) {
  const Value* context;
  ResolveJsonContext(tag_name, *parent_context, &context);
  if (context == NULL) return idx;
  if (context->IsString()) {
    if (!is_triple) {
      EscapeHtml(context->GetString(), out);
    } else {
      // TODO: Triple {{{ means don't escape
      (*out) << context->GetString();
    }
  } else if (context->IsInt()) {
    (*out) << context->GetInt();
  } else if (context->IsDouble()) {
    (*out) << context->GetDouble();
  } else if (context->IsBool()) {
    (*out) << boolalpha << context->GetBool();
  }
  return idx;
}

// Evaluates a LENGTH tag by replacing its contents with the type-dependent 'size' of the
// value.
int EvaluateLength(const string& document, const int idx, const Value* parent_context,
    const string& tag_name, stringstream* out) {
  const Value* context;
  ResolveJsonContext(tag_name, *parent_context, &context);
  if (context == NULL) return idx;
  if (context->IsArray()) {
    (*out) << context->Size();
  } else if (context->IsString()) {
    (*out) << context->GetStringLength();
  };

  return idx;
}

// Evaluates a 'partial' template by reading it fully from disk, then rendering it
// directly into the current output with the current context.
//
// TODO: This could obviously be more efficient (and there are lots of file accesses in a
// long list context).
void EvaluatePartial(const string& tag_name, const string& document_root,
    const Value* parent_context, stringstream* out) {
  stringstream ss;
  ss << document_root << tag_name;
  ifstream tmpl(ss.str().c_str());
  if (!tmpl.is_open()) {
    ss << ".mustache";
    tmpl.open(ss.str().c_str());
    if (!tmpl.is_open()) return;
  }
  stringstream file_ss;
  file_ss << tmpl.rdbuf();
  RenderTemplate(file_ss.str(), document_root, *parent_context, out);
}

// Given a tag name, and its operator, evaluate the tag in the given context and write the
// output to 'out'. The heavy-lifting is delegated to specific Evaluate*()
// methods. Returns the new cursor position within 'document', or -1 on error.
int EvaluateTag(const string& document, const string& document_root, int idx,
    const Value* context, TagOperator tag, const string& tag_name, const string& tag_arg,
    bool is_triple, stringstream* out) {
  if (idx == -1) return idx;
  switch (tag) {
    case SECTION_START:
    case PREDICATE_SECTION_START:
    case NEGATED_SECTION_START:
    case EQUALITY:
    case INEQUALITY:
      return EvaluateSection(document, document_root, idx, context, tag, tag_name,
          tag_arg, out);
    case SUBSTITUTION:
      return EvaluateSubstitution(document, idx, context, tag_name, is_triple, out);
    case COMMENT:
      return idx; // Ignored
    case PARTIAL:
      EvaluatePartial(tag_name, document_root, context, out);
      return idx;
    case LENGTH:
      return EvaluateLength(document, idx, context, tag_name, out);
    case NONE:
      return idx; // No tag was found
    default:
      cout << "Unknown tag: " << tag << endl;
      return -1;
  }
}

void RenderTemplate(const string& document, const string& document_root,
    const Value& context, stringstream* out) {
  int idx = 0;
  while (idx < document.size() && idx != -1) {
    string tag_name;
    string tag_arg;
    TagOperator tag_op;
    bool is_triple;
    idx = FindNextTag(document, idx, &tag_op, &tag_name, &tag_arg, &is_triple, out);
    idx = EvaluateTag(document, document_root, idx, &context, tag_op, tag_name, tag_arg, is_triple,
        out);
  }
}

}
