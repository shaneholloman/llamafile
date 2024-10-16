/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf is_keyword_cxx.gperf  */
/* Computed positions: -k'1,3,5,$' */

#if !((' ' == 32) && ('!' == 33) && ('"' == 34) && ('#' == 35) \
      && ('%' == 37) && ('&' == 38) && ('\'' == 39) && ('(' == 40) \
      && (')' == 41) && ('*' == 42) && ('+' == 43) && (',' == 44) \
      && ('-' == 45) && ('.' == 46) && ('/' == 47) && ('0' == 48) \
      && ('1' == 49) && ('2' == 50) && ('3' == 51) && ('4' == 52) \
      && ('5' == 53) && ('6' == 54) && ('7' == 55) && ('8' == 56) \
      && ('9' == 57) && (':' == 58) && (';' == 59) && ('<' == 60) \
      && ('=' == 61) && ('>' == 62) && ('?' == 63) && ('A' == 65) \
      && ('B' == 66) && ('C' == 67) && ('D' == 68) && ('E' == 69) \
      && ('F' == 70) && ('G' == 71) && ('H' == 72) && ('I' == 73) \
      && ('J' == 74) && ('K' == 75) && ('L' == 76) && ('M' == 77) \
      && ('N' == 78) && ('O' == 79) && ('P' == 80) && ('Q' == 81) \
      && ('R' == 82) && ('S' == 83) && ('T' == 84) && ('U' == 85) \
      && ('V' == 86) && ('W' == 87) && ('X' == 88) && ('Y' == 89) \
      && ('Z' == 90) && ('[' == 91) && ('\\' == 92) && (']' == 93) \
      && ('^' == 94) && ('_' == 95) && ('a' == 97) && ('b' == 98) \
      && ('c' == 99) && ('d' == 100) && ('e' == 101) && ('f' == 102) \
      && ('g' == 103) && ('h' == 104) && ('i' == 105) && ('j' == 106) \
      && ('k' == 107) && ('l' == 108) && ('m' == 109) && ('n' == 110) \
      && ('o' == 111) && ('p' == 112) && ('q' == 113) && ('r' == 114) \
      && ('s' == 115) && ('t' == 116) && ('u' == 117) && ('v' == 118) \
      && ('w' == 119) && ('x' == 120) && ('y' == 121) && ('z' == 122) \
      && ('{' == 123) && ('|' == 124) && ('}' == 125) && ('~' == 126))
/* The character set is not based on ISO-646.  */
#error "gperf generated tables don't work with this execution character set. Please report a bug to <bug-gperf@gnu.org>."
#endif

#line 1 "is_keyword_cxx.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 148
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 19
#define MIN_HASH_VALUE 3
#define MAX_HASH_VALUE 361
/* maximum key range = 359, duplicates = 0 */

#ifdef __GNUC__
__inline
#else
#ifdef __cplusplus
inline
#endif
#endif
static unsigned int
hash (register const char *str, register size_t len)
{
  static const unsigned short asso_values[] =
    {
      362, 362, 362, 362, 362, 362, 362, 362, 362, 362,
      362, 362, 362, 362, 362, 362, 362, 362, 362, 362,
      362, 362, 362, 362, 362, 362, 362, 362, 362, 362,
      362, 362,  95, 362, 362,  15, 362, 362, 362, 362,
      362, 362, 362, 362, 362, 362, 362, 362,   5,  70,
      140, 125,  75, 362,  30, 362,  15, 362, 362, 362,
      362, 362, 362, 362, 362, 362, 362, 362, 362,  15,
        0, 362, 362, 362, 362, 362, 362, 362,   0, 362,
       10, 362, 362, 362, 362, 362, 362, 362, 362, 362,
      362, 362, 362, 362, 362,  35, 362,  10, 175,  30,
      100,   0,   5,  25,  20,  10, 362,   0,   5,  55,
       15, 110, 140,  40,  50,  95,   0,  20,  20,  30,
       50,   0,   0, 362, 362, 362, 362, 362, 362, 362,
      362, 362, 362, 362, 362, 362, 362, 362, 362, 362,
      362, 362, 362, 362, 362, 362, 362, 362, 362, 362,
      362, 362, 362, 362, 362, 362, 362, 362, 362, 362,
      362, 362, 362, 362, 362, 362, 362, 362, 362, 362,
      362, 362, 362, 362, 362, 362, 362, 362, 362, 362,
      362, 362, 362, 362, 362, 362, 362, 362, 362, 362,
      362, 362, 362, 362, 362, 362, 362, 362, 362, 362,
      362, 362, 362, 362, 362, 362, 362, 362, 362, 362,
      362, 362, 362, 362, 362, 362, 362, 362, 362, 362,
      362, 362, 362, 362, 362, 362, 362, 362, 362, 362,
      362, 362, 362, 362, 362, 362, 362, 362, 362, 362,
      362, 362, 362, 362, 362, 362, 362, 362, 362, 362,
      362, 362, 362, 362, 362, 362
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[4]];
      /*FALLTHROUGH*/
      case 4:
      case 3:
        hval += asso_values[(unsigned char)str[2]];
      /*FALLTHROUGH*/
      case 2:
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval + asso_values[(unsigned char)str[len - 1]];
}

const char *
is_keyword_cxx (register const char *str, register size_t len)
{
  static const char * const wordlist[] =
    {
      "", "", "",
      "try",
      "", "", "", "", "", "", "", "", "",
      "int",
      "",
      "false",
      "",
      "if",
      "not",
      "", "", "", "", "",
      "true",
      "#else",
      "", "",
      "#if",
      "",
      "#line",
      "#ifdef",
      "#define",
      "volatile",
      "",
      "#elif",
      "inline",
      "",
      "#elifdef",
      "#elifndef",
      "",
      "#undef",
      "",
      "#include",
      "",
      "while",
      "",
      "alignof",
      "#include_next",
      "long",
      "const",
      "#endif",
      "concept",
      "noexcept",
      "constinit",
      "const_cast",
      "__attribute",
      "",
      "__inline",
      "consteval",
      "",
      "not_eq",
      "char8_t",
      "continue",
      "",
      "union",
      "__null",
      "",
      "template",
      "__alignof",
      "__volatile",
      "extern",
      "",
      "#warning",
      "_Float128",
      "catch",
      "reinterpret_cast",
      "thread_local",
      "new",
      "enum",
      "", "",
      "__FUNCTION__",
      "__extension__",
      "",
      "using",
      "__imag",
      "__const",
      "_Float16",
      "co_return",
      "", "", "",
      "__attribute__",
      "char",
      "__inline__",
      "", "",
      "__func__",
      "else",
      "compl",
      "__alignof__",
      "virtual",
      "co_await",
      "constexpr",
      "",
      "__real",
      "wchar_t",
      "for",
      "this",
      "",
      "delete",
      "",
      "reflexpr",
      "__PRETTY_FUNCTION__",
      "throw",
      "", "",
      "char16_t",
      "",
      "float",
      "return",
      "",
      "asm",
      "auto",
      "",
      "static_cast",
      "",
      "static_assert",
      "case",
      "",
      "double",
      "default",
      "_Float64",
      "void",
      "",
      "friend",
      "alignas",
      "decltype",
      "goto",
      "", "",
      "__asm__",
      "atomic_commit",
      "",
      "atomic_noexcept",
      "xor_eq",
      "",
      "atomic_cancel",
      "", "",
      "static",
      "",
      "xor",
      "", "",
      "and_eq",
      "",
      "explicit",
      "__imag__ ",
      "__asm",
      "switch",
      "or",
      "typename",
      "__float80",
      "",
      "__complex__",
      "private",
      "__volatile__ ",
      "", "", "", "",
      "char32_t",
      "namespace",
      "",
      "#embed",
      "",
      "operator",
      "__complex",
      "break",
      "struct",
      "dynamic_cast",
      "co_yield",
      "", "", "", "",
      "__typeof",
      "",
      "__restrict",
      "", "",
      "__thread",
      "", "",
      "export",
      "",
      "_Float32",
      "__real__ ",
      "__signed__",
      "", "",
      "requires",
      "", "", "", "", "", "",
      "short",
      "",
      "do",
      "and",
      "", "",
      "sizeof",
      "nullptr",
      "", "", "", "", "", "", "", "",
      "signed",
      "__restrict__",
      "register",
      "",
      "or_eq",
      "", "", "", "",
      "class",
      "",
      "mutable",
      "", "", "", "",
      "synchronized",
      "__builtin_offsetof",
      "", "", "", "",
      "unsigned",
      "", "", "",
      "typedef",
      "", "", "",
      "typeid",
      "", "",
      "__label__",
      "",
      "__builtin_va_arg",
      "",
      "__signed",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "",
      "bitor",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "",
      "bool",
      "",
      "bitand",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "",
      "__bf16",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "",
      "protected",
      "",
      "public"
    };

  if (len <= MAX_WORD_LENGTH && len >= MIN_WORD_LENGTH)
    {
      register unsigned int key = hash (str, len);

      if (key <= MAX_HASH_VALUE)
        {
          register const char *s = wordlist[key];

          if (*str == *s && !strncmp (str + 1, s + 1, len - 1) && s[len] == '\0')
            return s;
        }
    }
  return 0;
}
