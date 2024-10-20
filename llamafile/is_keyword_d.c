/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf llamafile/is_keyword_d.gperf  */
/* Computed positions: -k'1,3,$' */

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

#line 1 "llamafile/is_keyword_d.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 110
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 19
#define MIN_HASH_VALUE 3
#define MAX_HASH_VALUE 193
/* maximum key range = 191, duplicates = 0 */

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
  static const unsigned char asso_values[] =
    {
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
        0, 194, 194, 194, 194, 194,  25,  85, 194, 194,
        0, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194,  30, 194,  25, 110,  15,
       35,   5,  20,  35,   5,   0, 194,  30,  60,  75,
       35,   5,   0, 194,  85,   5,   0,  65,  30,  50,
       20,  40,  10, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
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
is_keyword_d (register const char *str, register size_t len)
{
  static const char * const wordlist[] =
    {
      "", "", "",
      "int",
      "", "",
      "import",
      "is",
      "out",
      "this",
      "inout",
      "export",
      "private",
      "",
      "interface",
      "short",
      "switch",
      "idouble",
      "",
      "else",
      "scope",
      "",
      "if",
      "override",
      "cast",
      "catch",
      "typeof",
      "package",
      "",
      "case",
      "float",
      "",
      "cdouble",
      "",
      "auto",
      "alias",
      "assert",
      "in",
      "abstract",
      "invariant",
      "",
      "typeid",
      "do",
      "__traits",
      "goto",
      "",
      "extern",
      "__parameters",
      "",
      "protected",
      "class",
      "static",
      "", "",
      "cent",
      "const",
      "pragma",
      "", "",
      "with",
      "while",
      "",
      "default",
      "continue",
      "",
      "align",
      "ifloat",
      "",
      "__FILE__",
      "void",
      "ireal",
      "shared",
      "__FUNCTION__",
      "unittest",
      "true",
      "ucent",
      "ushort",
      "",
      "__FILE_FULL_PATH__",
      "__PRETTY_FUNCTION__",
      "deprecated",
      "cfloat",
      "",
      "try",
      "",
      "creal",
      "",
      "synchronized",
      "template",
      "immutable",
      "false",
      "",
      "nothrow",
      "__LINE__",
      "pure",
      "super",
      "struct",
      "",
      "function",
      "",
      "macro",
      "",
      "finally",
      "",
      "uint",
      "union",
      "delete",
      "",
      "delegate",
      "__gshared",
      "ulong",
      "double",
      "", "",
      "lazy",
      "ubyte",
      "",
      "foreach",
      "",
      "byte",
      "final",
      "module",
      "", "", "",
      "foreach_reverse",
      "return",
      "",
      "ref",
      "char",
      "dchar",
      "public",
      "", "",
      "long",
      "mixin",
      "", "",
      "new",
      "",
      "throw",
      "", "", "", "",
      "wchar",
      "", "", "",
      "enum",
      "break",
      "", "",
      "__vector",
      "",
      "__MODULE__",
      "",
      "version",
      "",
      "null",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "",
      "real",
      "", "", "",
      "asm",
      "bool",
      "", "", "", "", "",
      "debug",
      "", "", "",
      "body",
      "", "", "",
      "for"
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