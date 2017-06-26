#pragma once

# if defined _WIN32 || defined __CYGWIN__
// On Microsoft Windows, use dllimport and dllexport to tag symbols.
#  define TVM_DLLIMPORT __declspec(dllimport)
#  define TVM_DLLEXPORT __declspec(dllexport)
#  define TVM_DLLLOCAL
# else
// On Linux, for GCC >= 4, tag symbols using GCC extension.
#  if __GNUC__ >= 4
#   define TVM_DLLIMPORT __attribute__ ((visibility("default")))
#   define TVM_DLLEXPORT __attribute__ ((visibility("default")))
#   define TVM_DLLLOCAL  __attribute__ ((visibility("hidden")))
#  else
// Otherwise (GCC < 4 or another compiler is used), export everything.
#   define TVM_DLLIMPORT
#   define TVM_DLLEXPORT
#   define TVM_DLLLOCAL
#  endif // __GNUC__ >= 4
# endif // defined _WIN32 || defined __CYGWIN__

# ifdef TVM_STATIC
// If one is using the library statically, get rid of
// extra information.
#  define TVM_DLLAPI
#  define TVM_LOCAL
# else
// Depending on whether one is building or using the
// library define DLLAPI to import or export.
#  ifdef variablemanagement_EXPORTS
#   define TVM_DLLAPI TVM_DLLEXPORT
#  else
#   define TVM_DLLAPI TVM_DLLIMPORT
#  endif // TVM_EXPORTS
#  define TVM_LOCAL TVM_DLLLOCAL
# endif // TVM_STATIC
