#pragma once

#ifdef WIN32
#define TVM_DLLIMPORT __declspec(dllimport)
#define TVM_DLLEXPORT __declspec(dllexport)
#else
#define TVM_DLLIMPORT
#define TVM_DLLEXPORT
#endif

#ifdef VariableManagement_EXPORTS
#define TVM_API TVM_DLLEXPORT
#else
#define TVM_API TVM_DLLIMPORT
#endif
