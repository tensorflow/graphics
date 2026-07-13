
#ifndef _EXTMODULE_H
#define _EXTMODULE_H

#include <Python.h>
#include <stdexcept>

// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL mcubes_PyArray_API
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"

#include <complex>

template<class T>
struct numpy_typemap;

#define define_numpy_type(ctype, dtype) \
    template<> \
    struct numpy_typemap<ctype> \
    {static const int type = dtype;};

define_numpy_type(bool, NPY_BOOL);
define_numpy_type(char, NPY_BYTE);
define_numpy_type(short, NPY_SHORT);
define_numpy_type(int, NPY_INT);
define_numpy_type(long, NPY_LONG);
define_numpy_type(long long, NPY_LONGLONG);
define_numpy_type(unsigned char, NPY_UBYTE);
define_numpy_type(unsigned short, NPY_USHORT);
define_numpy_type(unsigned int, NPY_UINT);
define_numpy_type(unsigned long, NPY_ULONG);
define_numpy_type(unsigned long long, NPY_ULONGLONG);
define_numpy_type(float, NPY_FLOAT);
define_numpy_type(double, NPY_DOUBLE);
define_numpy_type(long double, NPY_LONGDOUBLE);
define_numpy_type(std::complex<float>, NPY_CFLOAT);
define_numpy_type(std::complex<double>, NPY_CDOUBLE);
define_numpy_type(std::complex<long double>, NPY_CLONGDOUBLE);

template<typename T>
T PyArray_SafeGet(const PyArrayObject* aobj, const npy_intp* indaux)
{
    // HORROR.
    npy_intp* ind = const_cast<npy_intp*>(indaux);
    void* ptr = PyArray_GetPtr(const_cast<PyArrayObject*>(aobj), ind);
    switch(PyArray_TYPE(aobj))
    {
    case NPY_BOOL:
        return static_cast<T>(*reinterpret_cast<bool*>(ptr));
    case NPY_BYTE:
        return static_cast<T>(*reinterpret_cast<char*>(ptr));
    case NPY_SHORT:
        return static_cast<T>(*reinterpret_cast<short*>(ptr));
    case NPY_INT:
        return static_cast<T>(*reinterpret_cast<int*>(ptr));
    case NPY_LONG:
        return static_cast<T>(*reinterpret_cast<long*>(ptr));
    case NPY_LONGLONG:
        return static_cast<T>(*reinterpret_cast<long long*>(ptr));
    case NPY_UBYTE:
        return static_cast<T>(*reinterpret_cast<unsigned char*>(ptr));
    case NPY_USHORT:
        return static_cast<T>(*reinterpret_cast<unsigned short*>(ptr));
    case NPY_UINT:
        return static_cast<T>(*reinterpret_cast<unsigned int*>(ptr));
    case NPY_ULONG:
        return static_cast<T>(*reinterpret_cast<unsigned long*>(ptr));
    case NPY_ULONGLONG:
        return static_cast<T>(*reinterpret_cast<unsigned long long*>(ptr));
    case NPY_FLOAT:
        return static_cast<T>(*reinterpret_cast<float*>(ptr));
    case NPY_DOUBLE:
        return static_cast<T>(*reinterpret_cast<double*>(ptr));
    case NPY_LONGDOUBLE:
        return static_cast<T>(*reinterpret_cast<long double*>(ptr));
    default:
        throw std::runtime_error("data type not supported");
    }
}

template<typename T>
T PyArray_SafeSet(PyArrayObject* aobj, const npy_intp* indaux, const T& value)
{
    // HORROR.
    npy_intp* ind = const_cast<npy_intp*>(indaux);
    void* ptr = PyArray_GetPtr(aobj, ind);
    switch(PyArray_TYPE(aobj))
    {
    case NPY_BOOL:
        *reinterpret_cast<bool*>(ptr) = static_cast<bool>(value);
        break;
    case NPY_BYTE:
        *reinterpret_cast<char*>(ptr) = static_cast<char>(value);
        break;
    case NPY_SHORT:
        *reinterpret_cast<short*>(ptr) = static_cast<short>(value);
        break;
    case NPY_INT:
        *reinterpret_cast<int*>(ptr) = static_cast<int>(value);
        break;
    case NPY_LONG:
        *reinterpret_cast<long*>(ptr) = static_cast<long>(value);
        break;
    case NPY_LONGLONG:
        *reinterpret_cast<long long*>(ptr) = static_cast<long long>(value);
        break;
    case NPY_UBYTE:
        *reinterpret_cast<unsigned char*>(ptr) = static_cast<unsigned char>(value);
        break;
    case NPY_USHORT:
        *reinterpret_cast<unsigned short*>(ptr) = static_cast<unsigned short>(value);
        break;
    case NPY_UINT:
        *reinterpret_cast<unsigned int*>(ptr) = static_cast<unsigned int>(value);
        break;
    case NPY_ULONG:
        *reinterpret_cast<unsigned long*>(ptr) = static_cast<unsigned long>(value);
        break;
    case NPY_ULONGLONG:
        *reinterpret_cast<unsigned long long*>(ptr) = static_cast<unsigned long long>(value);
        break;
    case NPY_FLOAT:
        *reinterpret_cast<float*>(ptr) = static_cast<float>(value);
        break;
    case NPY_DOUBLE:
        *reinterpret_cast<double*>(ptr) = static_cast<double>(value);
        break;
    case NPY_LONGDOUBLE:
        *reinterpret_cast<long double*>(ptr) = static_cast<long double>(value);
        break;
    default:
        throw std::runtime_error("data type not supported");
    }
}

#endif
