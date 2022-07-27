// @author Josias
// struct containing 4 floats to use as 2x2 matrix on GPU
/*
struct matrix2
{
  float a00, a10;  // standard names for components
  float a01, a11;  // standard names for components
};
// \@author Josias
*/
#pragma once

#include "nvmath/nvmath_types.h"

using namespace nvmath;

template <class T>
struct matrix2;

namespace custommath {

template <class T>
struct matrix2
{
  matrix2()
      : matrix2(0, 0, 0, 0)
  {
  }
  matrix2(int one) { identity(); }
  matrix2(const T* array) { memcpy(mat_array, array, sizeof(T) * 4); }
  matrix2(const matrix2<T>& M) { memcpy(mat_array, M.mat_array, sizeof(T) * 4); }
  matrix2(const T& f0, const T& f1, const T& f2, const T& f3)
      : a00(f0)
      , a10(f1)
      , a01(f2)
      , a11(f3)
  {
  }

  matrix2<T>& identity()
  {
    mat_array[0] = T(1);
    mat_array[1] = T(0);
    mat_array[2] = T(0);
    mat_array[3] = T(1);

    return *this;
  }

  const vector2<T> col(const int i) const { return vector2<T>(&mat_array[i * 2]); }

  const vector2<T> row(const int i) const { return vector2<T>(mat_array[i], mat_array[i + 2]); }

  const vector2<T> operator[](int i) const { return vector2<T>(mat_array[i], mat_array[i + 2]); }

  const T& operator()(const int& i, const int& j) const { return mat_array[j * 2 + i]; }

  T& operator()(const int& i, const int& j) { return mat_array[j * 2 + i]; }

  matrix2<T>& operator*=(const T& lambda)
  {
    for(int i = 0; i < 4; ++i)
      mat_array[i] *= lambda;
    return *this;
  }

  matrix2<T> operator*(const matrix2<T>&) const;

  matrix2<T>& operator*=(const matrix2<T>& M)
  {
    *this = mult(*this, M);
    return *this;
  }


  matrix2<T>& operator-=(const matrix2<T>& M)
  {
    for(int i = 0; i < 4; ++i)
      mat_array[i] -= M.mat_array[i];
    return *this;
  }

  matrix2<T>& set_row(int i, const vector2<T>& v)
  {
    mat_array[i]     = v.x;
    mat_array[i + 2] = v.y;
    return *this;
  }

  matrix2<T>& set_col(int i, const vector2<T>& v)
  {
    mat_array[i * 2]     = v.x;
    mat_array[i * 2 + 1] = v.y;
    return *this;
  }

  matrix2<T>& set_rot(const T& theta, const vector2<T>& v);
  matrix2<T>& set_rot(const vector2<T>& u, const vector2<T>& v);

  // Matrix norms...
  // Compute || M ||
  //                1
  T norm_one();

  // Compute || M ||
  //                +inf
  T norm_inf();

  union
  {
    struct
    {
      T a00, a10;  // standard names for components
      T a01, a11;  // standard names for components
    };
    T mat_array[4];  // array access
  };

  T*       get_value() { return mat_array; }
  const T* get_value() const { return mat_array; }
};  //struct matrix2
//typedef matrix2<nv_scalar> mat2f;
}  // namespace custommath