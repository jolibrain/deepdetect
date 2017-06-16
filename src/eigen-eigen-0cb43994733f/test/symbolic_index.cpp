// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifdef EIGEN_TEST_PART_2
#define EIGEN_MAX_CPP_VER 03
#endif

#include "main.h"

template<typename T>
bool match(const T& xpr, std::string ref, std::string str_xpr = "") {
  EIGEN_UNUSED_VARIABLE(str_xpr);
  std::stringstream str;
  str << xpr;
  if(!(str.str() == ref))
    std::cout << str_xpr << "\n" << xpr << "\n\n";
  return str.str() == ref;
}

#define MATCH(X,R) match(X, R, #X)

template<typename T1,typename T2>
typename internal::enable_if<internal::is_same<T1,T2>::value,bool>::type
is_same_fixed(const T1& a, const T2& b)
{
  return (Index(a) == Index(b));
}

template<typename T1,typename T2>
bool is_same_seq(const T1& a, const T2& b)
{
  bool ok = a.first()==b.first() && a.size() == b.size() && Index(a.incrObject())==Index(b.incrObject());;
  if(!ok)
  {
    std::cerr << "seqN(" << a.first() << ", " << a.size() << ", " << Index(a.incrObject()) << ") != ";
    std::cerr << "seqN(" << b.first() << ", " << b.size() << ", " << Index(b.incrObject()) << ")\n";
  }
  return ok;
}

template<typename T1,typename T2>
typename internal::enable_if<internal::is_same<T1,T2>::value,bool>::type
is_same_type(const T1&, const T2&)
{
  return true;
}

template<typename T1,typename T2>
bool is_same_symb(const T1& a, const T2& b, Index size)
{
  using Eigen::placeholders::last;
  return a.eval(last=size-1) == b.eval(last=size-1);
}


#define VERIFY_EQ_INT(A,B) VERIFY_IS_APPROX(int(A),int(B))

void check_symbolic_index()
{
  using Eigen::placeholders::last;
  using Eigen::placeholders::end;

  Index size=100;

  // First, let's check FixedInt arithmetic:
  VERIFY( is_same_type( (fix<5>()-fix<3>())*fix<9>()/(-fix<3>()), fix<-(5-3)*9/3>() ) );
  VERIFY( is_same_type( (fix<5>()-fix<3>())*fix<9>()/fix<2>(), fix<(5-3)*9/2>() ) );
  VERIFY( is_same_type( fix<9>()/fix<2>(), fix<9/2>() ) );
  VERIFY( is_same_type( fix<9>()%fix<2>(), fix<9%2>() ) );
  VERIFY( is_same_type( fix<9>()&fix<2>(), fix<9&2>() ) );
  VERIFY( is_same_type( fix<9>()|fix<2>(), fix<9|2>() ) );
  VERIFY( is_same_type( fix<9>()/2, int(9/2) ) );

  VERIFY( is_same_symb( end-1, last, size) );
  VERIFY( is_same_symb( end-fix<1>, last, size) );

  VERIFY_IS_EQUAL( ( (last*5-2)/3 ).eval(last=size-1), ((size-1)*5-2)/3 );
  VERIFY_IS_EQUAL( ( (last*fix<5>-fix<2>)/fix<3> ).eval(last=size-1), ((size-1)*5-2)/3 );
  VERIFY_IS_EQUAL( ( -last*end  ).eval(last=size-1), -(size-1)*size );
  VERIFY_IS_EQUAL( ( end-3*last  ).eval(last=size-1), size- 3*(size-1) );
  VERIFY_IS_EQUAL( ( (end-3*last)/end  ).eval(last=size-1), (size- 3*(size-1))/size );

#if EIGEN_HAS_CXX14
  {
    struct x_tag {};  static const Symbolic::SymbolExpr<x_tag> x;
    struct y_tag {};  static const Symbolic::SymbolExpr<y_tag> y;
    struct z_tag {};  static const Symbolic::SymbolExpr<z_tag> z;

    VERIFY_IS_APPROX( int(((x+3)/y+z).eval(x=6,y=3,z=-13)), (6+3)/3+(-13) );
  }
#endif
}

void test_symbolic_index()
{
  CALL_SUBTEST_1( check_symbolic_index() );
  CALL_SUBTEST_2( check_symbolic_index() );
}
