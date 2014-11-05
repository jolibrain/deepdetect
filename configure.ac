# configure for deepdetect
AC_INIT(deepdetect,0.1)
AC_CONFIG_SRCDIR(Makefile.am)
AM_INIT_AUTOMAKE
AM_CONFIG_HEADER([dd_config.h])

AC_PROG_INSTALL
AC_CANONICAL_HOST

AC_PROG_CC(g++ c++)
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_LIBTOOL

m4_include([m4/ax_cxx_compile_stdcxx_11.m4])
AX_CXX_COMPILE_STDCXX_11()

AC_LANG_PUSH([C++])

# Check if there is google-gflags library installed.
AC_ARG_WITH(gflags, AS_HELP_STRING[--with-gflags=GFLAGS_DIR],
  GFLAGS_CFLAGS="-I${with_gflags}/include"
  GFLAGS_LIBS="-L${with_gflags}/lib -lgflags"
  CFLAGS="$CFLAGS $GFLAGS_CFLAGS"
  LIBS="$LIBS $GFLAGS_LIBS"
)
AC_CHECK_LIB(gflags, main, ac_cv_have_libgflags=1, ac_cv_have_libgflags=0)
if test x"$ac_cv_have_libgflags" = x"1"; then
   AC_DEFINE(HAVE_LIB_GFLAGS, 1, [define if you have google gflags library])
   if test x"$GFLAGS_LIBS" = x""; then
      GFLAGS_LIBS="-lgflags"
   fi
else
  GFLAGS_CFLAGS=
  GFLAGS_LIBS=
fi
AC_SUBST(GFLAGS_LIBS)
AM_CONDITIONAL(HAVE_GFLAGS, test x"$ac_cv_have_libgflags" = x"1")

AC_OUTPUT(Makefile src/Makefile main/Makefile)
