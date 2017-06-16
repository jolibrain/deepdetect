# libxsmm support.
# libxsmm provides matrix multiplication kernels optimized for
# the latest Intel architectures.
# Download the library from https://github.com/hfp/libxsmm
# Compile with make BLAS=0

if (LIBXSMM)
  set(XSMM_FIND_QUIETLY TRUE)
  set(XSMM_INCLUDES ${LIBXSMM}/include)
  set(XSMM_LIBRARIES ${LIBXSMM}/lib)
endif (LIBXSMM)

find_path(LIBXSMM 
  NAMES 
  libxsmm.h 
  PATHS 
  $ENV{XSMMDIR}/include 
  ${INCLUDE_INSTALL_DIR} 
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XSMM DEFAULT_MSG
                                  LIBXSMM)

mark_as_advanced(LIBXSMM)
