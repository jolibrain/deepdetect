
if HAVE_GTEST
TESTS = $(check_PROGRAMS)
check_PROGRAMS = ut_apidata
ut_apidata_SOURCES = ut-apidata.cc
endif

AM_CPPFLAGS=-I../src
AM_CXXFLAGS=-g -Wall -fopenmp
AM_LDFLAGS=-L../src
if HAVE_GTEST
AM_LDFLAGS += -lgtest -lgtest_main
endif
LDADD=-lddetect -lm $(GFLAGS_LIBS) -lcaffe -lopencv_core -lboost_regex
