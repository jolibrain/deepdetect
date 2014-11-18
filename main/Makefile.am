bin_PROGRAMS=dd
dd_SOURCES=dd.cc

AM_CPPFLAGS=-I../src
AM_CXXFLAGS=-g -Wall -fopenmp
AM_LDFLAGS=-L../src
LDADD=-lddetect -lm $(GFLAGS_LIBS) -lcaffe -lopencv_core -lboost_regex
