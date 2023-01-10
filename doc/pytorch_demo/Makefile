# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

CXX ?= g++
CXXFLAGS = -Wall -Wextra -Werror -std=c++17 -O2 -g -fPIC -DONNX_NAMESPACE=onnx

build/custom_ops.so: custom_ops.cpp
	mkdir -p build && $(CXX) $(CXXFLAGS) -shared $^ -o $@ -Wl,--no-undefined -lpoplar -lpopart -lgcl
