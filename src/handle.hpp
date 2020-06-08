#ifndef HANDLE_HPP
#define HANDLE_HPP

#define IS_LUX

#ifndef IS_LUX
#include <handlegraph/handle_graph.hpp>
#include <handlegraph/mutable_handle_graph.hpp>
#include <handlegraph/deletable_handle_graph.hpp>
#include <handlegraph/types.hpp>
#else 
#include "../deps/libhandlegraph/src/include/handlegraph/handle_graph.hpp"
#include "../deps/libhandlegraph/src/include/handlegraph/mutable_handle_graph.hpp"
#include "../deps/libhandlegraph/src/include/handlegraph/deletable_handle_graph.hpp"
#include "../deps/libhandlegraph/src/include/handlegraph/types.hpp"
#endif

using handle_t = handlegraph::handle_t;
using nid_t = handlegraph::nid_t;

using HandleGraph = handlegraph::HandleGraph;
using MutableHandleGraph = handlegraph::MutableHandleGraph;
using DeletableHandleGraph = handlegraph::DeletableHandleGraph;

#endif /* HANDLE_HPP */