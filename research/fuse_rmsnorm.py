#!/usr/bin/env python3
"""
Fuse RMSNorm subgraph patterns into SimplifiedLayerNormalization ops in ONNX models.

Pattern detected:
  Pow(x, 2) -> ReduceMean -> Add(eps) -> Sqrt -> Reciprocal -> Mul(x, rsqrt) -> Mul(weight, ...)

Replaced with:
  SimplifiedLayerNormalization(x, weight, epsilon=eps)

This reduces node count by ~5 nodes per RMSNorm instance (removing Pow, ReduceMean, Add, Sqrt,
Reciprocal, and one Mul), replacing the final weight-Mul with the fused op.

Usage:
  python fuse_rmsnorm.py input.onnx output.onnx
"""

import argparse
import sys
from collections import defaultdict

import onnx
from onnx import helper, numpy_helper


def build_maps(graph):
    """Build lookup maps for the graph."""
    output_to_node = {}
    for n in graph.node:
        for o in n.output:
            output_to_node[o] = n

    input_to_consumers = defaultdict(list)
    for n in graph.node:
        for inp in n.input:
            input_to_consumers[inp].append(n)

    # Map initializer names to their values
    initializer_map = {}
    for init in graph.initializer:
        initializer_map[init.name] = init

    return output_to_node, input_to_consumers, initializer_map


def find_rmsnorm_patterns(graph):
    """Find all RMSNorm patterns in the graph.

    Returns list of dicts with:
      - 'pow': Pow node
      - 'reduce_mean': ReduceMean node
      - 'add': Add (eps) node
      - 'sqrt': Sqrt node
      - 'reciprocal': Reciprocal node
      - 'mul_norm': Mul(x, rsqrt) node
      - 'mul_weight': Mul(weight, normalized) node
      - 'input_name': name of the input tensor x
      - 'weight_name': name of the weight initializer
      - 'eps': epsilon value
    """
    output_to_node, input_to_consumers, initializer_map = build_maps(graph)

    patterns = []
    used_nodes = set()  # Avoid overlapping patterns

    for node in graph.node:
        if node.op_type != "Pow":
            continue
        if id(node) in used_nodes:
            continue

        # Pow must have 2 inputs: x, exponent (scalar 2)
        if len(node.input) != 2:
            continue
        pow_input = node.input[0]
        pow_exp_name = node.input[1]

        # Check exponent is 2
        if pow_exp_name in initializer_map:
            exp_tensor = numpy_helper.to_array(initializer_map[pow_exp_name])
            if exp_tensor.size != 1 or float(exp_tensor.flat[0]) != 2.0:
                continue
        else:
            continue

        # Pow output -> ReduceMean
        pow_out = node.output[0]
        rm_consumers = [c for c in input_to_consumers[pow_out] if c.op_type == "ReduceMean"]
        if len(rm_consumers) != 1:
            continue
        reduce_mean = rm_consumers[0]

        # ReduceMean output -> Add(eps)
        rm_out = reduce_mean.output[0]
        add_consumers = [c for c in input_to_consumers[rm_out] if c.op_type == "Add"]
        if len(add_consumers) != 1:
            continue
        add_node = add_consumers[0]

        # Extract epsilon from Add
        eps_val = None
        for inp in add_node.input:
            if inp == rm_out:
                continue
            if inp in initializer_map:
                eps_tensor = numpy_helper.to_array(initializer_map[inp])
                if eps_tensor.size == 1:
                    eps_val = float(eps_tensor.flat[0])
        if eps_val is None:
            continue

        # Add -> Sqrt
        add_out = add_node.output[0]
        sqrt_consumers = [c for c in input_to_consumers[add_out] if c.op_type == "Sqrt"]
        if len(sqrt_consumers) != 1:
            continue
        sqrt_node = sqrt_consumers[0]

        # Sqrt -> Reciprocal
        sqrt_out = sqrt_node.output[0]
        recip_consumers = [c for c in input_to_consumers[sqrt_out] if c.op_type == "Reciprocal"]
        if len(recip_consumers) != 1:
            continue
        recip_node = recip_consumers[0]

        # Reciprocal -> Mul(x, rsqrt)
        recip_out = recip_node.output[0]
        mul_consumers = [c for c in input_to_consumers[recip_out] if c.op_type == "Mul"]
        if len(mul_consumers) != 1:
            continue
        mul_norm = mul_consumers[0]

        # Verify Mul uses the same input x as Pow
        other_mul_input = None
        for inp in mul_norm.input:
            if inp == recip_out:
                continue
            other_mul_input = inp
        if other_mul_input != pow_input:
            continue

        # mul_norm output -> Mul(weight, normalized)
        mul_norm_out = mul_norm.output[0]
        weight_consumers = [c for c in input_to_consumers[mul_norm_out] if c.op_type == "Mul"]
        if len(weight_consumers) != 1:
            continue
        mul_weight = weight_consumers[0]

        # Identify weight tensor
        weight_name = None
        for inp in mul_weight.input:
            if inp == mul_norm_out:
                continue
            weight_name = inp
        if weight_name is None:
            continue

        # Verify weight is an initializer (model parameter)
        if weight_name not in initializer_map:
            continue

        pattern = {
            "pow": node,
            "reduce_mean": reduce_mean,
            "add": add_node,
            "sqrt": sqrt_node,
            "reciprocal": recip_node,
            "mul_norm": mul_norm,
            "mul_weight": mul_weight,
            "input_name": pow_input,
            "weight_name": weight_name,
            "eps": eps_val,
        }
        patterns.append(pattern)
        for n in [node, reduce_mean, add_node, sqrt_node, recip_node, mul_norm, mul_weight]:
            used_nodes.add(id(n))

    return patterns


def fuse_patterns(graph, patterns):
    """Replace matched patterns with SimplifiedLayerNormalization nodes."""
    nodes_to_remove = set()
    new_nodes = []

    for i, p in enumerate(patterns):
        # Collect nodes to remove
        for key in ["pow", "reduce_mean", "add", "sqrt", "reciprocal", "mul_norm", "mul_weight"]:
            nodes_to_remove.add(id(p[key]))

        # The fused op replaces the mul_weight node — same output name
        fused_output = p["mul_weight"].output[0]

        fused_node = helper.make_node(
            "SimplifiedLayerNormalization",
            inputs=[p["input_name"], p["weight_name"]],
            outputs=[fused_output],
            name=f"fused_rmsnorm_{i}",
            # ORT contrib op domain
            domain="com.microsoft",
            epsilon=p["eps"],
            axis=-1,
            stash_type=1,  # float32 for intermediate computation
        )
        new_nodes.append(fused_node)

    # Rebuild node list: keep non-removed nodes, insert fused nodes at appropriate positions
    remaining = [n for n in graph.node if id(n) not in nodes_to_remove]

    # Insert fused nodes at the position of the first removed node in each pattern
    # For simplicity, append all fused nodes and let ORT topo-sort
    final_nodes = remaining + new_nodes

    del graph.node[:]
    graph.node.extend(final_nodes)

    return len(patterns)


def main():
    parser = argparse.ArgumentParser(description="Fuse RMSNorm patterns in ONNX models")
    parser.add_argument("input", help="Input ONNX model path")
    parser.add_argument("output", help="Output ONNX model path")
    parser.add_argument("--check", action="store_true", help="Only report patterns found, don't modify")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    # For large models with external data, we need to handle this carefully
    try:
        model = onnx.load(args.input)
    except Exception:
        # Try without external data (for inspection only)
        model = onnx.load(args.input, load_external_data=False)
        if not args.check:
            print("ERROR: Could not load external data. Cannot save fused model.")
            sys.exit(1)

    node_count_before = len(model.graph.node)
    patterns = find_rmsnorm_patterns(model.graph)
    print(f"Found {len(patterns)} RMSNorm patterns (of {node_count_before} total nodes)")

    if args.check:
        for i, p in enumerate(patterns):
            print(f"  [{i}] input={p['input_name']}, weight={p['weight_name']}, eps={p['eps']}")
        return

    if not patterns:
        print("No patterns to fuse.")
        return

    fused = fuse_patterns(model.graph, patterns)
    node_count_after = len(model.graph.node)
    print(f"Fused {fused} patterns: {node_count_before} -> {node_count_after} nodes")

    # Add com.microsoft opset import if not present
    has_ms_domain = any(o.domain == "com.microsoft" for o in model.opset_import)
    if not has_ms_domain:
        model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    print(f"Saving to {args.output}...")
    onnx.save(model, args.output)

    import os

    orig_size = os.path.getsize(args.input)
    new_size = os.path.getsize(args.output)
    print(f"Size: {orig_size / 1e6:.1f} MB -> {new_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
