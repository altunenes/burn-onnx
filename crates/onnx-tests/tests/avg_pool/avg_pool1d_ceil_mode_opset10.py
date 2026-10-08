#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: avg_pool1d_ceil_mode_opset10.onnx
#
# AveragePool with ceil_mode=1 at opset 10, the version that introduced ceil_mode. Guards against
# rejecting ceil_mode below opset 19 (dilations is the attribute that arrived in 19).
#
# kernel 3 / stride 2 over length 6: floor mode gives 2 outputs, ceil mode gives 3, the last one a
# partial window averaged over its 2 real elements (count_include_pad defaults to 0).

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 6])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3])

    avg_pool = helper.make_node(
        "AveragePool",
        inputs=["x"],
        outputs=["y"],
        kernel_shape=[3],
        strides=[2],
        ceil_mode=1,
    )

    graph = helper.make_graph([avg_pool], "avg_pool1d_ceil_mode_opset10", [x], [y])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    model.ir_version = 5

    onnx.checker.check_model(model)
    file_name = "avg_pool1d_ceil_mode_opset10.onnx"
    onnx.save(model, file_name)
    print("Finished exporting model to {}".format(file_name))

    session = ReferenceEvaluator(file_name)
    test_input = np.arange(6, dtype=np.float32).reshape(1, 1, 6)
    output = session.run(None, {"x": test_input})[0]
    print("input {} -> output {}".format(test_input.shape, output.shape))
    print(repr(output))


if __name__ == "__main__":
    main()
