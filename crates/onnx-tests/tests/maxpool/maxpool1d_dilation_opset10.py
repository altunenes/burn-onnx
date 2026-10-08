#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: maxpool1d_dilation_opset10.onnx
#
# MaxPool with dilations=2 at opset 10, the version that introduced dilations. Guards against
# rejecting dilations below opset 11.
#
# kernel 2 / dilation 2 / stride 1 over length 6: each window reads x[i] and x[i + 2], so there
# are 4 outputs.

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 6])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4])

    max_pool = helper.make_node(
        "MaxPool",
        inputs=["x"],
        outputs=["y"],
        kernel_shape=[2],
        strides=[1],
        dilations=[2],
    )

    graph = helper.make_graph([max_pool], "maxpool1d_dilation_opset10", [x], [y])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    model.ir_version = 5

    onnx.checker.check_model(model)
    file_name = "maxpool1d_dilation_opset10.onnx"
    onnx.save(model, file_name)
    print("Finished exporting model to {}".format(file_name))

    session = ReferenceEvaluator(file_name)
    test_input = np.array([[[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]]], dtype=np.float32)
    output = session.run(None, {"x": test_input})[0]
    print("input {} -> output {}".format(test_input.shape, output.shape))
    print(repr(output))


if __name__ == "__main__":
    main()
