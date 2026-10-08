#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/neg/neg_shape.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

# ONNX opset version to use for model generation
OPSET_VERSION = 16


def main():
    # Neg on a Shape value: Neg(Shape(x))
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4])

    shape_node = helper.make_node("Shape", inputs=["input"], outputs=["shape"])
    neg_node = helper.make_node("Neg", inputs=["shape"], outputs=["neg_shape"])

    output = helper.make_tensor_value_info("neg_shape", TensorProto.INT64, [3])

    graph_def = helper.make_graph(
        [shape_node, neg_node],
        "neg_shape_test",
        [input_tensor],
        [output],
    )

    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-tests",
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )

    onnx_name = "neg_shape.onnx"
    onnx.save(model_def, onnx_name)
    print("Finished exporting model to {}".format(onnx_name))

    test_input = np.ones((2, 3, 4), dtype=np.float32)
    session = ReferenceEvaluator(onnx_name, verbose=0)
    (neg_shape,) = session.run(None, {"input": test_input})

    print(f"Test output neg_shape: {repr(neg_shape)}")

    np.testing.assert_array_equal(neg_shape, [-2, -3, -4])


if __name__ == "__main__":
    main()
