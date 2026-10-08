#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/abs/abs_shape.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

# ONNX opset version to use for model generation
OPSET_VERSION = 16


def main():
    # Abs on a Shape value: Abs(Shape(x)) and Abs(Shape(x) * -1)
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4])

    shape_node = helper.make_node("Shape", inputs=["input"], outputs=["shape"])
    minus_one = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["minus_one"],
        value=helper.make_tensor(
            name="const_tensor", data_type=TensorProto.INT64, dims=[], vals=[-1]
        ),
    )
    negate_node = helper.make_node(
        "Mul", inputs=["shape", "minus_one"], outputs=["negated"]
    )
    abs_node = helper.make_node("Abs", inputs=["shape"], outputs=["abs_shape"])
    abs_negated_node = helper.make_node(
        "Abs", inputs=["negated"], outputs=["abs_negated"]
    )

    output1 = helper.make_tensor_value_info("abs_shape", TensorProto.INT64, [3])
    output2 = helper.make_tensor_value_info("abs_negated", TensorProto.INT64, [3])

    graph_def = helper.make_graph(
        [shape_node, minus_one, negate_node, abs_node, abs_negated_node],
        "abs_shape_test",
        [input_tensor],
        [output1, output2],
    )

    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-tests",
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )

    onnx_name = "abs_shape.onnx"
    onnx.save(model_def, onnx_name)
    print("Finished exporting model to {}".format(onnx_name))

    test_input = np.ones((2, 3, 4), dtype=np.float32)
    session = ReferenceEvaluator(onnx_name, verbose=0)
    abs_shape, abs_negated = session.run(None, {"input": test_input})

    print(f"Test output abs_shape: {repr(abs_shape)}")
    print(f"Test output abs_negated: {repr(abs_negated)}")

    np.testing.assert_array_equal(abs_shape, [2, 3, 4])
    np.testing.assert_array_equal(abs_negated, [2, 3, 4])


if __name__ == "__main__":
    main()
