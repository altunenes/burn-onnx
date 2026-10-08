use super::prelude::*;

impl NodeCodegen for onnx_ir::node::abs::AbsNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = self.inputs.first().unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        let input = scope.arg(input_arg);

        // Shape values are `[i64; N]` on the host; saturate like the other shape arithmetic.
        let abs_expr = match &input_arg.ty {
            ArgType::Shape(_) => quote! { #input.map(i64::saturating_abs) },
            _ => quote! { #input.abs() },
        };

        quote! {
            let #output = #abs_expr;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::abs::AbsNodeBuilder;

    #[test]
    fn test_abs_forward() {
        let node = AbsNodeBuilder::new("abs1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            let output = input.abs();
            output
        }
        ");
    }

    #[test]
    fn test_abs_forward_shape() {
        let node = AbsNodeBuilder::new("abs1")
            .input_shape("input", 3)
            .output_shape("output", 3)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: [i64; 3]) -> [i64; 3] {
            let output = input.map(i64::saturating_abs);
            output
        }
        ");
    }
}
