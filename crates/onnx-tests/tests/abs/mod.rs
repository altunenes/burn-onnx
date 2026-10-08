// Import the shared macro
use crate::include_models;
include_models!(abs, abs_shape);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData, Tolerance};

    #[test]
    fn abs() {
        let device = Default::default();
        let model: abs::Model = abs::Model::new(&device);

        let input = Tensor::<4>::from_floats([[[[-1.0, -4.0, 9.0, -25.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[1.0f32, 4.0, 9.0, 25.0]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn abs_shape() {
        let device = Default::default();
        let model =
            abs_shape::Model::from_file(concat!(env!("OUT_DIR"), "/model/abs_shape.bpk"), &device);

        let input = Tensor::<3>::ones([2, 3, 4], &device);
        let (abs_shape, abs_negated) = model.forward(input);

        assert_eq!(abs_shape, [2, 3, 4]);
        assert_eq!(abs_negated, [2, 3, 4]);
    }
}
