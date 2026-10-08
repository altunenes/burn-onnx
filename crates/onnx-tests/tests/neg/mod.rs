// Import the shared macro
use crate::include_models;
include_models!(neg, neg_shape);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData, Tolerance};

    #[test]
    fn neg() {
        let device = Default::default();
        let model: neg::Model = neg::Model::new(&device);

        let input1 = Tensor::<4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);
        let input2 = 99f64;

        let (output1, output2) = model.forward(input1, input2);
        let expected1 = TensorData::from([[[[-1.0f32, -4.0, -9.0, -25.0]]]]);
        let expected2 = -99f64;

        output1
            .to_data()
            .assert_approx_eq::<f32>(&expected1, Tolerance::default());

        assert_eq!(output2, expected2);
    }

    #[test]
    fn neg_shape() {
        let device = Default::default();
        let model =
            neg_shape::Model::from_file(concat!(env!("OUT_DIR"), "/model/neg_shape.bpk"), &device);

        let input = Tensor::<3>::ones([2, 3, 4], &device);
        let output = model.forward(input);

        assert_eq!(output, [-2, -3, -4]);
    }
}
