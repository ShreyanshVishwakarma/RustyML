use ndarray::prelude::*;
use rand::Rng;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_deri(x: f32) -> f32 {
    x * (1.0 - x)
}

struct XorModel {
    // input -> hidden layer
    w1: Array2<f32>,
    b1: Array1<f32>,

    // hidden -> out
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl XorModel {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        // init with random values
        XorModel {
            w1: Array2::from_shape_fn((2, 2), |_| rng.r#gen::<f32>()),
            b1: Array1::from_shape_fn(2, |_| rng.r#gen::<f32>()),
            w2: Array2::from_shape_fn((1, 2), |_| rng.r#gen::<f32>()),
            b2: Array1::from_shape_fn(1, |_| rng.r#gen::<f32>()),
        }
    }

    fn forward(&self, x: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        let z1 = self.w1.dot(x) + &self.b1; // (2,2) · (2,) = (2,)
        let a1 = z1.mapv(sigmoid);

        let z2 = self.w2.dot(&a1) + &self.b2; // (1,2) · (2,) = (1,)
        let a2 = z2.mapv(sigmoid);

        (a1, a2)
    }

    fn train(&mut self, x: &Array1<f32>, target: &Array1<f32>) {
        let lr: f32 = 0.5;
        let (a1, a2) = self.forward(x);

        let delta2 = (&a2 - target) * a2.mapv(sigmoid_deri);

        let delta1 = self.w2.t().dot(&delta2) * a1.mapv(sigmoid_deri);

        let grad_w2 = delta2
            .view()
            .insert_axis(Axis(1))
            .dot(&a1.insert_axis(Axis(0)));
        let grad_w1 = delta1
            .view()
            .insert_axis(Axis(1))
            .dot(&x.clone().insert_axis(Axis(0)));

        self.w1 -= &(grad_w1 * lr);
        self.b1 -= &(&delta1 * lr);

        self.w2 -= &(grad_w2 * lr);
        self.b2 -= &(&delta2 * lr)
    }
}

fn main() {
    let inputs = vec![
        array![0.0, 0.0],
        array![0.0, 1.0],
        array![1.0, 0.0],
        array![1.0, 1.0],
    ];

    let targets = vec![array![0.0], array![1.0], array![1.0], array![0.0]];

    let epochs = 50000;
    let mut model = XorModel::new();

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        for (x, y) in inputs.iter().zip(targets.iter()) {
            model.train(x, y);

            // loss
            let (_, a2) = model.forward(x);
            let error = &a2 - y;
            epoch_loss += error[0].powi(2);
        }
        if epoch % 1000 == 0 {
            println!("Epoch {}: Loss {:.4}", epoch, epoch_loss / 4.0);
        }
    }
    println!("\nFinal Predictions:");
    for x in &inputs {
        let (_, a2) = model.forward(x);
        println!("{:?} -> {:.4}", x.to_vec(), a2[0]);
    }
}
