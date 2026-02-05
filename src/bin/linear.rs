use ndarray::prelude::*;
struct Model {
    m: f32,
    c: f32,
}

impl Model {
    fn new() -> Self {
        Model {
            m: rand::random::<f32>(),
            c: rand::random::<f32>(),
        }
    }

    fn forward(&self, x: f32) -> f32 {
        self.m * x + self.c
    }
}

fn main() {
    let mut model = Model::new();
    // generating the dataset for y = 2x + 1
    let x = ndarray::Array1::linspace(0.0, 20.0, 20);
    let slope = 2.0;
    let inter = 1.0;
    let y: Array1<f32> = slope * &x + inter;

    let learning_rate: f32 = 0.001;
    let epoch = 1000;

    for i in 0..epoch {
        let mut t_error: f32 = 0.0;
        let mut grad_m: f32 = 0.0;
        let mut grad_c: f32 = 0.0;

        for (x, y) in x.iter().zip(y.iter()) {
            let y_pred = model.forward(*x);

            let error: f32 = y_pred - y;

            t_error += error * error;

            grad_m += 2.0 * error * x;
            grad_c += 2.0 * error;
        }

        let n = x.len() as f32;
        grad_m /= n;
        grad_c /= n;
        t_error /= n;

        // optimization
        model.m -= learning_rate * grad_m;
        model.c -= learning_rate * grad_c;

        if i % 100 == 0 {
            println!(
                "Epoch {}: Loss = {:.4}, m = {:.3}, c = {:.3}",
                i, t_error, model.m, model.c
            );
        }
    }
    println!("---------------------------------");
    println!("Final Result: y = {:.3}x + {:.3}", model.m, model.c);
    println!("Target was:   y = 2.000x + 1.000");
}
