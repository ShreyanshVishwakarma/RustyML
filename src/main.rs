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

fn main() {}
