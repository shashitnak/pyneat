use pyo3::prelude::*;
use neat_rs::Genotype;
use std::fs::File;
use std::io::prelude::*;

#[pyclass]
struct Network {
    net: slow_nn::Network
}

fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

fn tanh(x: f64) -> f64 {
    x.tanh()
}

#[pymethods]
impl Network {
    fn predict(&self, input: Vec<f64>, activation: &str) -> Vec<f64> {
        if activation == "sigmoid" {
            self.net.predict(&input, sigmoid)
        } else {
            self.net.predict(&input, tanh)
        }
    }

    fn save(&self, path: &str) -> PyResult<&'static str> {
        let mut file = File::create(path)?;
        let buffer: Vec<u8> = bincode::serialize(&self.net).expect("Error saving file");
        file.write(&buffer)?;
        Ok("File written successfully.")
    }

    #[new]
    fn new(path: String) -> PyResult<Self> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let net = bincode::deserialize(&buffer).expect("Error reading file");
        Ok(Self { net })
    }
}

#[pyclass]
struct Neat {
    neat: neat_rs::Neat<Genotype>
}

#[pymethods]
impl Neat {
    #[new]
    fn new(inputs: usize, outputs: usize, size: usize, mutation_rate: f64) -> Self {
        Self {
            neat: neat_rs::Neat::new(inputs, outputs, size, mutation_rate)
        }
    }

    fn eval(&mut self, func: PyObject) {
        self.neat.next_generation(|genome, display| {
            let gil = Python::acquire_gil();
            let py = gil.python();
            let network = Network { net: genome.get_network() };
            let score: f64 = func
                .call1(py, (network, display))
                .expect("Error occured when calling function")
                .extract(py)
                .expect("Could not convert to float");
            score
        });
    }
}

#[pymodule]
fn neat_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Network>()?;
    m.add_class::<Neat>()?;
    Ok(())
}
