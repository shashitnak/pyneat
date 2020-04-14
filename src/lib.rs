use pyo3::prelude::*;
use neat_rs::Genotype;

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
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let net = slow_nn::Network::load(path)?;
        Ok(Self { net })
    }

    fn predict(&self, input: Vec<f64>, activation: &str) -> Vec<f64> {
        if activation == "sigmoid" {
            self.net.predict(&input, sigmoid)
        } else {
            self.net.predict(&input, tanh)
        }
    }

    fn save(&self, path: &str) -> PyResult<&'static str> {
        self.net.save(path)?;
        Ok("File written successfully.")
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.net.to_bytes().unwrap()
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
        let (scores, total_score) = self.neat.calculate_fitness(|genome, display| {
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
        self.neat.next_generation(&scores, total_score);
    }
}

#[pymodule]
fn neat_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Network>()?;
    m.add_class::<Neat>()?;
    Ok(())
}
