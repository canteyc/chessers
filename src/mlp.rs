use candle_core::{Result, Tensor, Var};
use candle_nn::{linear, Activation, Linear, Module, VarBuilder};
use std::path::Path;

#[derive(Debug)]
pub struct Mlp {
    ln1: Linear,
    ln2: Linear,
}

impl Mlp {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        // Input: 13x8x8 = 832
        // Hidden layer: 512 neurons
        // Output: 64*64 = 4096 possible moves
        let ln1 = linear(832, 512, vb.pp("ln1"))?;
        let ln2 = linear(512, 4096, vb.pp("ln2"))?;
        Ok(Self { ln1, ln2 })
    }

    pub fn vars(&self) -> Vec<Var> {
        // Collect all trainable variables from the layers.
        let mut vars = vec![];
        vars.extend(Var::from_tensor(self.ln1.weight()));
        if let Some(bias) = self.ln1.bias() {
            vars.extend(Var::from_tensor(bias));
        }
        vars.extend(Var::from_tensor(self.ln2.weight()));
        if let Some(bias) = self.ln2.bias() {
            vars.extend(Var::from_tensor(bias));
        }
        vars
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Reshape the input from [832] to [1, 832] to add a batch dimension.
        let xs = xs.unsqueeze(0)?;
        // Manually apply the layers and activation function.
        let xs = self.ln1.forward(&xs)?;
        let xs = xs.apply(&Activation::Relu)?;
        let xs = self.ln2.forward(&xs)?;
        // Squeeze the batch dimension out to return a tensor of shape [4096].
        xs.squeeze(0)
    }
}

pub fn load_model(model_path: &Path, device: &candle_core::Device) -> anyhow::Result<Mlp> {
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], candle_core::DType::F32, device)? };
    let model = Mlp::new(vb)?;
    Ok(model)
}