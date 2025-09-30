use candle_core::{Result, Tensor};
use candle_nn::{linear, seq, Module, VarBuilder};

pub struct Mlp {
    net: candle_nn::Sequential,
}

impl Mlp {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        // Input: 13x8x8 = 832
        // Hidden layer: 512 neurons
        // Output: 64*64 = 4096 possible moves
        let net = seq()
            .add(linear(832, 512, vb.pp("ln1"))?)
            .add_fn(|xs| xs.relu())
            .add(linear(512, 4096, vb.pp("ln2"))?);
        Ok(Self { net })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Reshape the input from [832] to [1, 832] to add a batch dimension.
        let xs = xs.unsqueeze(0)?;
        // Get the output, which will have shape [1, 4096].
        let logits = self.net.forward(&xs)?;
        // Squeeze the batch dimension out to return a tensor of shape [4096].
        logits.squeeze(0)
    }
}