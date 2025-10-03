use candle_core::{Result, Tensor, Var};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, VarBuilder};
use std::path::Path;

#[derive(Debug)]
pub struct UNet {
    // A simple ResNet-like architecture is a great starting point.
    // It's simpler than a full U-Net but captures the spirit of skip connections.
    conv_in: Conv2d,
    res_block1: ResBlock,
    res_block2: ResBlock,
    conv_out: Conv2d,
}

#[derive(Debug)]
struct ResBlock {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl ResBlock {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let conv1 = conv2d(channels, channels, 3, conv_cfg, vb.pp("c1"))?;
        let conv2 = conv2d(channels, channels, 3, conv_cfg, vb.pp("c2"))?;
        Ok(Self { conv1, conv2 })
    }
}

impl Module for ResBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.conv1.forward(xs)?.relu()?;
        let xs = self.conv2.forward(&xs)?;
        (xs + residual)?.relu()
    }
}

impl UNet {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        const CHANNELS: usize = 64;
        let conv_cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let conv_in = conv2d(13, CHANNELS, 3, conv_cfg, vb.pp("in"))?;
        let res_block1 = ResBlock::new(CHANNELS, vb.pp("res1"))?;
        let res_block2 = ResBlock::new(CHANNELS, vb.pp("res2"))?;
        // The output has 64 channels, one for each "to" square.
        let conv_out = conv2d(CHANNELS, 64, 1, Default::default(), vb.pp("out"))?;
        Ok(Self {
            conv_in,
            res_block1,
            res_block2,
            conv_out,
        })
    }

}

impl Module for UNet {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Handle both single (3D) and batched (4D) inputs.
        let xs = if xs.rank() == 3 {
            xs.unsqueeze(0)?
        } else {
            xs.clone()
        };
        let b_sz = xs.dim(0)?;
        let xs = self.conv_in.forward(&xs)?.relu()?;
        let xs = self.res_block1.forward(&xs)?;
        let xs = self.res_block2.forward(&xs)?;
        let xs = self.conv_out.forward(&xs)?;
        // The output shape is (b_sz, 64, 8, 8). This represents, for each of the 64 "from" squares (as an 8x8 grid),
        // a value for each of the 64 "to" squares.
        // We need to reshape this to match our `move_to_index` logic, which is `from * 64 + to`.
        let xs = xs.reshape((b_sz, 64, 64))?; // (b_sz, from_square, to_square)
        xs.flatten_from(1) // (b_sz, 4096)
    }
}

pub fn load_model(model_path: &Path, device: &candle_core::Device) -> anyhow::Result<UNet> {
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], candle_core::DType::F32, device)? };
    let model = UNet::new(vb)?;
    Ok(model)
}