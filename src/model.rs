use candle_core::{Result, Tensor};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, Module, VarBuilder};
use std::path::Path;

#[derive(Debug)]
pub struct UNet {
    // A simple ResNet-like architecture is a great starting point.
    // It's simpler than a full U-Net but captures the spirit of skip connections.
    conv_in: Conv2d,
    res_block1: ResBlock,
    res_block2: ResBlock,
    conv_out: Conv2d,

    // Value head
    value_conv: Conv2d,
    value_ln1: Linear,
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
        let conv_out = conv2d(CHANNELS, 64, 1, Default::default(), vb.pp("policy_out"))?;

        // Value head layers
        let value_conv = conv2d(CHANNELS, 1, 1, Default::default(), vb.pp("v_conv1"))?;
        let value_ln1 = linear(8 * 8, 1, vb.pp("v_ln1"))?;

        Ok(Self {
            conv_in,
            res_block1,
            res_block2,
            conv_out,
            value_conv,
            value_ln1,
        })
    }

}

impl Module for UNet {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // This model now returns two heads. We'll separate them later.
        let xs = if xs.rank() == 3 {
            xs.unsqueeze(0)?
        } else {
            xs.clone()
        };
        let xs = self.conv_in.forward(&xs)?.relu()?;
        let res1 = self.res_block1.forward(&xs)?;
        let res2 = self.res_block2.forward(&res1)?;

        // --- Policy Head ---
        let policy_logits = self.conv_out.forward(&res2)?;
        // Apply log_softmax for numerical stability with cross_entropy loss
        let policy_logits = candle_nn::ops::log_softmax(&policy_logits, 1)?;
        let policy_logits = policy_logits.flatten_from(1)?; // Shape: (b_sz, 4096)

        // --- Value Head ---
        let value = self.value_conv.forward(&res2)?.relu()?;
        let value = value.flatten_from(1)?; // Shape: (b_sz, 64)
        let value = self.value_ln1.forward(&value)?.tanh()?; // Shape: (b_sz, 1)

        // We concatenate the two heads for simplicity in training and inference.
        // The first 4096 elements are policy, the last element is value.
        Tensor::cat(&[&policy_logits, &value], 1)
    }
}

pub fn load_model(model_path: &Path, device: &candle_core::Device) -> anyhow::Result<UNet> {
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], candle_core::DType::F32, device)? };
    let model = UNet::new(vb)?;
    Ok(model)
}