use candle_core::{Result, Tensor};
use candle_nn::{batch_norm, conv2d, linear, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear, Module, ModuleT, VarBuilder};
use std::path::Path;

#[derive(Debug)]
pub struct UNet {
    // A simple ResNet-like architecture is a great starting point.
    // It's simpler than a full U-Net but captures the spirit of skip connections.
    conv_in: Conv2d,
    res_block1: ResBlock,
    res_block2: ResBlock,
    res_block3: ResBlock,
    conv_out: Conv2d,

    // Value head
    value_conv: Conv2d,
    value_ln1: Linear,
}

#[derive(Debug)]
struct ResBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
}

impl ResBlock {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let conv1 = conv2d(channels, channels, 3, conv_cfg, vb.pp("c1"))?;
        let bn1 = batch_norm(channels, BatchNormConfig::default(), vb.pp("bn1"))?;
        let conv2 = conv2d(channels, channels, 3, conv_cfg, vb.pp("c2"))?;
        let bn2 = batch_norm(channels, BatchNormConfig::default(), vb.pp("bn2"))?;
        Ok(Self { conv1, bn1, conv2, bn2 })
    }
}

impl ResBlock {
    pub fn forward_is_training(&self, xs: &Tensor, is_training: bool) -> Result<Tensor> {
        let residual = xs;
        let xs = self.conv1.forward(xs)?;
        let xs = self.bn1.forward_t(&xs, is_training)?.relu()?;
        let xs = self.conv2.forward(&xs)?;
        let xs = self.bn2.forward_t(&xs, is_training)?;
        (xs + residual)?.relu()
    }
}

impl UNet {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        const CHANNELS: usize = 64;
        let conv_cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let conv_in = conv2d(6, CHANNELS, 3, conv_cfg, vb.pp("in"))?;
        let res_block1 = ResBlock::new(CHANNELS, vb.pp("res1"))?;
        let res_block2 = ResBlock::new(CHANNELS, vb.pp("res2"))?;
        let res_block3 = ResBlock::new(CHANNELS, vb.pp("res3"))?;
        // The output has 64 channels, one for each "to" square.
        let conv_out = conv2d(CHANNELS, 64, 1, Default::default(), vb.pp("policy_out"))?;

        // Value head layers
        let value_conv = conv2d(CHANNELS, 1, 1, Default::default(), vb.pp("v_conv1"))?;
        let value_ln1 = linear(8 * 8, 1, vb.pp("v_ln1"))?;

        Ok(Self {
            conv_in,
            res_block1,
            res_block2,
            res_block3,
            conv_out,
            value_conv,
            value_ln1,
        })
    }

}

impl UNet {
    pub fn forward_is_training(&self, xs: &Tensor, is_training: bool) -> Result<Tensor> {
        // This model now returns two heads. We'll separate them later.
        let xs = if xs.rank() == 3 {
            xs.unsqueeze(0)?
        } else {
            xs.clone()
        };
        let xs = self.conv_in.forward(&xs)?.relu()?;
        let res1 = self.res_block1.forward_is_training(&xs, is_training)?;
        let res2 = self.res_block2.forward_is_training(&res1, is_training)?;
        let res3 = self.res_block3.forward_is_training(&res2, is_training)?;

        // --- Policy Head ---
        let policy_logits = self.conv_out.forward(&res3)?;
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

impl Module for UNet {
    /// The standard forward pass for inference.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_is_training(xs, false)
    }
}

pub fn load_model(model_path: &Path, device: &candle_core::Device) -> anyhow::Result<UNet> {
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], candle_core::DType::F32, device)? };
    let model = UNet::new(vb)?;
    Ok(model)
}

// --- Move Scoring Model ---

#[derive(Debug)]
pub struct ScoringNet {
    conv_in: Conv2d,
    res_block1: ResBlock,
    res_block2: ResBlock,
    // The value head is the entire model now
    value_conv: Conv2d,
    value_ln1: Linear,
}

impl ScoringNet {
    /// Creates a new ScoringNet. Note the input channels are 8.
    pub fn new(vb: VarBuilder) -> Result<Self> {
        const CHANNELS: usize = 64;
        let conv_cfg = Conv2dConfig { padding: 1, ..Default::default() };
        // Input is 8 channels: 6 for pieces, 1 for 'from' square, 1 for 'to' square
        let conv_in = conv2d(8, CHANNELS, 3, conv_cfg, vb.pp("in"))?;
        let res_block1 = ResBlock::new(CHANNELS, vb.pp("res1"))?;
        let res_block2 = ResBlock::new(CHANNELS, vb.pp("res2"))?;

        // Value head layers
        let value_conv = conv2d(CHANNELS, 1, 1, Default::default(), vb.pp("v_conv1"))?;
        let value_ln1 = linear(8 * 8, 1, vb.pp("v_ln1"))?;

        Ok(Self {
            conv_in,
            res_block1,
            res_block2,
            value_conv,
            value_ln1,
        })
    }

    /// The forward pass for the scoring network.
    pub fn forward_is_training(&self, xs: &Tensor, is_training: bool) -> Result<Tensor> {
        let xs = if xs.rank() == 3 {
            xs.unsqueeze(0)?
        } else {
            xs.clone()
        };

        let xs = self.conv_in.forward(&xs)?.relu()?;
        let res1 = self.res_block1.forward_is_training(&xs, is_training)?;
        let res2 = self.res_block2.forward_is_training(&res1, is_training)?;

        let value = self.value_conv.forward(&res2)?.relu()?;
        let value = value.flatten_from(1)?; // Shape: (b_sz, 64)
        let value = self.value_ln1.forward(&value)?.tanh()?; // Shape: (b_sz, 1)

        Ok(value)
    }
}

impl Module for ScoringNet {
    /// The standard forward pass for inference.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_is_training(xs, false)
    }
}

pub fn load_scoring_model(model_path: &Path, device: &candle_core::Device) -> anyhow::Result<ScoringNet> {
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], candle_core::DType::F32, device)? };
    let model = ScoringNet::new(vb)?;
    Ok(model)
}