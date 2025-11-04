use candle_core::{Result, Tensor};
use candle_nn::{batch_norm, conv2d, conv_transpose2d, linear, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear, Module, ModuleT, VarBuilder};
use std::path::Path;

#[derive(Debug)]
pub struct ResNet {
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

impl ResNet {
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

impl ResNet {
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

impl Module for ResNet {
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

pub struct UNet {
    // A simple U-Net architecture for chess position evaluation.
    c1: Conv2d,
    c2: Conv2d,
    c3: Conv2d,
    u1: candle_nn::ConvTranspose2d,
    u2: candle_nn::ConvTranspose2d,
    conv_out: Conv2d,

    // Value head
    v_conv1: Conv2d,
    v_ln1: Linear,
}

impl UNet {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let c1 = conv2d(6, 32, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("c1"))?;
        let c2 = conv2d(32, 64, 2, Default::default(), vb.pp("c2"))?;
        let c3 = conv2d(64, 128, 2, Default::default(), vb.pp("c3"))?;
        let u1 = conv_transpose2d(128, 64, 2, Default::default(), vb.pp("u1"))?;
        let u2 = conv_transpose2d(128, 32, 2, Default::default(), vb.pp("u2"))?;
        let conv_out = conv2d(64, 2, 1, Default::default(), vb.pp("out"))?;

        // Value head layers
        let v_conv1 = conv2d(128, 1, 1, Default::default(), vb.pp("v_conv1"))?;
        let v_ln1 = linear(6 * 6, 1, vb.pp("v_ln1"))?;

        Ok(Self { c1, c2, c3, u1, u2, conv_out, v_conv1, v_ln1 })
    }
}

impl UNet {
    pub fn forward_all(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let xs = if xs.rank() == 3 {
            xs.unsqueeze(0)?
        } else {
            xs.clone()
        };
        let c1_out = self.c1.forward(&xs)?.relu()?;
        let c2_out = self.c2.forward(&c1_out)?.relu()?;
        let c3_out = self.c3.forward(&c2_out)?.relu()?;

        // Policy head
        let u1_out = self.u1.forward(&c3_out)?.relu()?;
        let u1_out = Tensor::cat(&[&u1_out, &c2_out], 1)?;
        let u2_out = self.u2.forward(&u1_out)?.relu()?;
        let u2_out = Tensor::cat(&[&u2_out, &c1_out], 1)?;
        let policy = self.conv_out.forward(&u2_out)?;

        // Value head
        let value = self.v_conv1.forward(&c3_out)?.relu()?;
        let value = value.flatten_from(1)?;
        let value = self.v_ln1.forward(&value)?.tanh()?;

        Ok((policy, value))
    }
}

impl Module for UNet {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (policy, value) = self.forward_all(xs)?;
        let policy = policy.flatten_from(1)?;
        Tensor::cat(&[&policy, &value], 1)
    }
}