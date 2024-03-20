use chess::{Board, ChessMove};
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module, seq, Sequential, VarBuilder};
use crate::player::Player;

struct ChessNet {
    ln1: Linear,
}

impl ChessNet {
    fn new(vs: VarBuilder) -> ChessNet {
        ChessNet {
            ln1: candle_nn::linear(6, 2, vs.pp("ln1")).expect(""),
        }
    }
}

// impl Player for ChessNet {
//     fn make_move(&self, board: &Board) -> ChessMove {
//
//     }
// }


#[cfg(test)]
mod test {
    use candle_core::{Device, DType, Module, Tensor};
    use candle_nn::{Linear, seq, VarBuilder, VarMap};

    #[test]
    fn dims() {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::U8, &Device::Cpu);
        let model = candle_nn::linear(6, 2, vs.pp("ln1")).expect("");
        let input = Tensor::randn(0f32, 1.0, (8, 8, 6), &Device::Cpu).expect("");
        let output = model.forward(&input).expect("");
        println!("{:?}", output.shape())
    }
}
