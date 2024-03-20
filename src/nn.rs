use chess::{BitBoard, Board, ChessMove, Color, Piece};
use candle_core::{Device, Tensor};
use candle_nn::{Conv2d, ConvTranspose2d, Module, VarBuilder};
use crate::player::Player;

struct ChessNet {
    c1: Conv2d,
    t1: ConvTranspose2d,
}

impl ChessNet {
    fn new(vs: VarBuilder) -> ChessNet {
        ChessNet {
            c1: candle_nn::conv2d(6, 2, 3, Default::default(), vs.pp("c1")).expect(""),
            t1: candle_nn::conv_transpose2d(2, 2, 3, Default::default(), vs.pp("t1")).expect(""),
        }
    }
}

impl Module for ChessNet {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        xs.apply(&self.c1).expect("c1")
            .relu().expect("relu1")
            .apply(&self.t1).expect("t1")
            .relu()
    }
}

impl ChessNet {
    fn bitboard_to_array(bitboard: &BitBoard, white: &BitBoard) -> [[f64; 8]; 8] {
        let mut array = [
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
        ];

        for x in 0..64 {
            if bitboard.0 & (1u64 << x) == (1u64 << x) {
                let i =  x / 8;
                let j = x % 8;
                let value = if white.0 & (1u64 << x) == (1u64 << x) { 1. } else { -1. };
                array[i][j] = value;
            }
        };
        array
    }
    fn board_to_tensor(board: &Board) -> candle_core::Result<Tensor> {
        let pawns = board.pieces(Piece::Pawn);
        let rooks = board.pieces(Piece::Rook);
        let knights = board.pieces(Piece::Knight);
        let bishops = board.pieces(Piece::Bishop);
        let kings = board.pieces(Piece::King);
        let queens = board.pieces(Piece::Queen);
        let white = board.color_combined(Color::White);
        let input_array = [
            ChessNet::bitboard_to_array(pawns, white),
            ChessNet::bitboard_to_array(rooks, white),
            ChessNet::bitboard_to_array(knights, white),
            ChessNet::bitboard_to_array(bishops, white),
            ChessNet::bitboard_to_array(kings, white),
            ChessNet::bitboard_to_array(queens, white),
        ];
        Tensor::new(&input_array, &Device::Cpu)
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
    use candle_nn::{Conv2d, Conv2dConfig, Linear, seq, VarBuilder, VarMap};
    use chess::{Board, Color, Piece};
    use crate::nn::ChessNet;

    #[test]
    fn dims() {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let model = ChessNet::new(vs);
        let input = match Tensor::randn(0f32, 1.0, (1, 6, 8, 8), &Device::Cpu) {
            Ok(input) => input,
            Err(e) => {
                panic!("Failed to create input tensor: \n{:?}\n", e);
            }
        };
        let output = match model.forward(&input) {
            Ok(output) => output,
            Err(e) => {
                panic!("Issue with forward: \n{:?}\n", e);
            }
        };
        println!("{:?}", output.shape());
    }

    #[test]
    fn pawns_array() {
        let board = Board::default();
        let pawns = ChessNet::bitboard_to_array(board.pieces(Piece::Pawn), board.color_combined(Color::White));
        let expected = [
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [-1., -1., -1., -1., -1., -1., -1., -1.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
        ];
        assert_eq!(pawns, expected);
    }
}
