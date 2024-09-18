use chess::{BitBoard, Board, ChessMove, Color, MoveGen, Piece};
use candle_core::{Device, DType, Tensor};
use candle_nn::{Conv2d, ConvTranspose2d, Module, VarBuilder, VarMap};
use crate::player::Player;

pub struct ChessNet {
    varmap: VarMap,
    c1: Conv2d,
    t1: ConvTranspose2d,
}

impl  ChessNet {
    pub fn new(varmap: VarMap) -> ChessNet {
        let vs = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        ChessNet {
            varmap,
            c1: candle_nn::conv2d(6, 2, 3, Default::default(), vs.pp("c1")).expect(""),
            t1: candle_nn::conv_transpose2d(2, 2, 3, Default::default(), vs.pp("t1")).expect(""),
        }
    }
    
    pub fn from_file(safe_tensors_file: &str) -> ChessNet {
        let mut varmap = VarMap::new();
        varmap.load(safe_tensors_file).expect("Coulnd't read safetensors file");
        ChessNet::new(varmap)
    }

    pub fn get_weights_and_biases(&self, name: &str) -> (&Tensor, &Tensor) {
        match name {
            "c1" => (self.c1.weight(), self.c1.bias().unwrap()),
            "t1" => (self.t1.weight(), self.t1.bias().unwrap()),
            _ => panic!("Invalid layer name: {}", name)
        }
    }

    pub fn save(&self, file: String) {
        self.varmap.save(file).expect("TODO: panic message");
    }
}

impl  Module for ChessNet {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        xs.apply(&self.c1).expect("c1")
            .relu().expect("relu1")
            .apply(&self.t1).expect("t1")
            .relu()
    }
}

impl  ChessNet {
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

    fn move_to_score(chess_move: &ChessMove, scores: &Tensor) -> f64 {
        let source = chess_move.get_source();
        let dest = chess_move.get_dest();
        let source_score: f64 = scores
            .get(0).unwrap()
            .get(source.get_file().to_index()).unwrap()
            .get(source.get_rank().to_index()).unwrap()
            .to_scalar().unwrap();
        let dest_score: f64 = scores
            .get(1).unwrap()
            .get(dest.get_file().to_index()).unwrap()
            .get(dest.get_rank().to_index()).unwrap()
            .to_scalar().unwrap();
        source_score + dest_score
    }
}

impl  Player for ChessNet {
    fn make_move(&self, board: &Board) -> ChessMove {
        let x = match ChessNet::board_to_tensor(board) {
            Ok(ok) => ok.unsqueeze(0).unwrap(),
            Err(e) => panic!("{:?}", e)
        };
        // find desirable board positions
        let scores = match self.forward(&x) {
            Ok(s) => s.get(0).unwrap(),
            Err(e) => panic!("{:?}", e)
        };
        let moves = MoveGen::new_legal(board);
        // compare legal moves to desirable position to find which one gets closest
        let best_move = match moves.max_by(|m, n| {
            ChessNet::move_to_score(m, &scores)
                .partial_cmp(&ChessNet::move_to_score(n, &scores)).unwrap()
        }) {
            Some(m) => m,
            None => panic!("Didn't find a best move")
        };
        // println!("{}", best_move);
        best_move
    }
}


#[cfg(test)]
mod test {
    use candle_core::{Device, Module, Tensor};
    use candle_nn::{VarMap};
    use chess::{Board, Color, Piece};
    use crate::nn::ChessNet;

    #[test]
    fn dims() {
        let varmap = VarMap::new();
        let model = ChessNet::new(varmap);
        let input = match Tensor::randn(0f64, 1.0, (1, 6, 8, 8), &Device::Cpu) {
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
        println!("{:?}", model.get_weights_and_biases("c1").1.shape())
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
