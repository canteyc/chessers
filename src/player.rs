use chess::{Board, ChessMove};

pub trait Player {
    fn make_move(&self, board: &Board) -> ChessMove;
}