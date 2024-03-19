use rand;
use chess::{Board, ChessMove, MoveGen};
use rand::seq::IteratorRandom;
use crate::player::Player;

pub struct RandomPlayer {
    
}

impl Player for RandomPlayer {
    fn make_move(&self, board: &Board) -> ChessMove {
        let moves = MoveGen::new_legal(board);
        moves.choose(&mut rand::thread_rng()).expect("There should be a legal move")
    }
}


#[cfg(test)]
mod test {
    use chess::{Board, ChessMove, MoveGen};
    use crate::player::Player;
    use crate::players::RandomPlayer;

    #[test]
    fn get_legal_move() {
        let p1 = RandomPlayer {};
        let board = Board::default();
        let moves = MoveGen::new_legal(&board);
        let p1_move = p1.make_move(&board);
        let moves_vec: Vec<ChessMove> = moves.collect();
        assert!(moves_vec.contains(&p1_move));
    }
}
