use chess::{Board, ChessMove, MoveGen};
use rand::prelude::IteratorRandom;

pub trait Player {
    fn make_move(&self, board: &Board) -> ChessMove;
}

pub struct RandomPlayer {

}

impl Player for RandomPlayer {
    fn make_move(&self, board: &Board) -> ChessMove {
        let moves = MoveGen::new_legal(board);
        let random_move = moves.choose(&mut rand::thread_rng()).expect("There should be a legal move");
        println!("{}", random_move);
        random_move
    }
}

pub struct HumanPlayer {

}

impl HumanPlayer {
    fn get_input(&self, board: &Board) -> Option<ChessMove> {
        let mut user_input = String::new();
        std::io::stdin()
            .read_line(&mut user_input)
            .expect("Failed to read move");
        if user_input.contains("list") {
            println!("Here are all the moves you can do:");
            for legal_move in MoveGen::new_legal(board) {
                println!("{}", legal_move);
            };
            None
        }
        else {
            match ChessMove::from_san(board, &*user_input) {
                Ok(user_move) => Some(user_move),
                _ => {
                    println!("Invalid move");
                    None
                }
            }
        }
    }
}

impl Player for HumanPlayer {
    fn make_move(&self, board: &Board) -> ChessMove {
        let mut user_move = self.get_input(board);
        while user_move.is_none() {
            user_move = self.get_input(board);
        }
        user_move.expect("How did we get out of the while loop?")
    }
}

#[cfg(test)]
mod test {
    use chess::{Board, ChessMove, MoveGen};
    use crate::player::Player;
    use crate::player::RandomPlayer;

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
