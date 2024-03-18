
#[cfg(test)]
mod test {
    use chess::{Board, ChessMove, Square};

    #[test]
    fn what_is_a_bitboard() {
        let board = Board::default();
        // let bit_board = board.combined();
        println!("{}", board);

        let board = board.make_move_new(ChessMove::new(Square::D2, Square::D4, None));
        println!("{}", board);
    }
}