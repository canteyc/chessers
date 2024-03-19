use chess::Board;

pub trait UI {
    fn update(&self, board: &Board);
}

pub struct ConsoleUI {

}

impl UI for ConsoleUI {
    fn update(&self, board: &Board) {
        // white
        let pawns = board.p
    }
}