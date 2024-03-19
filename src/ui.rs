use chess::Board;

pub trait UI {
    fn update(&self, board: &Board);
}

pub struct ConsoleUI {

}

impl UI for ConsoleUI {
    fn update(&self, board: &Board) {
        let fen = board.to_string();
        let mut printed: u32 = 0;
        for row in fen.split("/") {
            let mut output = "".to_owned();
            for c in row.chars() {
                if printed == 64 {
                    break;
                }
                if c.is_ascii_digit() {
                    let num = c.to_digit(10).expect("All digits are less than 8");
                    output.push_str(&". ".repeat(num as usize));
                    printed += num;
                }
                else {
                    output.push(c);
                    output.push(' ');
                    printed += 1;
                }
            }
            println!("{}", output);
        }
        println!();
        println!();
    }
}


#[cfg(test)]
mod test {
    use chess::Board;
    use crate::ui::{ConsoleUI, UI};

    #[test]
    fn print_default() {
        let ui = ConsoleUI {};
        ui.update(&Board::default());
    }
}