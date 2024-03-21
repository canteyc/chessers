use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use chess::{Game, GameResult};
use rand::distributions::WeightedIndex;
use rand::prelude::IteratorRandom;
use rand::Rng;
use crate::nn::ChessNet;
use crate::player::Player;

impl ChessNet {
    fn merge(&self, other: &ChessNet) -> ChessNet {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        ChessNet::new(vs)
    }
}

pub struct Arena {
    members: Vec<ChessNet>,
}

impl Arena {
    const NUM_MEMBERS: i32 = 64;
    const NUM_EPOCHS: i32 = 10;

    fn new() -> Arena {
        let mut members: Vec<ChessNet> = vec!();
        let varmap = VarMap::new();
        for _ in 0..Arena::NUM_MEMBERS {
            let vs = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
            members.push(ChessNet::new(vs));
        }
        Arena {
            members
        }
    }

    fn play_game(white: &ChessNet, black: &ChessNet) -> GameResult {
        let mut game = Game::new();
        for i in 1..100 {
            game.make_move(white.make_move(&game.current_position()));
            if check_game(&mut game, i) { break; };
            game.make_move(black.make_move(&game.current_position()));
            if check_game(&mut game, i) { break; };
        }
        game.result().unwrap()
    }

    fn train(&mut self) {
        for epoch in 0..Arena::NUM_EPOCHS {
            let mut scores = Tensor::zeros((8, 8), DType::U8, &Device::Cpu).unwrap().to_vec2().unwrap();
            for (i, member_white) in self.members.iter().enumerate() {
                for (j, member_black) in self.members.iter().enumerate() {
                    match Arena::play_game(&member_white, &member_black) {
                        GameResult::WhiteCheckmates => scores[i][j] = 1,
                        GameResult::BlackCheckmates => scores[j][i] = 1,
                        _ => (),
                    }
                }
            }
            let totals = scores.iter().sum();
            let dist: WeightedIndex<i32> = WeightedIndex::new(totals).unwrap();
            let mut new_members: Vec<ChessNet> = vec!();
            for member in self.members {
                let partner = self.members[dist.sample(&mut rand::thread_rng())];
                new_members.push(member.merge(partner));
            };
            self.members = new_members;
        }
    }
}

pub fn check_game(game: &mut Game, i: i32) -> bool{
    if game.can_declare_draw() {
        game.declare_draw();
    }
    match game.result() {
        Some(_) => {
            println!("finished after {}", i);
            if game.result().is_some() {
                println!("{:?}", game.result().expect(""))
            }
            true
        },
        None => false,
    }
}
