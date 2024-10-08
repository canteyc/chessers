use std::fs::{create_dir_all, File};
use std::env;
use candle_core::{Device, DType, Tensor, NdArray};
use candle_nn::{VarMap};
use chess::{Game, GameResult};
use chrono::Datelike;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use crate::nn::ChessNet;
use crate::player::Player;

impl  ChessNet {
    fn merge(&self, other: &ChessNet, scores: [u64; 2]) -> ChessNet {
        let mutation_threshold = u32::MAX / 10;
        let mut varmap = VarMap::new();
        for layer in ["c1", "t1"] {
            let my_vars = self.get_weights_and_biases(layer);
            let other_vars = other.get_weights_and_biases(layer);
            let dist = WeightedIndex::new(scores).unwrap();

            let w_dim = my_vars.0.dims4().unwrap();
            let mut new_weight: Vec<Vec<Vec<Vec<f64>>>> = vec![vec![vec![vec![0f64; w_dim.3]; w_dim.2]; w_dim.1]; w_dim.0];
            for i in 0..w_dim.0 {
                for j in 0..w_dim.1 {
                    for k in 0..w_dim.2 {
                        for l in 0..w_dim.3 {
                            let inherit_weight: f64 = match thread_rng().sample(&dist) {
                                1 => my_vars.0
                                    .get(i).expect("i")
                                    .get(j).expect("j")
                                    .get(k).expect("k")
                                    .get(l).expect("l")
                                    .to_scalar().expect("scalar"),
                                0 => other_vars.0
                                    .get(i).expect("i")
                                    .get(j).expect("j")
                                    .get(k).expect("k")
                                    .get(l).expect("l")
                                    .to_scalar().expect("scalar"),
                                _ => panic!("Got something else")
                            };
                            let mutation = match thread_rng().next_u32() {
                                v if v < mutation_threshold => 0.5,
                                v if v > u32::MAX - mutation_threshold => 2.,
                                _ => 1.,
                            };
                            new_weight[i][j][k][l] = inherit_weight * mutation;
                        }
                    }
                }
            }
            varmap.get(new_weight.shape().unwrap(), &*(layer.to_string() + ".weight"), Default::default(), DType::F64, &Device::Cpu).expect("TODO: panic message");
            varmap.set_one(layer.to_string() + ".weight", Tensor::new(new_weight, &Device::Cpu).unwrap()).expect("Error setting weight");

            let b_dim = my_vars.1.dims1().unwrap();
            let mut new_bias: Vec<f64> = vec![0f64; 2];
            for i in 0..b_dim {
                new_bias[i] = match thread_rng().sample(&dist) {
                    1 => my_vars.1.get(i).unwrap().to_scalar().unwrap(),
                    0 => other_vars.1.get(i).unwrap().to_scalar().unwrap(),
                    _ => panic!("Got something else")
                };
            }
            varmap.get(new_bias.shape().unwrap(), &*(layer.to_string() + ".bias"), Default::default(), DType::F64, &Device::Cpu).expect("TODO: panic message");
            varmap.set_one(layer.to_string() + ".bias", Tensor::new(new_bias, &Device::Cpu).unwrap()).expect("Error setting bias");
        }

        ChessNet::new(varmap)
    }
}

pub struct Arena {
    num_members: usize,
    num_epochs: i32,
    members: Vec<ChessNet>,
    log_dir: String,
}

impl Arena {
    pub fn new(num_members: usize, num_epochs: i32) -> Arena {
        let mut members: Vec<ChessNet> = vec!();
        for _ in 0..num_members {
            let varmap = VarMap::new();
            members.push(ChessNet::new(varmap));
        }

        let date = chrono::Utc::now();
        let resources = env::var("HOME").expect("No HOME dir?") + "/.chessers";
        let log_dir = format!("{}/logs/{}_{:02}_{:02}",
                              resources, date.year(), date.month(), date.day());
        create_dir_all(&log_dir).expect("Error creating log directory");

        Arena {
            num_members,
            num_epochs,
            members,
            log_dir,
        }
    }

    fn play_game(white: &ChessNet, black: &ChessNet) -> GameResult {
        let mut game = Game::new();
        for _ in 1..1000 {
            game.make_move(white.make_move(&game.current_position()));
            if check_game(&mut game) { break; };
            game.make_move(black.make_move(&game.current_position()));
            if check_game(&mut game) { break; };
        }
        game.result().unwrap()
    }

    pub(crate) fn train(&mut self) {
        // create champs.csv for logging champion index of each epoch
        let champ_file = format!("{}/champs.csv", &self.log_dir);
        let mut champ_writer = csv::Writer::from_path(&champ_file)
            .expect(format!("Failed to open {} for writing", &champ_file).as_str());
        champ_writer.write_record(&["Epoch", "Champ index", "Wins"])
            .expect("TODO: panic message");
        
        
        for epoch in 0..self.num_epochs {
            let mut scores = vec![vec![1u64; self.num_members]; self.num_members];
            for (i, member_white) in (&self.members).iter().enumerate() {
                for (j, member_black) in (&self.members).iter().enumerate() {
                    let result = Arena::play_game(&member_white, &member_black);
                    match result {
                        GameResult::WhiteCheckmates => scores[i][j] += 1,
                        GameResult::BlackCheckmates => scores[j][i] += 1,
                        _ => (),
                    };
                    // println!("Game {}, {}: {:?}", i, j, result);
                }
            }
            let totals = scores.iter().map(|row| row.iter().sum::<u64>());
            let dist: WeightedIndex<u64> = WeightedIndex::new(totals.clone()).unwrap();
            let mut new_members: Vec<ChessNet> = vec!();
            for (i, member) in (&self.members).iter().enumerate() {
                let j = dist.sample(&mut thread_rng());
                let partner = &self.members[j];
                new_members.push(member.merge(partner, [scores[i][j], scores[j][i]]));
            };
            self.members = new_members;
            println!("Finished epoch {}", epoch);
            println!("Scores: {:?}", scores);
            println!("Wins: {:?}", scores.iter().flatten().filter(|&&v| v == 2).count());
            for (i, member) in self.members.iter().enumerate() {
                member.save(format!("{}/{:04}_{:04}.safetensors", &self.log_dir, epoch, i));
            }
            let champ_id = totals.enumerate().max_by(|(_, value0), (_, value1)| value0.cmp(value1)).unwrap().0;
            let champ = &self.members[champ_id];
            let wins = self.evaluate(champ, champ_file.as_str());
            self.log_champ(&mut champ_writer, epoch, champ_id, wins);
        }
    }

    fn evaluate(&self, champion: &ChessNet, champs_file: &str) -> i32 {
        // Compare champion to the best from all previous epochs
        let mut reader = csv::Reader::from_path(champs_file).unwrap();
        {
            reader.headers().unwrap();
        }
        let mut wins = 0;
        for result in reader.records() {
            let row = result.unwrap();
            let epoch: i32 = row.get(0).unwrap().parse().unwrap();
            let index: i32 = row.get(1).unwrap().parse().unwrap();
            let path = format!("{}/{:04}_{:04}.safetensors", &self.log_dir, epoch, index);
            let opponent = ChessNet::from_file(path.as_str());
            let result = Arena::play_game(&champion, &opponent);
            match result {
                GameResult::WhiteCheckmates => wins += 1,
                GameResult::BlackCheckmates => (),
                _ => (),
            };
            let result = Arena::play_game(&opponent, &champion);
            match result {
                GameResult::WhiteCheckmates => (),
                GameResult::BlackCheckmates => wins += 1,
                _ => (),
            };
        }
        wins
    }
    
    fn log_champ(&self, writer: &mut csv::Writer<File>, epoch: i32, index: usize, wins: i32) {
        writer.write_record(&[epoch.to_string(), index.to_string(), wins.to_string()])
            .expect("TODO: panic message");
        writer.flush().unwrap();
    }
}

pub fn check_game(game: &mut Game) -> bool{
    if game.can_declare_draw() {
        game.declare_draw();
    }
    match game.result() {
        Some(r) => {
            println!("{:?}", r);
            true
        },
        None => false,
    }
}


#[cfg(test)]
mod test {
    use candle_core::{Device, DType, Tensor};
    use rand::distributions::WeightedIndex;
    use rand::Rng;
    use rand::prelude::*;
    use crate::arena::Arena;

    #[test]
    fn run() {
        let mut arena = Arena::new(2, 2);
        arena.train();
    }

    #[test]
    fn masking() {
        let a = Tensor::new(vec![3f32; 2], &Device::Cpu);
        let t = Tensor::ones((6, 2), DType::F64, &Device::Cpu).unwrap();
        let dist = WeightedIndex::new([2, 1]).unwrap();
        let mut indices: Vec<Vec<f64>> = t.zeros_like().unwrap().to_vec2().unwrap();
        for i in 0..t.dim(0).unwrap() {
            for j in 0..t.dim(1).unwrap() {
                indices[i][j] = thread_rng().sample(&dist) as f64;
            }
        }
        let indices = Tensor::new(indices, &Device::Cpu).unwrap();
        let indices = t.rand_like(0., 1.).unwrap();
        println!("{:?}", indices.to_string());
    }
}
