use std::fs;

use tch::{
    nn::{self, Adam, Module, OptimizerConfig},
    vision, TchError, Tensor,
};

const LR: f64 = 0.01;
const D: i64 = 28 * 28;

fn main() {
    match run() {
        Ok(_) => println!("Success!"),
        Err(e) => eprintln!("Error: {}", e),
    }
}

fn network(vs: &nn::Path) -> nn::Sequential {
    let enc_vs = &(vs / "encoder");
    let encoder = nn::seq()
        .add(nn::linear(enc_vs / "l1", D, 256, Default::default()))
        .add_fn(|x| x.sigmoid())
        .add(nn::linear(enc_vs / "l2", 256, 128, Default::default()))
        .add_fn(|x| x.sigmoid());

    let dec_vs = &(vs / "decoder");
    let decoder = nn::seq()
        .add(nn::linear(dec_vs / "l1", 128, 256, Default::default()))
        .add_fn(|x| x.sigmoid())
        .add(nn::linear(dec_vs / "l2", 256, D, Default::default()))
        .add_fn(|x| x.sigmoid());

    return nn::seq()
        .add_fn(|x| x.flatten(1, -1))
        .add(encoder)
        .add(decoder)
        .add_fn(|x| x.unflatten(-1, &[1, 28, 28]));
}

fn run() -> Result<(), TchError> {
    let mut vs = nn::VarStore::new(tch::Device::cuda_if_available());

    // Warn if we don't use CUDA
    if vs.device().is_cuda() {
        println!("Running on CUDA");
    } else {
        eprintln!("Running on CPU");
    }
    // Try loading existing checkpoint file
    if fs::metadata("model.tch").is_ok() {
        vs.load("model.tch")?;
        println!("Loaded existing model from ./model.tch");
    }

    let net = network(&vs.root());
    let m = vision::mnist::load_dir("data")?;
    let mut opt = Adam::default().build(&vs, LR)?;

    let test_data = &m.test_images;
    for epoch in 1..=10 {
        for (ref batch, _) in m.train_iter(64) {
            let out = net.forward(batch);
            let y = 1 - batch.view(&out.size()[..]);
            let loss = out.binary_cross_entropy::<Tensor>(&y, None, tch::Reduction::Mean);
            opt.backward_step(&loss);
        }

        let out = net.forward(test_data);
        let test_label = 1 - test_data.view(&out.size()[..]);
        let loss = out.f_mse_loss(&test_label, tch::Reduction::Mean)?;
        println!("Epoch {:3}: loss = {:.5}", epoch, f64::from(&loss));
    }
    vs.save("model.tch")?;
    Ok(())
}
