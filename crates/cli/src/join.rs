//! `peerinfer join`: runs a range of a team's layers on this machine's GPU.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Args;
use iroh_blobs::Hash;
use peerinfer_core::{Config, Engine, Input, Output, Sampling, Weights, model::WeightIndex};
use peerinfer_net::{
    SecretKey, bind_endpoint,
    team::{FileEntry, Invitation, StageMessage, StageReply, TeamMember},
};

#[derive(Debug, Args)]
pub struct JoinArgs {
    /// The invitation that `peerinfer worker --devices N` printed.
    invitation: String,
    /// Directory for received model files. Files already there are reused.
    #[arg(long)]
    cache: Option<PathBuf>,
}

pub async fn run(args: JoinArgs) -> Result<()> {
    let invitation = Invitation::decode(&args.invitation)?;
    let cache = args.cache.unwrap_or_else(default_cache);
    std::fs::create_dir_all(&cache).with_context(|| format!("create {}", cache.display()))?;

    let endpoint = bind_endpoint(SecretKey::generate(), vec![]).await?;
    println!("joining team at {}", invitation.host.id.fmt_short());
    let member = TeamMember::join(endpoint.clone(), &invitation).await?;
    let a = member.assignment().clone();
    let m = &a.manifest;
    println!(
        "model {} · layers {}–{} of {}",
        m.model,
        a.start,
        a.end - 1,
        m.layers
    );

    let file = |name: &str| -> Result<&FileEntry> {
        m.files
            .iter()
            .find(|f| f.name == name)
            .with_context(|| format!("the host offers no {name}"))
    };
    let index_name = if m.files.iter().any(|f| f.name == "tensor-cache.json") {
        "tensor-cache.json"
    } else {
        "ndarray-cache.json"
    };
    let chat_config = fetch(&member, file("mlc-chat-config.json")?, &cache).await?;
    let index_bytes = fetch(&member, file(index_name)?, &cache).await?;
    let config = Config::from_mlc_chat_config(std::str::from_utf8(&chat_config)?)?;
    let index = WeightIndex::from_ndarray_cache(std::str::from_utf8(&index_bytes)?)?;
    let layers = a.start as usize..a.end as usize;
    let need = index.shards_for(&layers, config.num_hidden_layers);
    let mut shards = Vec::with_capacity(index.shards.len());
    for (i, name) in index.shards.clone().iter().enumerate() {
        if need.contains(&i) {
            shards.push(fetch(&member, file(name)?, &cache).await?);
        } else {
            shards.push(Vec::new());
        }
    }
    let weights = Weights {
        config,
        index,
        shards,
    };
    let mut engine = Engine::new_stage(&weights, layers).await?;
    drop(weights);
    let info = engine.adapter_info();
    println!("gpu {} ({:?})", info.name, info.backend);
    member.ready().await?;
    println!("ready");

    loop {
        let command = tokio::select! {
            _ = tokio::signal::ctrl_c() => break,
            c = member.next() => c,
        };
        let Some(command) = command else {
            println!("the host closed the team");
            endpoint.close().await;
            return Ok(());
        };
        let reply = match &command.message {
            StageMessage::Begin {
                temperature, seed, ..
            } => {
                engine.reset();
                let sampling = (*temperature > 0.0).then_some(Sampling {
                    temperature: *temperature,
                    top_p: 1.0,
                    seed: u64::from(*seed),
                });
                match engine.set_sampling(sampling) {
                    Ok(()) => StageReply::Ok,
                    Err(e) => StageReply::Error(e.to_string()),
                }
            }
            StageMessage::Step { hidden, .. } => match engine.run(Input::Hidden(hidden)).await {
                Ok(Output::Hidden(h)) => StageReply::Hidden(h),
                Ok(Output::Token(t)) => StageReply::Token(t),
                Err(e) => StageReply::Error(format!("{e:#}")),
            },
            StageMessage::End { .. } => {
                engine.reset();
                StageReply::Ok
            }
        };
        command.reply(reply);
    }
    member.leave().await;
    endpoint.close().await;
    Ok(())
}

fn default_cache() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("peerinfer").join("blobs")
}

/// Returns a model file from the cache, or receives it from the host and
/// saves it. Files are named by their BLAKE3 hash and checked on read.
async fn fetch(member: &TeamMember, file: &FileEntry, cache: &Path) -> Result<Vec<u8>> {
    let path = cache.join(file.hash.to_hex());
    if let Ok(bytes) = std::fs::read(&path)
        && bytes.len() as u64 == file.bytes
        && Hash::new(&bytes) == file.hash
    {
        println!("{} · reused", file.name);
        return Ok(bytes);
    }
    let started = std::time::Instant::now();
    let bytes = member.fetch(file).await?;
    let secs = started.elapsed().as_secs_f64();
    println!(
        "{} · {:.1} MB received in {secs:.1} s",
        file.name,
        bytes.len() as f64 / 1e6
    );
    let partial = path.with_extension("part");
    std::fs::write(&partial, &bytes)?;
    std::fs::rename(&partial, &path)?;
    Ok(bytes)
}
