//! `peerinfer worker`: answers pool requests with a model on this machine's
//! GPU, alone or as the host of a team.

use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use clap::Args;
use iroh::protocol::Router;
use iroh_blobs::{BlobsProtocol, store::mem::MemStore};
use peerinfer_core::{Engine, Input, Output, Sampling, Weights, model::WeightIndex};
use peerinfer_net::{
    GOSSIP_ALPN, Gossip, bind_endpoint, default_bootstrap,
    pool::Pool,
    presence::{WorkerInfo, WorkerState},
    proto::{ErrorCode, INFER_ALPN},
    team::{FileEntry, Manifest, TEAM_ALPN, TeamHost, split_layers},
    worker::{Job, JobQueue, job_queue},
};

use crate::{
    PoolArgs, default_key_path,
    model::{CONTEXT_LIMIT, DEFAULT_NEW_TOKENS, MAX_NEW_TOKENS, ModelDir, TextStream},
    net,
};

/// Requests that may wait while one runs.
const QUEUE_CAPACITY: usize = 2;

#[derive(Debug, Args)]
pub struct WorkerArgs {
    /// Model directory in MLC q4f16_1 format.
    #[arg(long)]
    model: PathBuf,
    /// Model ID to announce. Default: the directory name. Clients choose
    /// workers by this ID.
    #[arg(long)]
    model_id: Option<String>,
    /// Devices that share the model, including this one. With more than 1,
    /// this worker hosts a team and prints an invitation for the others.
    #[arg(long, default_value_t = 1)]
    devices: u32,
    /// File that holds this worker's secret key. The endpoint ID stays the
    /// same while the file exists.
    #[arg(long)]
    key_file: Option<PathBuf>,
    #[command(flatten)]
    pool: PoolArgs,
}

pub async fn run(args: WorkerArgs) -> Result<()> {
    let model = Arc::new(ModelDir::open(&args.model, args.model_id.clone())?);
    let index_file = if args.model.join("tensor-cache.json").exists() {
        "tensor-cache.json"
    } else {
        "ndarray-cache.json"
    };
    let config = peerinfer_core::Config::from_mlc_chat_config(&std::fs::read_to_string(
        args.model.join("mlc-chat-config.json"),
    )?)?;
    let layers = config.num_hidden_layers as u32;
    ensure!(args.devices >= 1, "--devices must be at least 1");
    ensure!(
        args.devices <= layers,
        "--devices cannot be more than the model's {layers} layers"
    );
    let ranges = split_layers(layers, args.devices);
    let mine = ranges[0].0 as usize..ranges[0].1 as usize;

    let key_path = args
        .key_file
        .clone()
        .unwrap_or_else(|| default_key_path("worker.key"));
    let key = net::load_or_create_key(&key_path)?;
    let endpoint = bind_endpoint(
        key.clone(),
        vec![
            INFER_ALPN.to_vec(),
            GOSSIP_ALPN.to_vec(),
            TEAM_ALPN.to_vec(),
            iroh_blobs::ALPN.to_vec(),
        ],
    )
    .await?;
    let gossip = Gossip::builder().spawn(endpoint.clone());
    let (handler, queue) = job_queue(QUEUE_CAPACITY);
    queue.set_paused(true);
    let mut router = Router::builder(endpoint.clone())
        .accept(INFER_ALPN, handler)
        .accept(GOSSIP_ALPN, gossip.clone());

    // A team host offers the model files and a slot for each other device.
    let team = if args.devices > 1 {
        let store = MemStore::new();
        let mut names = vec!["mlc-chat-config.json".to_string(), index_file.to_string()];
        let index = WeightIndex::from_ndarray_cache(&std::fs::read_to_string(
            args.model.join(index_file),
        )?)?;
        names.extend(index.shards.iter().cloned());
        let mut files = Vec::new();
        for name in names {
            let bytes =
                std::fs::read(args.model.join(&name)).with_context(|| format!("read {name}"))?;
            let len = bytes.len() as u64;
            let tag = store.add_bytes(bytes).await?;
            files.push(FileEntry {
                name,
                hash: tag.hash,
                bytes: len,
            });
        }
        let manifest = Manifest {
            model: model.id.clone(),
            layers,
            files,
        };
        let host = TeamHost::new(endpoint.clone(), manifest, &ranges);
        router = router
            .accept(TEAM_ALPN, host.handler())
            .accept(iroh_blobs::ALPN, BlobsProtocol::new(&store, None));
        Some((host, store))
    } else {
        None
    };
    let router = router.spawn();

    let mut bootstrap = args.pool.bootstrap.clone();
    if bootstrap.is_empty() {
        bootstrap = default_bootstrap();
    }
    if bootstrap.is_empty() {
        bail!("no bootstrap node is set. Use --bootstrap <ENDPOINT_ID> or set PEERINFER_BOOTSTRAP");
    }
    let pool = Pool::join(&gossip, key, &args.pool.pool, bootstrap).await?;

    println!("worker {}", endpoint.id());
    println!(
        "model {} (layers {}–{} of {layers})",
        model.id,
        mine.start,
        mine.end - 1
    );
    let weights = Weights::read_dir_stage(&args.model, Some(mine.clone()))?;
    let mut engine = Engine::new_stage(&weights, mine).await?;
    drop(weights);
    let adapter = {
        let i = engine.adapter_info();
        format!("{} ({:?})", i.name, i.backend)
    };
    println!("gpu {adapter}");

    let mut status = Status::new(model.id.clone(), adapter);
    if let Some((host, _)) = &team {
        println!();
        println!("This worker hosts a team of {} devices.", args.devices);
        println!("On each other device, run:");
        println!();
        println!("peerinfer join {}", host.invitation().encode());
        println!();
        for (i, (start, end)) in ranges.iter().enumerate().skip(1) {
            println!("device {}: layers {start}–{}", i + 1, end - 1);
        }
    }

    let team_ready =
        |team: &Option<(TeamHost, MemStore)>| team.as_ref().is_none_or(|(h, _)| h.is_complete());
    let mut was_ready = false;
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    loop {
        let ready = team_ready(&team);
        if ready != was_ready {
            was_ready = ready;
            queue.set_paused(!ready);
            if team.is_some() {
                println!(
                    "{}",
                    if ready {
                        "team ready"
                    } else {
                        "waiting for team members"
                    }
                );
            } else {
                println!("ready");
            }
        }
        status.state = if ready {
            WorkerState::Idle
        } else {
            WorkerState::Paused
        };
        status.waiting = queue.waiting();
        pool.announce(status.info());

        tokio::select! {
            _ = tokio::signal::ctrl_c() => break,
            _ = async { team.as_ref().unwrap().0.changed().await }, if team.is_some() => {
                if let Some((host, _)) = &team {
                    for (i, s) in host.slots().iter().enumerate() {
                        let who = s.member.map(|m| m.fmt_short().to_string()).unwrap_or_else(|| "-".into());
                        println!("device {}: layers {}–{} {who} {}", i + 2, s.start, s.end - 1, if s.ready { "ready" } else { "" });
                    }
                }
            }
            _ = tick.tick() => {}
            job = next_job(&queue), if ready => {
                let Some(job) = job else { break };
                status.state = WorkerState::Busy;
                status.active = 1;
                pool.announce(status.info());
                let started = Instant::now();
                let team_host = team.as_ref().map(|(h, _)| h);
                match answer(&mut engine, &model, job, team_host).await {
                    Ok(tokens) => {
                        status.served += 1;
                        status.tokens_out += u64::from(tokens);
                        let secs = started.elapsed().as_secs_f32();
                        if secs > 0.0 {
                            status.tokens_per_sec = tokens as f32 / secs;
                        }
                    }
                    Err(e) => eprintln!("request failed: {e:#}"),
                }
                status.active = 0;
            }
        }
    }
    let _ = pool.leave().await;
    let _ = tokio::time::timeout(Duration::from_secs(3), router.shutdown()).await;
    Ok(())
}

async fn next_job(queue: &JobQueue) -> Option<Job> {
    queue.next().await
}

/// Self-reported numbers for pool announcements.
struct Status {
    state: WorkerState,
    model: String,
    adapter: String,
    active: u8,
    waiting: usize,
    served: u64,
    tokens_out: u64,
    tokens_per_sec: f32,
    started: Instant,
}

impl Status {
    fn new(model: String, adapter: String) -> Self {
        Self {
            state: WorkerState::Paused,
            model,
            adapter,
            active: 0,
            waiting: 0,
            served: 0,
            tokens_out: 0,
            tokens_per_sec: 0.0,
            started: Instant::now(),
        }
    }

    fn info(&self) -> WorkerInfo {
        WorkerInfo {
            state: self.state,
            model: self.model.clone(),
            adapter: self.adapter.clone(),
            active: self.active,
            waiting: self.waiting.min(255) as u8,
            served: self.served,
            tokens_out: self.tokens_out,
            tokens_per_sec: self.tokens_per_sec,
            uptime_secs: self.started.elapsed().as_secs(),
        }
    }
}

/// Answers one request and returns the number of generated tokens.
async fn answer(
    engine: &mut Engine,
    model: &Arc<ModelDir>,
    mut job: Job,
    team: Option<&TeamHost>,
) -> Result<u32> {
    let request = job.request.clone();
    let prepared = (|| -> Result<_> {
        let prompt = model.encode(&model.format(&request.messages)?)?;
        let max = request.max_tokens.unwrap_or(DEFAULT_NEW_TOKENS);
        ensure!(
            (1..=MAX_NEW_TOKENS).contains(&max),
            "choose between 1 and {MAX_NEW_TOKENS} output tokens"
        );
        let temperature = request.temperature.unwrap_or(0.7);
        ensure!(
            (0.0..=2.0).contains(&temperature),
            "temperature must be between 0 and 2"
        );
        ensure!(
            prompt.len() + max as usize <= CONTEXT_LIMIT,
            "the prompt and reply must fit in {CONTEXT_LIMIT} tokens"
        );
        Ok((prompt, max, temperature))
    })();
    let (prompt, max, temperature) = match prepared {
        Ok(p) => p,
        Err(e) => {
            job.fail(ErrorCode::BadRequest, e.to_string()).await?;
            return Ok(0);
        }
    };
    let seed = getrandom::u64()?;
    engine.reset();
    let result = match team {
        None => {
            let (returned, result) =
                generate_alone(engine, model, job, &prompt, max, temperature, seed).await;
            job = returned;
            result
        }
        Some(team) => {
            generate_team(
                engine,
                model,
                &mut job,
                team,
                &prompt,
                max,
                temperature,
                seed,
            )
            .await
        }
    };
    engine.reset();
    match result {
        Ok(n) => {
            job.done(model.id.clone(), prompt.len() as u32, n).await?;
            Ok(n)
        }
        Err(e) => {
            let _ = job.fail(ErrorCode::Inference, format!("{e:#}")).await;
            Err(e)
        }
    }
}

/// Generates with the whole model on this GPU. Tokens stay on the GPU; a
/// separate task turns them into text and sends it. Returns the job.
async fn generate_alone(
    engine: &mut Engine,
    model: &Arc<ModelDir>,
    job: Job,
    prompt: &[u32],
    max: u32,
    temperature: f32,
    seed: u64,
) -> (Job, Result<u32>) {
    let sampling = (temperature > 0.0).then_some(Sampling {
        temperature,
        top_p: 1.0,
        seed,
    });
    if let Err(e) = engine.set_sampling(sampling) {
        return (job, Err(e));
    }
    let first = match engine.forward(prompt).await {
        Ok(t) => t,
        Err(e) => return (job, Err(e)),
    };
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Vec<u32>>();
    tx.send(vec![first]).ok();

    let sender = {
        let model = model.clone();
        let stop = stop.clone();
        let mut job = job;
        tokio::spawn(async move {
            let mut text = TextStream::new();
            let result: Result<()> = async {
                while let Some(batch) = rx.recv().await {
                    for t in batch {
                        if model.stop.contains(&t)
                            || text.len() >= max as usize
                            || job.is_cancelled()
                        {
                            return Ok(());
                        }
                        if let Some(piece) = text.push(&model, t)? {
                            job.chunk(piece).await?;
                        }
                    }
                }
                Ok(())
            }
            .await;
            stop.store(true, Ordering::Relaxed);
            (job, text.len() as u32, result)
        })
    };
    let mut generated = Ok(());
    if max > 1 && !model.stop.contains(&first) {
        let s = stop.clone();
        generated = engine
            .generate(max as usize - 1, |batch| {
                let _ = tx.send(batch.to_vec());
                !s.load(Ordering::Relaxed)
            })
            .await;
    }
    drop(tx);
    let (job, count, result) = sender.await.expect("the sender task does not panic");
    (job, generated.and(result).map(|()| count))
}

/// Generates with a team: this GPU runs the first layers, the members the
/// rest, one token at a time.
#[allow(clippy::too_many_arguments)]
async fn generate_team(
    engine: &mut Engine,
    model: &ModelDir,
    job: &mut Job,
    team: &TeamHost,
    prompt: &[u32],
    max: u32,
    temperature: f32,
    seed: u64,
) -> Result<u32> {
    let mut run = team.begin(temperature, seed as u32).await?;
    let mut text = TextStream::new();
    let mut input = prompt.to_vec();
    let result: Result<()> = async {
        loop {
            let Output::Hidden(hidden) = engine.run(Input::Tokens(&input)).await? else {
                bail!("the first stage returned a token");
            };
            let token = run.step(hidden).await?;
            if model.stop.contains(&token) || text.len() >= max as usize || job.is_cancelled() {
                return Ok(());
            }
            if let Some(piece) = text.push(model, token)? {
                job.chunk(piece).await?;
            }
            input = vec![token];
        }
    }
    .await;
    run.end().await;
    result.map(|()| text.len() as u32)
}
