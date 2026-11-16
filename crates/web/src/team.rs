//! Browser bindings for teams (`peerinfer_net::team`): a tab can host a
//! team or run one layer range of another device's team.

use std::sync::Arc;

use iroh::{SecretKey, protocol::Router};
use iroh_blobs::{BlobsProtocol, Hash, store::mem::MemStore};
use peerinfer_net::team::{
    Command, FileEntry, Invitation, Manifest, Run, StageMessage, StageReply, TEAM_ALPN, TeamHost,
    TeamMember, split_layers,
};
use serde::Serialize;
use tokio::sync::Mutex;
use wasm_bindgen::prelude::*;

use crate::{js_err, to_js, transport::bind};

/// Hosts a team: offers the model files and routes each step through the
/// members in layer order.
#[wasm_bindgen]
pub struct TeamHostPeer {
    router: Router,
    host: TeamHost,
    first: (u32, u32),
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SlotJs {
    start: u32,
    end: u32,
    member: Option<String>,
    ready: bool,
}

#[wasm_bindgen]
impl TeamHostPeer {
    /// Starts a team of `devices` for a model with `layers` layers. `names`
    /// and `files` are the model files a member may need, in the same order.
    pub async fn start(
        model: String,
        layers: u32,
        devices: u32,
        names: Vec<String>,
        files: Vec<js_sys::Uint8Array>,
        relay: Option<String>,
    ) -> Result<TeamHostPeer, JsError> {
        if names.len() != files.len() {
            return Err(js_err("names and files differ in length"));
        }
        if devices < 2 || devices > layers {
            return Err(js_err(
                "a team needs between 2 devices and one device per layer",
            ));
        }
        let endpoint = bind(
            SecretKey::generate(),
            vec![TEAM_ALPN.to_vec(), iroh_blobs::ALPN.to_vec()],
            relay,
        )
        .await?;
        let store = MemStore::new();
        let mut entries = Vec::with_capacity(names.len());
        for (name, file) in names.into_iter().zip(files) {
            let bytes = file.to_vec();
            let len = bytes.len() as u64;
            let tag = store.add_bytes(bytes).await.map_err(js_err)?;
            entries.push(FileEntry {
                name,
                hash: tag.hash,
                bytes: len,
            });
        }
        let ranges = split_layers(layers, devices);
        let host = TeamHost::new(
            endpoint.clone(),
            Manifest {
                model,
                layers,
                files: entries,
            },
            &ranges,
        );
        let router = Router::builder(endpoint)
            .accept(TEAM_ALPN, host.handler())
            .accept(iroh_blobs::ALPN, BlobsProtocol::new(&store, None))
            .spawn();
        Ok(Self {
            router,
            host,
            first: ranges[0],
        })
    }

    /// Waits until the endpoint has a relay address, so the invitation works.
    pub async fn online(&self) {
        self.router.endpoint().online().await;
    }

    /// The text a member needs to join.
    pub fn invitation(&self) -> String {
        self.host.invitation().encode()
    }

    /// The layers the host runs, as `[start, end]`.
    #[wasm_bindgen(js_name = hostLayers)]
    pub fn host_layers(&self) -> Vec<u32> {
        vec![self.first.0, self.first.1]
    }

    /// Member slots: `{ start, end, member, ready }`.
    pub fn slots(&self) -> Result<JsValue, JsError> {
        let slots: Vec<SlotJs> = self
            .host
            .slots()
            .into_iter()
            .map(|s| SlotJs {
                start: s.start,
                end: s.end,
                member: s.member.map(|m| m.to_string()),
                ready: s.ready,
            })
            .collect();
        to_js(&slots)
    }

    #[wasm_bindgen(js_name = isComplete)]
    pub fn is_complete(&self) -> bool {
        self.host.is_complete()
    }

    /// Resolves when a member joins, gets ready or leaves.
    pub async fn changed(&self) {
        self.host.changed().await;
    }

    /// Starts one generation on every member.
    pub async fn begin(&self, temperature: f32, seed: u32) -> Result<TeamRun, JsError> {
        let run = self.host.begin(temperature, seed).await.map_err(js_err)?;
        Ok(TeamRun(Mutex::new(Some(run))))
    }

    pub async fn shutdown(&self) {
        let _ = self.router.shutdown().await;
    }
}

/// One generation on a team.
#[wasm_bindgen]
pub struct TeamRun(Mutex<Option<Run>>);

#[wasm_bindgen]
impl TeamRun {
    /// Passes the host's hidden states through the members and returns the
    /// token the last member picked.
    pub async fn step(&self, hidden: Vec<f32>) -> Result<u32, JsError> {
        let mut run = self.0.lock().await;
        let run = run
            .as_mut()
            .ok_or_else(|| js_err("this generation has ended"))?;
        run.step(hidden).await.map_err(js_err)
    }

    pub async fn end(&self) {
        if let Some(run) = self.0.lock().await.take() {
            run.end().await;
        }
    }
}

/// Runs one layer range of a team.
#[wasm_bindgen]
pub struct TeamMemberPeer {
    member: Arc<TeamMember>,
    endpoint: iroh::Endpoint,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AssignmentJs {
    start: u32,
    end: u32,
    model: String,
    layers: u32,
    files: Vec<FileJs>,
}

#[derive(Serialize)]
struct FileJs {
    name: String,
    hash: String,
    bytes: u64,
}

#[wasm_bindgen]
impl TeamMemberPeer {
    /// Joins the team in `invitation` and takes the next free layer range.
    pub async fn join(
        invitation: String,
        relay: Option<String>,
    ) -> Result<TeamMemberPeer, JsError> {
        let invitation = Invitation::decode(&invitation).map_err(js_err)?;
        let endpoint = bind(SecretKey::generate(), vec![], relay).await?;
        let member = TeamMember::join(endpoint.clone(), &invitation)
            .await
            .map_err(js_err)?;
        Ok(Self {
            member: Arc::new(member),
            endpoint,
        })
    }

    /// `{ start, end, model, layers, files: [{ name, hash, bytes }] }`.
    pub fn assignment(&self) -> Result<JsValue, JsError> {
        let a = self.member.assignment();
        to_js(&AssignmentJs {
            start: a.start,
            end: a.end,
            model: a.manifest.model.clone(),
            layers: a.manifest.layers,
            files: a
                .manifest
                .files
                .iter()
                .map(|f| FileJs {
                    name: f.name.clone(),
                    hash: f.hash.to_string(),
                    bytes: f.bytes,
                })
                .collect(),
        })
    }

    /// Receives one model file from the host, checked against its hash.
    pub async fn fetch(&self, name: String) -> Result<Vec<u8>, JsError> {
        let file = self
            .member
            .assignment()
            .manifest
            .files
            .iter()
            .find(|f| f.name == name)
            .cloned()
            .ok_or_else(|| js_err(format!("the host offers no {name}")))?;
        self.member.fetch(&file).await.map_err(js_err)
    }

    /// Tells the host that this device's layers are loaded.
    pub async fn ready(&self) -> Result<(), JsError> {
        self.member.ready().await.map_err(js_err)
    }

    /// The next message from the host. `undefined` when the host disconnects.
    pub async fn next(&self) -> Option<TeamCommand> {
        self.member.next().await.map(|c| TeamCommand(Some(c)))
    }

    /// Leaves the team and closes the endpoint.
    pub async fn leave(&self) {
        self.member.leave().await;
        self.endpoint.close().await;
    }

    /// True when `bytes` have the BLAKE3 hash `hash`.
    pub fn verify(bytes: &[u8], hash: String) -> Result<bool, JsError> {
        let hash: Hash = hash.parse().map_err(js_err)?;
        Ok(Hash::new(bytes) == hash)
    }
}

/// One message from the host, with a way to answer it.
#[wasm_bindgen]
pub struct TeamCommand(Option<Command>);

#[wasm_bindgen]
impl TeamCommand {
    /// "begin", "step" or "end".
    pub fn kind(&self) -> String {
        match self.0.as_ref().map(|c| &c.message) {
            Some(StageMessage::Begin { .. }) => "begin",
            Some(StageMessage::Step { .. }) => "step",
            Some(StageMessage::End { .. }) => "end",
            None => "answered",
        }
        .into()
    }

    pub fn temperature(&self) -> f32 {
        match self.0.as_ref().map(|c| &c.message) {
            Some(StageMessage::Begin { temperature, .. }) => *temperature,
            _ => 0.0,
        }
    }

    pub fn seed(&self) -> u32 {
        match self.0.as_ref().map(|c| &c.message) {
            Some(StageMessage::Begin { seed, .. }) => *seed,
            _ => 0,
        }
    }

    /// The hidden states of a step.
    pub fn hidden(&self) -> Vec<f32> {
        match self.0.as_ref().map(|c| &c.message) {
            Some(StageMessage::Step { hidden, .. }) => hidden.clone(),
            _ => Vec::new(),
        }
    }

    #[wasm_bindgen(js_name = replyOk)]
    pub fn reply_ok(&mut self) {
        self.reply(StageReply::Ok);
    }

    #[wasm_bindgen(js_name = replyHidden)]
    pub fn reply_hidden(&mut self, hidden: Vec<f32>) {
        self.reply(StageReply::Hidden(hidden));
    }

    #[wasm_bindgen(js_name = replyToken)]
    pub fn reply_token(&mut self, token: u32) {
        self.reply(StageReply::Token(token));
    }

    #[wasm_bindgen(js_name = replyError)]
    pub fn reply_error(&mut self, message: String) {
        self.reply(StageReply::Error(message));
    }
}

impl TeamCommand {
    fn reply(&mut self, reply: StageReply) {
        if let Some(c) = self.0.take() {
            c.reply(reply);
        }
    }
}
