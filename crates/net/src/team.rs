//! A team runs one model on several devices. Each device runs a contiguous
//! range of layers.
//!
//! The host runs the first layers, answers pool requests and owns the model
//! files. Members join with an [`Invitation`]. The host assigns each member
//! the next free layer range and the [`Manifest`] of the model's files; the
//! member downloads the files it needs from the host over iroh-blobs.
//!
//! A member connects to the host and keeps that connection. The host sends
//! each step over it: it opens one bidirectional stream per message, writes
//! one frame and reads one frame back. For each token, the host passes the
//! hidden states through the members in layer order; the last member returns
//! the token. A closed connection removes the member from the team.
//!
//! Frames use [`crate::proto`]'s format with a larger size limit, because the
//! hidden states of a long prompt can exceed one megabyte.

use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
    time::Duration,
};

use bao_tree::io::BaoContentItem;
use iroh::{
    Endpoint, EndpointAddr, EndpointId,
    endpoint::Connection,
    protocol::{AcceptError, ProtocolHandler},
};
use iroh_blobs::{Hash, get::request::GetBlobItem};
use n0_future::{StreamExt, task, time::timeout};
use serde::{Deserialize, Serialize};
use tokio::sync::{Notify, oneshot};
use tracing::{debug, warn};

use crate::proto::{FrameError, read_frame_max, write_frame_max};

/// ALPN for team messages.
pub const TEAM_ALPN: &[u8] = b"peerinfer/team/0";

/// Largest team frame: the hidden states of a long prompt.
pub const MAX_TEAM_FRAME: usize = 64 << 20;

/// Time a peer has to answer one message. A prompt step can take a while
/// on a slow device and over a relay.
const DEADLINE: Duration = Duration::from_secs(120);

/// Commands waiting for the member's model.
const COMMAND_CAPACITY: usize = 4;

#[derive(Debug, thiserror::Error)]
pub enum TeamError {
    #[error(transparent)]
    Frame(#[from] FrameError),
    #[error("connection: {0}")]
    Connection(String),
    #[error("timed out")]
    Timeout,
    #[error("{0}")]
    Remote(String),
    #[error("unexpected reply")]
    Unexpected,
    #[error("the invitation is not valid")]
    BadInvitation,
    #[error("the team is not complete")]
    Incomplete,
    #[error("blob: {0}")]
    Blob(String),
}

/// What a member needs to join: the host's address and the team secret.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Invitation {
    pub host: EndpointAddr,
    pub secret: [u8; 16],
}

impl Invitation {
    /// Hex text for a command line or a link.
    pub fn encode(&self) -> String {
        let bytes = postcard::to_stdvec(self).expect("an invitation serializes");
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    pub fn decode(text: &str) -> Result<Self, TeamError> {
        let text = text.trim();
        if !text.len().is_multiple_of(2) || text.len() > 8192 {
            return Err(TeamError::BadInvitation);
        }
        let bytes = (0..text.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&text[i..i + 2], 16))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| TeamError::BadInvitation)?;
        postcard::from_bytes(&bytes).map_err(|_| TeamError::BadInvitation)
    }
}

/// One model file the host offers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileEntry {
    pub name: String,
    pub hash: Hash,
    pub bytes: u64,
}

/// The model and its files.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub model: String,
    pub layers: u32,
    pub files: Vec<FileEntry>,
}

/// The layers a member runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Assignment {
    pub start: u32,
    pub end: u32,
    pub manifest: Manifest,
}

impl Assignment {
    /// True when this member runs the model's last layer and picks tokens.
    pub fn is_last(&self) -> bool {
        self.end == self.manifest.layers
    }
}

/// Splits `layers` into `devices` contiguous ranges that differ by at most
/// one layer. The host runs the first range.
pub fn split_layers(layers: u32, devices: u32) -> Vec<(u32, u32)> {
    let devices = devices.clamp(1, layers.max(1));
    let (base, extra) = (layers / devices, layers % devices);
    let mut start = 0;
    (0..devices)
        .map(|i| {
            let end = start + base + u32::from(i < extra);
            let range = (start, end);
            start = end;
            range
        })
        .collect()
}

/// Messages from a member to the host.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum ToHost {
    Join { secret: [u8; 16] },
    Ready,
    Leave,
}

/// Messages from the host to a member.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StageMessage {
    /// Starts a generation: clear the context and use these sampling settings.
    Begin {
        run: u64,
        temperature: f32,
        seed: u32,
    },
    /// Runs the member's layers on hidden states from the stage before.
    Step {
        run: u64,
        step: u32,
        hidden: Vec<f32>,
    },
    /// Ends a generation.
    End { run: u64 },
}

/// A member's answer to a [`StageMessage`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StageReply {
    Ok,
    Hidden(Vec<f32>),
    Token(u32),
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum FromHost {
    Assigned(Assignment),
    Ok,
    Error(String),
}

/// Sends one message on a new stream and reads one reply.
async fn request<Q: Serialize, R: serde::de::DeserializeOwned>(
    conn: &Connection,
    message: &Q,
) -> Result<R, TeamError> {
    timeout(DEADLINE, async {
        let (mut send, mut recv) = conn
            .open_bi()
            .await
            .map_err(|e| TeamError::Connection(e.to_string()))?;
        write_frame_max(&mut send, message, MAX_TEAM_FRAME).await?;
        send.finish()
            .map_err(|e| TeamError::Connection(e.to_string()))?;
        read_frame_max(&mut recv, MAX_TEAM_FRAME)
            .await?
            .ok_or(TeamError::Unexpected)
    })
    .await
    .map_err(|_| TeamError::Timeout)?
}

#[derive(Debug)]
struct Slot {
    start: u32,
    end: u32,
    member: Option<(EndpointId, Connection)>,
    ready: bool,
}

#[derive(Debug)]
struct HostState {
    secret: [u8; 16],
    manifest: Manifest,
    slots: Mutex<Vec<Slot>>,
    changed: Notify,
}

/// The host side of a team. Register [`TeamHost::handler`] for
/// [`TEAM_ALPN`] and serve the model files with iroh-blobs.
#[derive(Debug, Clone)]
pub struct TeamHost {
    endpoint: Endpoint,
    state: Arc<HostState>,
    next_run: Arc<Mutex<u64>>,
}

/// Status of one member slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotStatus {
    pub start: u32,
    pub end: u32,
    pub member: Option<EndpointId>,
    pub ready: bool,
}

impl TeamHost {
    /// A team with the host on `layers[0]` and one member slot for each
    /// further range.
    pub fn new(endpoint: Endpoint, manifest: Manifest, layers: &[(u32, u32)]) -> Self {
        let mut secret = [0u8; 16];
        getrandom::fill(&mut secret).expect("random bytes");
        let slots = layers
            .iter()
            .skip(1)
            .map(|&(start, end)| Slot {
                start,
                end,
                member: None,
                ready: false,
            })
            .collect();
        Self {
            endpoint,
            state: Arc::new(HostState {
                secret,
                manifest,
                slots: Mutex::new(slots),
                changed: Notify::new(),
            }),
            next_run: Arc::new(Mutex::new(1)),
        }
    }

    /// The handler to register for [`TEAM_ALPN`].
    pub fn handler(&self) -> TeamHandler {
        TeamHandler(self.state.clone())
    }

    pub fn invitation(&self) -> Invitation {
        Invitation {
            host: self.endpoint.addr(),
            secret: self.state.secret,
        }
    }

    pub fn slots(&self) -> Vec<SlotStatus> {
        let slots = self.state.slots.lock().expect("poisoned");
        slots
            .iter()
            .map(|s| SlotStatus {
                start: s.start,
                end: s.end,
                member: s.member.as_ref().map(|(id, _)| *id),
                ready: s.ready
                    && s.member
                        .as_ref()
                        .is_some_and(|(_, c)| c.close_reason().is_none()),
            })
            .collect()
    }

    /// True when every slot has a connected member that loaded its layers.
    pub fn is_complete(&self) -> bool {
        self.slots().iter().all(|s| s.ready)
    }

    /// Waits until the team changes: a member joins, gets ready or leaves.
    pub async fn changed(&self) {
        self.state.changed.notified().await;
    }

    /// Starts one generation on every member. Members run in layer order.
    pub async fn begin(&self, temperature: f32, seed: u32) -> Result<Run, TeamError> {
        let members: Vec<Connection> = {
            let slots = self.state.slots.lock().expect("poisoned");
            slots
                .iter()
                .map(|s| match (&s.member, s.ready) {
                    (Some((_, c)), true) if c.close_reason().is_none() => Ok(c.clone()),
                    _ => Err(TeamError::Incomplete),
                })
                .collect::<Result<_, _>>()?
        };
        let run = {
            let mut next = self.next_run.lock().expect("poisoned");
            *next += 1;
            *next
        };
        for conn in &members {
            expect_ok(
                request(
                    conn,
                    &StageMessage::Begin {
                        run,
                        temperature,
                        seed,
                    },
                )
                .await?,
            )?;
        }
        Ok(Run {
            run,
            step: 0,
            members,
        })
    }
}

fn expect_ok(reply: StageReply) -> Result<(), TeamError> {
    match reply {
        StageReply::Ok => Ok(()),
        StageReply::Error(e) => Err(TeamError::Remote(e)),
        _ => Err(TeamError::Unexpected),
    }
}

/// One generation on a team.
#[derive(Debug)]
pub struct Run {
    run: u64,
    step: u32,
    members: Vec<Connection>,
}

impl Run {
    /// Passes the host's hidden states through every member and returns the
    /// token that the last member picked.
    pub async fn step(&mut self, mut hidden: Vec<f32>) -> Result<u32, TeamError> {
        let step = self.step;
        self.step += 1;
        let last = self.members.len().saturating_sub(1);
        for (i, conn) in self.members.iter().enumerate() {
            let reply = request(
                conn,
                &StageMessage::Step {
                    run: self.run,
                    step,
                    hidden,
                },
            )
            .await?;
            match (reply, i == last) {
                (StageReply::Hidden(h), false) => hidden = h,
                (StageReply::Token(t), true) => return Ok(t),
                (StageReply::Error(e), _) => return Err(TeamError::Remote(e)),
                _ => return Err(TeamError::Unexpected),
            }
        }
        Err(TeamError::Incomplete)
    }

    /// Ends the generation on every member that is still connected.
    pub async fn end(self) {
        for conn in &self.members {
            let _ = request::<_, StageReply>(conn, &StageMessage::End { run: self.run }).await;
        }
    }
}

/// Accepts members on [`TEAM_ALPN`].
#[derive(Debug, Clone)]
pub struct TeamHandler(Arc<HostState>);

impl ProtocolHandler for TeamHandler {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        let state = &self.0;
        let peer = conn.remote_id();
        while let Ok((mut send, mut recv)) = conn.accept_bi().await {
            let Ok(Ok(Some(message))) = timeout(
                DEADLINE,
                read_frame_max::<_, ToHost>(&mut recv, MAX_TEAM_FRAME),
            )
            .await
            else {
                break;
            };
            let reply = match message {
                ToHost::Join { secret } if secret == state.secret => {
                    let mut slots = state.slots.lock().expect("poisoned");
                    // A member that reconnects takes its old slot back.
                    let free = slots
                        .iter()
                        .position(|s| s.member.as_ref().is_some_and(|(id, _)| *id == peer))
                        .or_else(|| {
                            slots.iter().position(|s| {
                                s.member
                                    .as_ref()
                                    .is_none_or(|(_, c)| c.close_reason().is_some())
                            })
                        });
                    match free {
                        Some(i) => {
                            let slot = &mut slots[i];
                            slot.member = Some((peer, conn.clone()));
                            slot.ready = false;
                            debug!(%peer, start = slot.start, end = slot.end, "member joined");
                            FromHost::Assigned(Assignment {
                                start: slot.start,
                                end: slot.end,
                                manifest: state.manifest.clone(),
                            })
                        }
                        None => FromHost::Error("This team already has all its devices.".into()),
                    }
                }
                ToHost::Join { .. } => FromHost::Error("The invitation is not valid.".into()),
                ToHost::Ready | ToHost::Leave => {
                    let mut slots = state.slots.lock().expect("poisoned");
                    match slots
                        .iter_mut()
                        .find(|s| s.member.as_ref().is_some_and(|(id, _)| *id == peer))
                    {
                        Some(slot) if matches!(message, ToHost::Ready) => {
                            slot.ready = true;
                            FromHost::Ok
                        }
                        Some(slot) => {
                            slot.member = None;
                            slot.ready = false;
                            FromHost::Ok
                        }
                        None => FromHost::Error("Join this team first.".into()),
                    }
                }
            };
            state.changed.notify_waiters();
            if write_frame_max(&mut send, &reply, MAX_TEAM_FRAME)
                .await
                .is_err()
                || send.finish().is_err()
            {
                break;
            }
        }
        // The connection stays open while the host sends steps over it; this
        // returns when the member closes it.
        conn.closed().await;
        warn!(%peer, "member disconnected");
        state.changed.notify_waiters();
        Ok(())
    }
}

/// One [`StageMessage`] for the member's model, with a way to answer it.
#[derive(Debug)]
pub struct Command {
    pub message: StageMessage,
    reply: oneshot::Sender<StageReply>,
}

impl Command {
    pub fn reply(self, reply: StageReply) {
        let _ = self.reply.send(reply);
    }
}

/// The member side of a team.
#[derive(Debug)]
pub struct TeamMember {
    endpoint: Endpoint,
    conn: Connection,
    host: EndpointAddr,
    assignment: Assignment,
    commands: async_channel::Receiver<Command>,
    _accept: n0_future::task::AbortOnDropHandle<()>,
}

impl TeamMember {
    /// Connects to the host and takes a layer range.
    pub async fn join(endpoint: Endpoint, invitation: &Invitation) -> Result<Self, TeamError> {
        let conn = endpoint
            .connect(invitation.host.clone(), TEAM_ALPN)
            .await
            .map_err(|e| TeamError::Connection(e.to_string()))?;
        let assignment = match request(
            &conn,
            &ToHost::Join {
                secret: invitation.secret,
            },
        )
        .await?
        {
            FromHost::Assigned(a) => a,
            FromHost::Error(e) => return Err(TeamError::Remote(e)),
            FromHost::Ok => return Err(TeamError::Unexpected),
        };
        let (tx, commands) = async_channel::bounded(COMMAND_CAPACITY);
        let accept = task::spawn(serve_host(conn.clone(), tx));
        Ok(Self {
            endpoint,
            conn,
            host: invitation.host.clone(),
            assignment,
            commands,
            _accept: n0_future::task::AbortOnDropHandle::new(accept),
        })
    }

    pub fn assignment(&self) -> &Assignment {
        &self.assignment
    }

    /// Downloads one model file from the host. iroh-blobs verifies it
    /// against the hash in the manifest.
    pub async fn fetch(&self, file: &FileEntry) -> Result<Vec<u8>, TeamError> {
        let conn = self
            .endpoint
            .connect(self.host.clone(), iroh_blobs::ALPN)
            .await
            .map_err(|e| TeamError::Connection(e.to_string()))?;
        let expected =
            usize::try_from(file.bytes).map_err(|_| TeamError::Blob("file too large".into()))?;
        let mut stream = iroh_blobs::get::request::get_blob(conn, file.hash);
        let mut bytes = Vec::with_capacity(expected);
        while let Some(item) = stream.next().await {
            match item {
                GetBlobItem::Item(BaoContentItem::Leaf(leaf)) => {
                    if leaf.offset != bytes.len() as u64
                        || leaf.data.len() > expected.saturating_sub(bytes.len())
                    {
                        return Err(TeamError::Blob(format!(
                            "{} has invalid offsets",
                            file.name
                        )));
                    }
                    bytes.extend_from_slice(&leaf.data);
                }
                GetBlobItem::Item(_) => {}
                GetBlobItem::Done(_) if bytes.len() == expected => return Ok(bytes),
                GetBlobItem::Done(_) => {
                    return Err(TeamError::Blob(format!("{} has the wrong size", file.name)));
                }
                GetBlobItem::Error(e) => return Err(TeamError::Blob(e.to_string())),
            }
        }
        Err(TeamError::Blob(format!(
            "{} transfer ended early",
            file.name
        )))
    }

    /// Tells the host that the member's layers are loaded.
    pub async fn ready(&self) -> Result<(), TeamError> {
        match request(&self.conn, &ToHost::Ready).await? {
            FromHost::Ok => Ok(()),
            FromHost::Error(e) => Err(TeamError::Remote(e)),
            FromHost::Assigned(_) => Err(TeamError::Unexpected),
        }
    }

    /// The next message for the model. `None` when the host disconnects.
    pub async fn next(&self) -> Option<Command> {
        self.commands.recv().await.ok()
    }

    /// Leaves the team and closes the connection.
    pub async fn leave(self) {
        let _ = request::<_, FromHost>(&self.conn, &ToHost::Leave).await;
        self.conn.close(0u32.into(), b"leave");
    }
}

/// Answers the host's messages on the member's connection, in order.
async fn serve_host(conn: Connection, commands: async_channel::Sender<Command>) {
    let mut session = Session::default();
    while let Ok((mut send, mut recv)) = conn.accept_bi().await {
        let Ok(Ok(Some(message))) = timeout(
            DEADLINE,
            read_frame_max::<_, StageMessage>(&mut recv, MAX_TEAM_FRAME),
        )
        .await
        else {
            break;
        };
        let reply = match session.check(&message) {
            Err(e) => StageReply::Error(e.into()),
            Ok(()) => {
                let (tx, rx) = oneshot::channel();
                if commands.send(Command { message, reply: tx }).await.is_err() {
                    break;
                }
                rx.await
                    .unwrap_or_else(|_| StageReply::Error("the model stopped".into()))
            }
        };
        if write_frame_max(&mut send, &reply, MAX_TEAM_FRAME)
            .await
            .is_err()
            || send.finish().is_err()
        {
            break;
        }
    }
    commands.close();
}

/// Rejects steps out of order, so a replayed or lost step cannot corrupt
/// the KV cache.
#[derive(Debug, Default)]
struct Session {
    run: Option<u64>,
    step: u32,
    seen: HashSet<u64>,
}

impl Session {
    fn check(&mut self, message: &StageMessage) -> Result<(), &'static str> {
        match *message {
            StageMessage::Begin { run, .. } => {
                if !self.seen.insert(run) {
                    return Err("This generation was already used.");
                }
                self.run = Some(run);
                self.step = 0;
                Ok(())
            }
            StageMessage::Step { run, step, .. } => {
                if self.run != Some(run) || step != self.step {
                    return Err("Out-of-order generation step.");
                }
                self.step += 1;
                Ok(())
            }
            StageMessage::End { run } => {
                if self.run != Some(run) {
                    return Err("Unknown generation.");
                }
                self.run = None;
                self.step = 0;
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_layers_is_contiguous_and_balanced() {
        assert_eq!(split_layers(30, 2), vec![(0, 15), (15, 30)]);
        assert_eq!(split_layers(32, 3), vec![(0, 11), (11, 22), (22, 32)]);
        assert_eq!(split_layers(4, 9), vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
        assert_eq!(split_layers(30, 1), vec![(0, 30)]);
    }

    #[test]
    fn session_rejects_replays_and_reordering() {
        let mut s = Session::default();
        let step = |step| StageMessage::Step {
            run: 1,
            step,
            hidden: vec![],
        };
        assert!(s.check(&step(0)).is_err());
        assert!(
            s.check(&StageMessage::Begin {
                run: 1,
                temperature: 0.0,
                seed: 0
            })
            .is_ok()
        );
        assert!(s.check(&step(1)).is_err());
        assert!(s.check(&step(0)).is_ok());
        assert!(s.check(&step(0)).is_err());
        assert!(s.check(&step(1)).is_ok());
        assert!(s.check(&StageMessage::End { run: 1 }).is_ok());
        assert!(
            s.check(&StageMessage::Begin {
                run: 1,
                temperature: 0.0,
                seed: 0
            })
            .is_err()
        );
    }
}
