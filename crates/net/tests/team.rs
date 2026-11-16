//! A three-device team over real iroh endpoints on localhost.

use iroh::{Endpoint, SecretKey, endpoint::presets, protocol::Router};
use iroh_blobs::{BlobsProtocol, store::mem::MemStore};
use peerinfer_net::team::{
    FileEntry, Manifest, StageMessage, StageReply, TEAM_ALPN, TeamError, TeamHost, TeamMember,
    split_layers,
};

async fn local_endpoint(alpns: Vec<Vec<u8>>) -> Endpoint {
    Endpoint::builder(presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(alpns)
        .bind()
        .await
        .unwrap()
}

/// A member whose model adds `offset` to each hidden value, or, on the last
/// stage, returns the sum of the values as the token.
fn fake_stage(member: TeamMember, offset: f32) -> tokio::task::JoinHandle<Vec<StageMessage>> {
    tokio::spawn(async move {
        let last = member.assignment().is_last();
        let mut seen = Vec::new();
        while let Some(cmd) = member.next().await {
            let reply = match &cmd.message {
                StageMessage::Step { hidden, .. } if last => {
                    StageReply::Token(hidden.iter().sum::<f32>() as u32)
                }
                StageMessage::Step { hidden, .. } => {
                    StageReply::Hidden(hidden.iter().map(|x| x + offset).collect())
                }
                _ => StageReply::Ok,
            };
            seen.push(cmd.message.clone());
            cmd.reply(reply);
        }
        seen
    })
}

#[tokio::test]
async fn three_devices_pass_hidden_states_in_layer_order() {
    let file = b"model weights".repeat(1000);
    let store = MemStore::new();
    let tag = store.add_bytes(file.clone()).await.unwrap();
    let manifest = Manifest {
        model: "test-model".into(),
        layers: 30,
        files: vec![FileEntry {
            name: "params_shard_0.bin".into(),
            hash: tag.hash,
            bytes: file.len() as u64,
        }],
    };
    let layers = split_layers(30, 3);
    let host_ep = local_endpoint(vec![TEAM_ALPN.to_vec(), iroh_blobs::ALPN.to_vec()]).await;
    let host = TeamHost::new(host_ep.clone(), manifest.clone(), &layers);
    let router = Router::builder(host_ep)
        .accept(TEAM_ALPN, host.handler())
        .accept(iroh_blobs::ALPN, BlobsProtocol::new(&store, None))
        .spawn();
    let invitation = host.invitation();
    assert!(!host.is_complete());

    // The invitation survives its text form.
    let invitation = peerinfer_net::team::Invitation::decode(&invitation.encode()).unwrap();

    let a = TeamMember::join(local_endpoint(vec![]).await, &invitation)
        .await
        .unwrap();
    let b = TeamMember::join(local_endpoint(vec![]).await, &invitation)
        .await
        .unwrap();
    assert_eq!((a.assignment().start, a.assignment().end), (10, 20));
    assert_eq!((b.assignment().start, b.assignment().end), (20, 30));
    assert_eq!(a.assignment().manifest, manifest);

    // A third member finds no free slot.
    let extra = TeamMember::join(local_endpoint(vec![]).await, &invitation).await;
    assert!(matches!(extra, Err(TeamError::Remote(_))));

    // Files arrive verified against the manifest.
    assert_eq!(a.fetch(&manifest.files[0]).await.unwrap(), file);
    let mut wrong = manifest.files[0].clone();
    wrong.bytes += 1;
    assert!(a.fetch(&wrong).await.is_err());

    assert!(matches!(
        host.begin(0.0, 1).await,
        Err(TeamError::Incomplete)
    ));
    a.ready().await.unwrap();
    b.ready().await.unwrap();
    assert!(host.is_complete());

    let seen_a = fake_stage(a, 1.0);
    let seen_b = fake_stage(b, 0.0);

    let mut run = host.begin(0.5, 7).await.unwrap();
    // Member a adds 1 to each of 4 values; member b returns their sum.
    assert_eq!(run.step(vec![1.0, 2.0, 3.0, 4.0]).await.unwrap(), 14);
    assert_eq!(run.step(vec![10.0]).await.unwrap(), 11);
    run.end().await;

    // Closing the host's connections ends both members.
    drop(host);
    router.shutdown().await.unwrap();
    let seen_a = seen_a.await.unwrap();
    assert_eq!(
        seen_a.first(),
        Some(&StageMessage::Begin {
            run: 2,
            temperature: 0.5,
            seed: 7
        })
    );
    assert_eq!(seen_a.len(), 4); // begin, two steps, end
    assert_eq!(seen_b.await.unwrap().len(), 4);
}

#[tokio::test]
async fn a_wrong_secret_cannot_join() {
    let manifest = Manifest {
        model: "m".into(),
        layers: 2,
        files: vec![],
    };
    let host_ep = local_endpoint(vec![TEAM_ALPN.to_vec()]).await;
    let host = TeamHost::new(host_ep.clone(), manifest, &split_layers(2, 2));
    let _router = Router::builder(host_ep)
        .accept(TEAM_ALPN, host.handler())
        .spawn();
    let mut invitation = host.invitation();
    invitation.secret[0] ^= 1;
    let joined = TeamMember::join(local_endpoint(vec![]).await, &invitation).await;
    assert!(matches!(joined, Err(TeamError::Remote(_))));
}
