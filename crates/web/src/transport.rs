//! Endpoint setup for browser nodes.

use iroh::SecretKey;
use peerinfer_net::bind_endpoint;
use wasm_bindgen::prelude::*;

use crate::js_err;

/// Optional custom relay is for explicit local testing; normal deployment uses n0.
pub(crate) async fn bind(
    key: SecretKey,
    alpns: Vec<Vec<u8>>,
    relay: Option<String>,
) -> Result<iroh::Endpoint, JsError> {
    if let Some(url) = relay {
        let url: iroh::RelayUrl = url.parse().map_err(js_err)?;
        iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
            .secret_key(key)
            .relay_mode(iroh::RelayMode::Custom(iroh::RelayMap::from(url)))
            .alpns(alpns)
            .bind()
            .await
            .map_err(js_err)
    } else {
        bind_endpoint(key, alpns).await.map_err(js_err)
    }
}
