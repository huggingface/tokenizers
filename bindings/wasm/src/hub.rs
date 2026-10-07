//! Utils to download files from the HuggingFace Hub

use js_sys::{Function, Promise};
use serde::Deserialize;
use tsify::{Ts, Tsify};
use wasm_bindgen::prelude::*;
use web_sys::{Headers, RequestInit, Response};

const HUB_URL: &str = "https://huggingface.co";

/// Fetches the content of tokenizers.json from the Hub for the given repository
pub(crate) async fn fetch_tokenizer_json(
    repo_name: &str,
    options: Option<Ts<PretrainedOptions>>,
) -> Result<String, JsValue> {
    let options = options
        .map(|options| options.to_rust())
        .transpose()
        .map_err(JsError::from)?
        .unwrap_or_default();

    let url = options.make_url(repo_name);
    let headers = options.make_headers()?;

    let response = fetch_url(&url, &headers, options.fetch.as_ref())?.await?;
    if !response.ok() {
        return Err(JsError::new(&format!("{} fetching {url}", response.status())).into());
    }
    Ok(response.text()?.await?.as_string().unwrap())
}

#[derive(Tsify, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct PretrainedOptions {
    /// The git revision to download the file from: commit ID, branch name or tag name.
    /// Defaults to `"main"` when omitted.
    #[tsify(optional)]
    revision: Option<String>,
    /// An optional HF access token (looks like `hf_xxxx`) to authenticate the download call.
    /// Required to download files from private or gated repositories.
    #[tsify(optional)]
    token: Option<String>,
    /// The folder of the repo that holds `tokenizer.json`.
    /// Defaults to the repo root when omitted.
    #[tsify(optional)]
    subfolder: Option<String>,
    /// The URL to download from.
    /// Defaults to `"https://huggingface.co"` when omitted.
    #[tsify(optional)]
    hub_url: Option<String>,
    /// An optional override of the `fetch` function.
    /// Defaults to the global `fetch` when omitted.
    #[tsify(optional, type = "typeof globalThis.fetch")]
    #[serde(default, deserialize_with = "deserialize_function")]
    fetch: Option<Function>,
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = fetch)]
    fn global_fetch(url: &str, init: &RequestInit) -> Promise<Response>;
}

impl PretrainedOptions {
    fn make_url(&self, identifier: &str) -> String {
        let hub_url = self
            .hub_url
            .as_deref()
            .unwrap_or(HUB_URL)
            .trim_end_matches('/');
        // Need to escape slashes (`/`) for the hub to resolve a ref like refs/pr/1
        let revision = String::from(js_sys::encode_uri_component(
            self.revision.as_deref().unwrap_or("main"),
        ));
        let file = match &self.subfolder {
            Some(subfolder) => format!("{subfolder}/tokenizer.json"),
            None => "tokenizer.json".to_owned(),
        };
        format!("{hub_url}/{identifier}/resolve/{revision}/{file}")
    }

    fn make_headers(&self) -> Result<Headers, JsValue> {
        let headers = Headers::new()?;
        if let Some(token) = &self.token {
            headers.set("Authorization", &format!("Bearer {token}"))?;
        }
        Ok(headers)
    }
}

/// Fetch the URL, return the Response object
fn fetch_url(
    url: &str,
    headers: &Headers,
    fetch: Option<&Function>,
) -> Result<Promise<Response>, JsValue> {
    let init = RequestInit::new();
    init.set_headers(headers);
    let Some(fetch) = fetch else {
        return Ok(global_fetch(url, &init));
    };
    Ok(fetch
        .call2(&JsValue::NULL, &url.into(), &init)?
        .unchecked_into())
}

// serde has no `Deserialize` for a JS function. `preserve` passes the value through unchanged.
fn deserialize_function<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Function>, D::Error> {
    serde_wasm_bindgen::preserve::deserialize(deserializer)
        .map(Some)
        .map_err(|_| serde::de::Error::custom("fetch must be a function"))
}
