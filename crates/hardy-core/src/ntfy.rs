//! Minimal ntfy client: publish notifications and poll a topic for messages.

use std::{fmt, time::Duration};

use chrono::{DateTime, Utc};
use futures::future::BoxFuture;
use serde::Deserialize;

use crate::{
    config::NetworkConfig,
    error::{AppError, NetworkErrorKind},
    traits::Notifier,
};

/// A message received from a topic.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct NtfyMessage {
    pub id: String,
    pub time: i64,
    #[serde(default)]
    pub message: String,
}

/// Where a poll starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PollSince {
    /// Messages after the message with this id.
    Id(String),
    /// Messages published after this instant.
    Time(DateTime<Utc>),
}

impl PollSince {
    fn query_value(&self) -> String {
        match self {
            PollSince::Id(id) => id.clone(),
            PollSince::Time(t) => t.timestamp().to_string(),
        }
    }
}

#[derive(Deserialize)]
struct Event {
    event: String,
    #[serde(flatten)]
    message: Option<NtfyMessage>,
}

/// HTTP client for one ntfy server, optionally authenticated with an access
/// token.
#[derive(Clone)]
pub struct NtfyClient {
    http: reqwest::Client,
    server: String,
    token: Option<String>,
}

impl fmt::Debug for NtfyClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NtfyClient")
            .field("server", &self.server)
            .field("token", &self.token.as_ref().map(|_| "<redacted>"))
            .finish_non_exhaustive()
    }
}

impl NtfyClient {
    pub fn new(
        server: &str,
        token: Option<String>,
        network: &NetworkConfig,
    ) -> Result<Self, AppError> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(network.request_timeout_secs))
            .connect_timeout(Duration::from_secs(network.connect_timeout_secs))
            .build()
            .map_err(|e| AppError::Network {
                message: format!("Failed to create ntfy HTTP client: {e}"),
                kind: NetworkErrorKind::Unknown,
            })?;
        Ok(Self {
            http,
            server: server.trim_end_matches('/').to_string(),
            token: token.filter(|t| !t.is_empty()),
        })
    }

    fn authorized(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    /// Publishes a notification to `topic`.
    #[tracing::instrument(skip(self, body), fields(server = %self.server))]
    pub async fn publish(&self, topic: &str, title: &str, body: &str) -> Result<(), AppError> {
        let request = self
            .http
            .post(format!("{}/{topic}", self.server))
            .header("Title", title)
            .body(body.to_string());
        let response = self
            .authorized(request)
            .send()
            .await
            .map_err(AppError::from_reqwest)?;
        check_status(&response)
    }

    /// Returns the messages on `topic` published after `since`, oldest
    /// first.
    #[tracing::instrument(skip(self), fields(server = %self.server))]
    pub async fn poll(&self, topic: &str, since: &PollSince) -> Result<Vec<NtfyMessage>, AppError> {
        // SAFETY: message ids are alphanumeric and timestamps numeric, so the
        // value needs no URL encoding.
        let request = self.http.get(format!(
            "{}/{topic}/json?poll=1&since={}",
            self.server,
            since.query_value()
        ));
        let response = self
            .authorized(request)
            .send()
            .await
            .map_err(AppError::from_reqwest)?;
        check_status(&response)?;
        let text = response.text().await.map_err(AppError::from_reqwest)?;

        text.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                serde_json::from_str::<Event>(line)
                    .map_err(|e| AppError::validation(format!("invalid ntfy event {line:?}: {e}")))
            })
            .filter_map(|event| match event {
                Ok(Event {
                    event,
                    message: Some(message),
                }) if event == "message" => Some(Ok(message)),
                Ok(_) => None,
                Err(e) => Some(Err(e)),
            })
            .collect()
    }
}

fn check_status(response: &reqwest::Response) -> Result<(), AppError> {
    let status = response.status();
    if status.is_success() {
        Ok(())
    } else {
        Err(AppError::api_error(
            status.as_u16(),
            format!("ntfy returned {status}"),
        ))
    }
}

/// [`Notifier`] that publishes to one ntfy topic.
#[derive(Debug, Clone)]
pub struct NtfyNotifier {
    client: NtfyClient,
    topic: String,
}

impl NtfyNotifier {
    pub fn new(client: NtfyClient, topic: String) -> Self {
        Self { client, topic }
    }
}

impl Notifier for NtfyNotifier {
    fn notify<'s>(&'s self, title: &str, body: &str) -> BoxFuture<'s, anyhow::Result<()>> {
        let title = title.to_string();
        let body = body.to_string();
        Box::pin(async move {
            self.client
                .publish(&self.topic, &title, &body)
                .await
                .map_err(anyhow::Error::from)
        })
    }
}
