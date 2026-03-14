use std::time::Duration;

use anyhow::Result;
use futures::future::BoxFuture;
use hardy_core::traits::Notifier;
use tracing::warn;

#[derive(Debug, Clone, Default)]
pub struct SystemNotifier;

impl Notifier for SystemNotifier {
    fn notify<'s>(&'s self, title: &str, body: &str) -> BoxFuture<'s, Result<()>> {
        let title = title.to_string();
        let body = body.to_string();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                notify_rust::Notification::new()
                    .summary(&title)
                    .body(&body)
                    .appname("Hardy Monitor")
                    .show()
            })
            .await
            .map_err(|e| anyhow::anyhow!("desktop notification task panicked: {e}"))??;
            Ok(())
        })
    }
}

#[derive(Debug, Clone)]
pub struct CombinedNotifier {
    ntfy_topic: Option<String>,
    ntfy_server: String,
    http_client: reqwest::Client,
}

impl CombinedNotifier {
    pub fn new(ntfy_topic: Option<String>, ntfy_server: String) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_else(|e| {
                warn!(error = %e, "failed to build HTTP client with custom timeout; using default");
                reqwest::Client::new()
            });
        Self {
            ntfy_topic,
            ntfy_server,
            http_client,
        }
    }
}

impl Notifier for CombinedNotifier {
    fn notify<'s>(&'s self, title: &str, body: &str) -> BoxFuture<'s, Result<()>> {
        let title = title.to_string();
        let body = body.to_string();
        Box::pin(async move {
            let t = title.clone();
            let b = body.clone();
            match tokio::task::spawn_blocking(move || {
                notify_rust::Notification::new()
                    .summary(&t)
                    .body(&b)
                    .appname("Hardy Monitor")
                    .show()
            })
            .await
            {
                Err(e) => warn!(error = %e, "desktop notification task panicked"),
                Ok(Err(e)) => warn!(error = %e, "desktop notification failed"),
                Ok(Ok(())) => {}
            }

            if let Some(ref topic) = self.ntfy_topic {
                let url = format!("{}/{topic}", self.ntfy_server);
                let message = format!("{title}\n{body}");
                match self.http_client.post(&url).body(message).send().await {
                    Err(e) => warn!(error = %e, %url, "ntfy send failed"),
                    Ok(resp) if !resp.status().is_success() => {
                        warn!(status = %resp.status(), %url, "ntfy returned error status");
                    }
                    Ok(_) => {}
                }
            }

            Ok(())
        })
    }
}
