use anyhow::Result;
use futures::future::BoxFuture;
use hardy_core::traits::Notifier;

/// [`Notifier`] showing desktop notifications.
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
