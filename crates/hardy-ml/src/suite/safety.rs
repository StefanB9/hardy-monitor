//! Keeps one misbehaving model from taking down the rest: a panic inside a
//! model (ours or a crate's) becomes an error for that model only.

use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::error::MlError;

/// Runs `f`, turning a panic into [`MlError::Panicked`].
pub fn guarded<T>(f: impl FnOnce() -> Result<T, MlError>) -> Result<T, MlError> {
    // SAFETY (unwind): a model that panics is discarded for the night by the
    // caller, so state it may have left half-updated is never read again.
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|payload| {
        let message = payload
            .downcast_ref::<&str>()
            .map(ToString::to_string)
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown panic".to_string());
        Err(MlError::Panicked(message))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guarded_passes_results_through() {
        assert_eq!(guarded(|| Ok(3)), Ok(3));
        assert_eq!(
            guarded::<()>(|| Err(MlError::Fit("x".to_string()))),
            Err(MlError::Fit("x".to_string()))
        );
    }

    #[test]
    fn test_guarded_turns_a_panic_into_an_error() {
        let result = guarded(|| {
            let empty: Vec<f64> = std::hint::black_box(Vec::new());
            Ok(empty[std::hint::black_box(3)])
        });
        assert!(
            matches!(&result, Err(MlError::Panicked(message)) if message.contains("index out of bounds")),
            "{result:?}"
        );
    }
}
