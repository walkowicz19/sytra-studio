//! Hugging Face Xet memory caps.
//!
//! Default hf-xet reconstruction buffers are sized for high-RAM machines
//! (2 GiB download buffer, 8 GiB hard limit, 1 GiB prefetch, adaptive
//! download concurrency up to 64). Those reservations page Windows, macOS,
//! and Linux desktops even for tiny GGUF files. Sytra overwrites the
//! defaults on every runner, mergekit, download, convert, and serve child.

/// Environment variables applied to every Python child that may touch Hub/Xet.
pub fn xet_safety_env() -> &'static [(&'static str, &'static str)] {
    &[
        ("HF_XET_HIGH_PERFORMANCE", "0"),
        ("HF_XET_HP", "0"),
        ("HF_HUB_ENABLE_HF_TRANSFER", "0"),
        ("HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY", "false"),
        ("HF_XET_FIXED_DOWNLOAD_CONCURRENCY", "2"),
        ("HF_XET_FIXED_UPLOAD_CONCURRENCY", "1"),
        ("HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS", "1"),
        ("HF_XET_DATA_MAX_CONCURRENT_FILE_INGESTION", "1"),
        ("HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE", "64mb"),
        ("HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE", "32mb"),
        ("HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT", "128mb"),
        ("HF_XET_RECONSTRUCTION_MIN_RECONSTRUCTION_FETCH_SIZE", "16mb"),
        ("HF_XET_RECONSTRUCTION_MAX_RECONSTRUCTION_FETCH_SIZE", "128mb"),
        ("HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER", "16mb"),
        ("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0"),
    ]
}

pub fn apply_xet_safety(cmd: &mut std::process::Command) {
    for (key, value) in xet_safety_env() {
        cmd.env(key, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(key: &str) -> &'static str {
        xet_safety_env()
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| *v)
            .unwrap_or("")
    }

    #[test]
    fn disables_high_performance_and_adaptive_concurrency() {
        assert_eq!(value("HF_XET_HIGH_PERFORMANCE"), "0");
        assert_eq!(value("HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY"), "false");
        assert_eq!(value("HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"), "1");
        assert_eq!(value("HF_HUB_ENABLE_HF_TRANSFER"), "0");
    }

    #[test]
    fn reconstruction_buffers_stay_well_under_one_gigabyte() {
        for key in [
            "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE",
            "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE",
            "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT",
            "HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER",
            "HF_XET_RECONSTRUCTION_MAX_RECONSTRUCTION_FETCH_SIZE",
        ] {
            let raw = value(key).to_ascii_lowercase();
            assert!(
                raw.ends_with("mb"),
                "{key}={raw} must be an explicit megabyte cap"
            );
            let mb: u64 = raw.trim_end_matches("mb").parse().expect(key);
            assert!(mb <= 128, "{key}={raw} must be <= 128mb to avoid OS paging");
        }
    }
}
