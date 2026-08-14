use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use serde_json::Value;

const MAX_TAIL: usize = 200;
const CHUNK: usize = 8192;

/// Read the last `n` non-empty JSONL lines without loading the whole file.
pub fn tail_jsonl(path: &Path, n: usize) -> Vec<Value> {
    let n = n.min(MAX_TAIL);
    if n == 0 {
        return Vec::new();
    }
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let len = match file.metadata() {
        Ok(meta) => meta.len(),
        Err(_) => return Vec::new(),
    };
    if len == 0 {
        return Vec::new();
    }

    let mut pos = len;
    let mut buf = Vec::new();
    let mut found = 0usize;
    while pos > 0 && found <= n {
        let read_size = CHUNK.min(pos as usize) as u64;
        pos -= read_size;
        if file.seek(SeekFrom::Start(pos)).is_err() {
            return Vec::new();
        }
        let mut chunk = vec![0u8; read_size as usize];
        if file.read_exact(&mut chunk).is_err() {
            return Vec::new();
        }
        buf.splice(0..0, chunk);
        found = buf.iter().filter(|b| **b == b'\n').count();
        if pos == 0 {
            break;
        }
    }

    let text = String::from_utf8_lossy(&buf);
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    lines
        .iter()
        .skip(lines.len().saturating_sub(n))
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tails_last_n_json_objects() {
        let dir = std::env::temp_dir().join(format!("sytra-tail-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.jsonl");
        let mut body = String::new();
        for i in 0..50 {
            body.push_str(&format!("{{\"i\":{i}}}\n"));
        }
        std::fs::write(&path, body).unwrap();
        let tail = tail_jsonl(&path, 3);
        assert_eq!(tail.len(), 3);
        assert_eq!(tail[0]["i"], 47);
        assert_eq!(tail[2]["i"], 49);
        std::fs::remove_dir_all(&dir).ok();
    }
}
