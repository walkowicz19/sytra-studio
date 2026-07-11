use anyhow::{anyhow, Result};
use std::fs;
use std::path::{Path, PathBuf};
use sytra_contracts::OpRecord;
use uuid::Uuid;

pub struct RunArchive {
    archive_dir: PathBuf,
}

impl RunArchive {
    pub fn new(app_data_dir: &Path) -> Self {
        Self {
            archive_dir: app_data_dir.join("runs"),
        }
    }

    fn file_path(&self, op_id: Uuid) -> PathBuf {
        self.archive_dir.join(format!("{}.json", op_id))
    }

    /// Stores or updates an operation record in the file-based runs archive.
    pub fn store(&self, record: &OpRecord) -> Result<()> {
        fs::create_dir_all(&self.archive_dir)?;
        let path = self.file_path(record.op_id);
        let json_str = serde_json::to_string_pretty(record)?;
        fs::write(path, json_str)?;
        Ok(())
    }

    /// Loads an operation record by operation ID.
    pub fn load(&self, op_id: Uuid) -> Result<OpRecord> {
        let path = self.file_path(op_id);
        if !path.exists() {
            return Err(anyhow!("Run record not found for ID: {op_id}"));
        }
        let json_str = fs::read_to_string(path)?;
        let record: OpRecord = serde_json::from_str(&json_str)?;
        Ok(record)
    }

    /// Lists all run records in the archive.
    pub fn list(&self) -> Result<Vec<OpRecord>> {
        fs::create_dir_all(&self.archive_dir)?;
        let mut records = Vec::new();

        for entry in fs::read_dir(&self.archive_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("json") {
                let json_str = fs::read_to_string(&path)?;
                if let Ok(record) = serde_json::from_str::<OpRecord>(&json_str) {
                    records.push(record);
                }
            }
        }

        // Sort runs (could be by timestamp if timestamp was in OpRecord, but we can sort by op_id/stable identifier for now)
        records.sort_by(|a, b| a.op_id.cmp(&b.op_id));
        Ok(records)
    }

    /// Deletes a run record from the archive.
    pub fn delete(&self, op_id: Uuid) -> Result<()> {
        let path = self.file_path(op_id);
        if path.exists() {
            fs::remove_file(path)?;
        }
        Ok(())
    }
}
