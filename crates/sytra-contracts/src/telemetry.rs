use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// One line of the shared train/merge telemetry protocol (Contract 3).
/// `parse_line` never fails: anything that isn't a recognized JSON shape
/// becomes a `Log` line, so a stray `print()` in the runner can't crash
/// the host's parser.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TelemetryLine {
    Event {
        ts: f64,
        op_id: Option<String>,
        event: String,
        payload: Value,
    },
    Metric {
        ts: f64,
        #[serde(flatten)]
        fields: Map<String, Value>,
    },
    Log {
        ts: Option<f64>,
        stream: Option<String>,
        line: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RawLine {
    Event {
        ts: f64,
        #[serde(default)]
        op_id: Option<String>,
        event: String,
        #[serde(default)]
        payload: Value,
    },
    Metric {
        ts: f64,
        #[serde(flatten)]
        fields: Map<String, Value>,
    },
    Log {
        #[serde(default)]
        ts: Option<f64>,
        #[serde(default)]
        stream: Option<String>,
        line: String,
    },
}

pub fn parse_line(raw: &str) -> TelemetryLine {
    match serde_json::from_str::<RawLine>(raw) {
        Ok(RawLine::Event {
            ts,
            op_id,
            event,
            payload,
        }) => TelemetryLine::Event {
            ts,
            op_id,
            event,
            payload,
        },
        Ok(RawLine::Metric { ts, mut fields }) => {
            fields.remove("type");
            TelemetryLine::Metric { ts, fields }
        }
        Ok(RawLine::Log { ts, stream, line }) => TelemetryLine::Log { ts, stream, line },
        Err(_) => TelemetryLine::Log {
            ts: None,
            stream: None,
            line: raw.to_string(),
        },
    }
}

/// Closed event vocabulary shared by train and merge (Contract 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventName {
    Starting,
    Epoch,
    Checkpoint,
    Eval,
    Stage,
    Done,
    Error,
}

impl EventName {
    pub fn is_terminal(&self) -> bool {
        matches!(self, EventName::Done | EventName::Error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_event_line() {
        let line = r#"{"type":"event","event":"starting","ts":1719400000.12,"op_id":"0190","payload":{"protocol_version":1}}"#;
        match parse_line(line) {
            TelemetryLine::Event { event, op_id, .. } => {
                assert_eq!(event, "starting");
                assert_eq!(op_id, Some("0190".to_string()));
            }
            other => panic!("expected Event, got {other:?}"),
        }
    }

    #[test]
    fn parses_valid_train_metric_line() {
        let line = r#"{"type":"metric","ts":1719400003.5,"step":42,"loss":1.231,"lr":1.93e-4}"#;
        match parse_line(line) {
            TelemetryLine::Metric { fields, .. } => {
                assert_eq!(fields.get("step").unwrap(), 42);
                assert!(fields.get("loss").is_some());
            }
            other => panic!("expected Metric, got {other:?}"),
        }
    }

    #[test]
    fn parses_valid_merge_metric_line_with_progress() {
        let line = r#"{"type":"metric","ts":1.0,"progress":0.35,"stage":"computing_task_vectors","mem_used_mb":9200}"#;
        match parse_line(line) {
            TelemetryLine::Metric { fields, .. } => {
                assert_eq!(fields.get("progress").unwrap(), 0.35);
            }
            other => panic!("expected Metric, got {other:?}"),
        }
    }

    #[test]
    fn unparseable_garbage_falls_back_to_log() {
        let line = "this is not json at all";
        match parse_line(line) {
            TelemetryLine::Log { line: l, .. } => assert_eq!(l, line),
            other => panic!("expected Log fallback, got {other:?}"),
        }
    }

    #[test]
    fn truncated_json_falls_back_to_log() {
        let line = r#"{"type":"event","event":"starting","ts":1.0"#;
        match parse_line(line) {
            TelemetryLine::Log { line: l, .. } => assert_eq!(l, line),
            other => panic!("expected Log fallback, got {other:?}"),
        }
    }

    #[test]
    fn unknown_type_tag_falls_back_to_log() {
        let line = r#"{"type":"mystery","foo":"bar"}"#;
        match parse_line(line) {
            TelemetryLine::Log { line: l, .. } => assert_eq!(l, line),
            other => panic!("expected Log fallback, got {other:?}"),
        }
    }

    #[test]
    fn explicit_log_line_passes_through() {
        let line = r#"{"type":"log","ts":1.0,"stream":"stderr","line":"warning: deprecated"}"#;
        match parse_line(line) {
            TelemetryLine::Log {
                stream, line: l, ..
            } => {
                assert_eq!(stream, Some("stderr".to_string()));
                assert_eq!(l, "warning: deprecated");
            }
            other => panic!("expected Log, got {other:?}"),
        }
    }

    #[test]
    fn unknown_metric_keys_are_forwarded_untouched() {
        let line = r#"{"type":"metric","ts":1.0,"reward":0.9,"kl":0.02}"#;
        match parse_line(line) {
            TelemetryLine::Metric { fields, .. } => {
                assert_eq!(fields.get("reward").unwrap(), 0.9);
                assert_eq!(fields.get("kl").unwrap(), 0.02);
            }
            other => panic!("expected Metric, got {other:?}"),
        }
    }

    #[test]
    fn done_and_error_events_are_terminal() {
        assert!(EventName::Done.is_terminal());
        assert!(EventName::Error.is_terminal());
        assert!(!EventName::Stage.is_terminal());
    }
}
