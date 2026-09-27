//! rosbag2 sqlite3 storage (`.db3`): `topics` / `messages` / `message_definitions` read through a read-only rusqlite connection; the message id doubles as the index offset.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use rusqlite::{Connection as Db, OpenFlags};

use super::reader::{BagError, Connection, IndexEntry, RawMessage};
use super::storage::{Storage, is_cancelled};
use crate::tf::buffer::TimeNs;

/// Rows between two looks at the cancel flag while indexing.
const CANCEL_CHECK_ROWS: usize = 4096;

/// Reads a rosbag2 sqlite3 database; one row of `messages` is one message, fetched by primary key.
pub struct Sqlite3Reader {
    db: Db,
    path: PathBuf,
    size_bytes: u64,
    connections: Vec<Connection>,
    /// Rows counted by `read_message_index`; None until it ran.
    indexed: Option<u64>,
    start: TimeNs,
    /// Payload of the last `message_at`, lent out as the returned slice.
    last: Vec<u8>,
    warnings: Vec<String>,
}

/// `OpenStorage` for `.db3` files; the chunk cache size does not apply (sqlite pages are cached by SQLite itself).
pub fn open_storage(
    path: &Path,
    _cache: usize,
    cancel: &AtomicBool,
) -> Result<Box<dyn Storage>, BagError> {
    if is_cancelled(cancel) {
        return Err(BagError::Cancelled);
    }
    Ok(Box::new(Sqlite3Reader::open(path)?))
}

fn sql_error(e: rusqlite::Error) -> BagError {
    BagError::Malformed(format!("sqlite3: {e}"))
}

impl Sqlite3Reader {
    /// Open read-only and read the topic table (plus `message_definitions` when the schema has it).
    pub fn open(path: &Path) -> Result<Self, BagError> {
        let size_bytes = std::fs::metadata(path)?.len();
        let db = Db::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(|e| BagError::Io(format!("{}: {e}", path.display())))?;
        let tables = table_names(&db)?;
        if !tables.iter().any(|t| t == "topics") || !tables.iter().any(|t| t == "messages") {
            return Err(BagError::Malformed(format!(
                "`{}` has no rosbag2 `topics` / `messages` tables",
                path.display()
            )));
        }
        let mut warnings = Vec::new();
        let definitions = if tables.iter().any(|t| t == "message_definitions") {
            read_definitions(&db)?
        } else {
            warnings.push(
                "no `message_definitions` table (recorded before Iron); bundled definitions are used".to_owned(),
            );
            HashMap::new()
        };
        let has_hash = column_names(&db, "topics")?
            .iter()
            .any(|c| c == "type_description_hash");
        let connections = read_topics(&db, has_hash, &definitions)?;
        // MIN(timestamp) is answered from rosbag2's `timestamp_idx`, so open stays O(log n); the message count comes from the (cancellable) index read instead of a COUNT(*) scan that could not be interrupted.
        let start: Option<i64> = db
            .query_row("SELECT MIN(timestamp) FROM messages", [], |row| row.get(0))
            .map_err(sql_error)?;
        Ok(Self {
            db,
            path: path.to_path_buf(),
            size_bytes,
            connections,
            indexed: None,
            start: start.unwrap_or(0),
            last: Vec::new(),
            warnings,
        })
    }

    /// One entry per `messages` row; the row id becomes `offset`, and there are no chunks.
    pub fn read_message_index(&mut self, cancel: &AtomicBool) -> Result<Vec<IndexEntry>, BagError> {
        let mut statement = self
            .db
            .prepare("SELECT id, topic_id, timestamp FROM messages ORDER BY id")
            .map_err(sql_error)?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })
            .map_err(sql_error)?;
        let mut entries = Vec::new();
        for (n, row) in rows.enumerate() {
            if n % CANCEL_CHECK_ROWS == 0 && is_cancelled(cancel) {
                return Err(BagError::Cancelled);
            }
            let (id, topic_id, timestamp) = row.map_err(sql_error)?;
            let offset = u32::try_from(id).map_err(|_| {
                BagError::Malformed(format!(
                    "message id {id} does not fit the 32-bit index offset"
                ))
            })?;
            let conn = u32::try_from(topic_id)
                .map_err(|_| BagError::Malformed(format!("topic id {topic_id} is negative")))?;
            entries.push(IndexEntry {
                time: timestamp,
                conn,
                file: 0,
                chunk: 0,
                offset,
            });
        }
        self.indexed = Some(entries.len() as u64);
        Ok(entries)
    }

    /// Fetch one row by id; the payload is kept in the reader and lent to the caller.
    pub fn message_at(&mut self, entry: &IndexEntry) -> Result<RawMessage<'_>, BagError> {
        let (topic_id, timestamp, data): (i64, i64, Vec<u8>) = self
            .db
            .prepare_cached("SELECT topic_id, timestamp, data FROM messages WHERE id = ?1")
            .map_err(sql_error)?
            .query_row([i64::from(entry.offset)], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    BagError::Malformed(format!("no message with id {}", entry.offset))
                }
                other => sql_error(other),
            })?;
        self.last = data;
        Ok(RawMessage {
            conn: topic_id.max(0) as u32,
            time: timestamp,
            data: &self.last,
        })
    }
}

impl Storage for Sqlite3Reader {
    fn path(&self) -> &Path {
        &self.path
    }

    fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    fn connections(&self) -> &[Connection] {
        &self.connections
    }

    fn chunk_count(&self) -> usize {
        0
    }

    fn compression(&self) -> Option<&str> {
        Some("none")
    }

    fn message_count(&self) -> u64 {
        self.indexed.unwrap_or(0)
    }

    fn start(&self) -> TimeNs {
        self.start
    }

    fn read_message_index(&mut self, cancel: &AtomicBool) -> Result<Vec<IndexEntry>, BagError> {
        Sqlite3Reader::read_message_index(self, cancel)
    }

    fn message_at(&mut self, entry: &IndexEntry) -> Result<RawMessage<'_>, BagError> {
        Sqlite3Reader::message_at(self, entry)
    }

    fn warnings(&self) -> &[String] {
        &self.warnings
    }
}

fn table_names(db: &Db) -> Result<Vec<String>, BagError> {
    let mut statement = db
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table'")
        .map_err(sql_error)?;
    let names = statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(sql_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(sql_error)?;
    Ok(names)
}

fn column_names(db: &Db, table: &str) -> Result<Vec<String>, BagError> {
    let mut statement = db
        .prepare("SELECT name FROM pragma_table_info(?1)")
        .map_err(sql_error)?;
    let names = statement
        .query_map([table], |row| row.get::<_, String>(0))
        .map_err(sql_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(sql_error)?;
    Ok(names)
}

/// `topic_type -> (encoding, definition text)` from the Iron-and-later `message_definitions` table.
fn read_definitions(db: &Db) -> Result<HashMap<String, (String, String)>, BagError> {
    let mut statement = db
        .prepare("SELECT topic_type, encoding, encoded_message_definition FROM message_definitions")
        .map_err(sql_error)?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .map_err(sql_error)?;
    let mut out = HashMap::new();
    for row in rows {
        let (topic_type, encoding, text) = row.map_err(sql_error)?;
        out.insert(topic_type, (encoding, text));
    }
    Ok(out)
}

fn read_topics(
    db: &Db,
    has_hash: bool,
    definitions: &HashMap<String, (String, String)>,
) -> Result<Vec<Connection>, BagError> {
    let sql = if has_hash {
        "SELECT id, name, type, serialization_format, type_description_hash FROM topics ORDER BY id"
    } else {
        "SELECT id, name, type, serialization_format, '' FROM topics ORDER BY id"
    };
    let mut statement = db.prepare(sql).map_err(sql_error)?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<String>>(4)?,
            ))
        })
        .map_err(sql_error)?;
    let mut connections = Vec::new();
    for row in rows {
        let (id, name, type_raw, serialization_format, type_hash) = row.map_err(sql_error)?;
        let id = u32::try_from(id)
            .map_err(|_| BagError::Malformed(format!("topic id {id} is negative")))?;
        let (definition_encoding, definition) =
            definitions.get(&type_raw).cloned().unwrap_or_default();
        connections.push(Connection {
            id,
            topic_raw: name,
            type_raw,
            type_hash: type_hash.unwrap_or_default(),
            definition,
            message_encoding: serialization_format,
            definition_encoding,
        });
    }
    Ok(connections)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rosbag2 Jazzy schema (`schema_version` 4), filled with two topics and three messages.
    fn write_db(name: &str, with_definitions: bool) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "visor_sqlite3_test_{}_{name}.db3",
            std::process::id()
        ));
        std::fs::remove_file(&path).ok();
        let db = Db::open(&path).unwrap();
        db.execute_batch(
            "CREATE TABLE schema(schema_version INTEGER PRIMARY KEY, ros_distro TEXT NOT NULL);\
             INSERT INTO schema VALUES (4, 'jazzy');\
             CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL, serialization_format TEXT NOT NULL, offered_qos_profiles TEXT NOT NULL, type_description_hash TEXT NOT NULL);\
             CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL, timestamp INTEGER NOT NULL, data BLOB NOT NULL);\
             INSERT INTO topics VALUES (1, '/chatter', 'std_msgs/msg/String', 'cdr', '', 'RIHS01_a');\
             INSERT INTO topics VALUES (2, '/tf', 'tf2_msgs/msg/TFMessage', 'cdr', '', 'RIHS01_b');\
             INSERT INTO messages VALUES (1, 1, 100, X'01');\
             INSERT INTO messages VALUES (2, 2, 101, X'0202');\
             INSERT INTO messages VALUES (3, 1, 102, X'030303');",
        )
        .unwrap();
        if with_definitions {
            db.execute_batch(
                "CREATE TABLE message_definitions(id INTEGER PRIMARY KEY, topic_type TEXT NOT NULL, encoding TEXT NOT NULL, encoded_message_definition TEXT NOT NULL, type_description_hash TEXT NOT NULL);\
                 INSERT INTO message_definitions VALUES (1, 'std_msgs/msg/String', 'ros2msg', 'string data', 'RIHS01_a');",
            )
            .unwrap();
        }
        path
    }

    #[test]
    fn topics_definitions_and_messages_come_through() {
        let path = write_db("full", true);
        let mut reader = Sqlite3Reader::open(&path).unwrap();
        assert!(reader.warnings.is_empty());
        // Opening does not scan the table: the count is known once the index has been read.
        assert_eq!(reader.message_count(), 0);
        assert_eq!(reader.start(), 100);
        let conns = reader.connections();
        assert_eq!(conns.len(), 2);
        assert_eq!(conns[0].id, 1);
        assert_eq!(conns[0].topic_raw, "/chatter");
        assert_eq!(conns[0].type_raw, "std_msgs/msg/String");
        assert_eq!(conns[0].type_hash, "RIHS01_a");
        assert_eq!(conns[0].message_encoding, "cdr");
        assert_eq!(conns[0].definition_encoding, "ros2msg");
        assert_eq!(conns[0].definition, "string data");
        // A type with no row in message_definitions carries an empty definition, which the registry treats as "use the fallback".
        assert_eq!(conns[1].definition_encoding, "");
        assert_eq!(conns[1].definition, "");
        let index = reader.read_message_index(&AtomicBool::new(false)).unwrap();
        assert_eq!(index.len(), 3);
        assert_eq!(reader.message_count(), 3);
        assert_eq!(
            index[1],
            IndexEntry {
                time: 101,
                conn: 2,
                file: 0,
                chunk: 0,
                offset: 2
            }
        );
        let message = reader.message_at(&index[2]).unwrap();
        assert_eq!(message.conn, 1);
        assert_eq!(message.time, 102);
        assert_eq!(message.data, &[3, 3, 3]);
        assert!(matches!(
            reader.message_at(&IndexEntry {
                offset: 99,
                ..index[0]
            }),
            Err(BagError::Malformed(_))
        ));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_pre_iron_database_opens_with_a_warning_and_no_definitions() {
        let path = write_db("old", false);
        let reader = Sqlite3Reader::open(&path).unwrap();
        assert_eq!(reader.warnings().len(), 1);
        assert!(reader.warnings()[0].contains("message_definitions"));
        assert!(reader.connections().iter().all(|c| c.definition.is_empty()));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_database_without_the_rosbag2_tables_and_a_cancel_flag_are_both_refused() {
        let path = std::env::temp_dir().join(format!(
            "visor_sqlite3_test_{}_empty.db3",
            std::process::id()
        ));
        std::fs::remove_file(&path).ok();
        Db::open(&path)
            .unwrap()
            .execute_batch("CREATE TABLE other(x INTEGER);")
            .unwrap();
        assert!(matches!(
            Sqlite3Reader::open(&path),
            Err(BagError::Malformed(_))
        ));
        std::fs::remove_file(&path).ok();
        let path = write_db("cancel", true);
        let cancelled = AtomicBool::new(true);
        assert!(matches!(
            open_storage(&path, 1, &cancelled),
            Err(BagError::Cancelled)
        ));
        let mut reader = Sqlite3Reader::open(&path).unwrap();
        assert!(matches!(
            reader.read_message_index(&cancelled),
            Err(BagError::Cancelled)
        ));
        std::fs::remove_file(&path).ok();
    }
}
