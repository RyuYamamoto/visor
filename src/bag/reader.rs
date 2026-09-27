//! ROS 1 bag v2.0 reading: records, header fields, index section, chunk decompression (`none` / `lz4`) with an LRU cache.

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use super::storage::{ChunkCache, Storage, is_cancelled};
use crate::tf::buffer::TimeNs;

/// MCAP's well-known names for what a ROS 1 bag carries: the payload encoding and the `.msg` text encoding.
pub const ROS1_MESSAGE_ENCODING: &str = "ros1";
pub const ROS1_DEFINITION_ENCODING: &str = "ros1msg";

/// First line of every v2.0 bag; anything else is rejected outright.
const MAGIC: &[u8] = b"#ROSBAG V2.0\n";

/// Record op codes (ROS wiki "Bag format 2.0").
const OP_MSG_DATA: u8 = 0x02;
const OP_BAG_HEADER: u8 = 0x03;
const OP_INDEX_DATA: u8 = 0x04;
const OP_CHUNK: u8 = 0x05;
const OP_CHUNK_INFO: u8 = 0x06;
const OP_CONNECTION: u8 = 0x07;

/// One index_data entry is time(2x u32) + offset(u32); verified against all four sample bags.
const INDEX_ENTRY_LEN: usize = 12;

/// Decompressed chunks kept resident across a whole set; split between files so many bags cannot multiply memory.
pub const CHUNK_CACHE_TOTAL: usize = 8;

/// Guard against a corrupt length prefix demanding a huge allocation (records are at most a few MB in practice).
const MAX_RECORD_LEN: usize = 512 * 1024 * 1024;

/// Why a bag could not be read.
#[derive(Debug)]
pub enum BagError {
    Io(String),
    /// First line is not `#ROSBAG V2.0` (v1.x bags and non-bags land here).
    UnsupportedVersion(String),
    /// `index_pos` is 0 or unreadable: the bag was still being recorded (`.bag.active`) or the writer crashed (Q10 = A).
    MissingIndex,
    /// Chunk compression other than `none` / `lz4` (notably `bz2`, absent from the corpus; Q3 = A).
    UnsupportedCompression(String),
    /// Record header lacks a field the op requires.
    MissingField {
        op: u8,
        field: &'static str,
    },
    /// Structurally broken record (bad length, wrong op where one is mandated, field with no `=`).
    Malformed(String),
    /// lz4 / zstd frame decode failed, or the result length disagrees with the chunk header's size.
    Decompress(String),
    /// The owner dropped the source while the file was being opened or indexed; nothing to report to the user.
    Cancelled,
    /// A recognized format or layout that this build does not handle (unchunked MCAP, rosbag2 file compression, mixed storage).
    Unsupported(String),
}

impl fmt::Display for BagError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BagError::Io(e) => write!(f, "{e}"),
            BagError::UnsupportedVersion(v) => {
                write!(
                    f,
                    "not a ROS 1 bag v2.0 or an MCAP file (first bytes: {v:?})"
                )
            }
            BagError::MissingIndex => write!(
                f,
                "bag has no index (still recording, or the writer crashed); `rosbag reindex` it first"
            ),
            BagError::UnsupportedCompression(c) => write!(f, "unsupported chunk compression `{c}`"),
            BagError::MissingField { op, field } => {
                write!(f, "record op 0x{op:02x} is missing field `{field}`")
            }
            BagError::Malformed(m) => write!(f, "malformed bag: {m}"),
            BagError::Decompress(m) => write!(f, "chunk decompression failed: {m}"),
            BagError::Cancelled => write!(f, "cancelled"),
            BagError::Unsupported(m) => write!(f, "unsupported: {m}"),
        }
    }
}

impl std::error::Error for BagError {}

impl From<std::io::Error> for BagError {
    fn from(e: std::io::Error) -> Self {
        BagError::Io(e.to_string())
    }
}

/// Bag header record (op=0x03), which always sits right after the magic line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BagHeader {
    pub index_pos: u64,
    pub conn_count: u32,
    pub chunk_count: u32,
}

/// One recorded connection (a ROS 1 connection record, an MCAP channel, a sqlite3 topic row): topic, type, and self-describing definition.
#[derive(Debug, Clone)]
pub struct Connection {
    pub id: u32,
    /// Topic exactly as recorded, which may lack the leading `/` (normalized by `naming`).
    pub topic_raw: String,
    /// Type name as recorded: ROS 1 `pkg/Type`, or ROS 2 `pkg/msg/Type`.
    pub type_raw: String,
    /// ROS 1 md5sum, or rosbag2's `type_description_hash` (`RIHS01_…`); empty when the file has none.
    pub type_hash: String,
    /// Concatenated message definition: this type plus every dependency (requirements §2.1-d); empty when the file has none.
    pub definition: String,
    /// Payload encoding in MCAP's words: `ros1` or `cdr`.
    pub message_encoding: String,
    /// Encoding of `definition` in MCAP's words: `ros1msg`, `ros2msg`, `ros2idl`, or empty when there is no definition.
    pub definition_encoding: String,
}

/// One chunk_info record (op=0x06): where a chunk is and which connections it holds.
#[derive(Debug, Clone)]
pub struct ChunkInfo {
    /// File offset of the chunk record (op=0x05).
    pub pos: u64,
    pub start: TimeNs,
    pub end: TimeNs,
    /// Per-connection message counts inside this chunk (the record's `count` field is the connection count, not the message count).
    pub conn_counts: Vec<(u32, u32)>,
}

impl ChunkInfo {
    /// Total messages in this chunk.
    pub fn message_count(&self) -> u64 {
        self.conn_counts.iter().map(|(_, n)| u64::from(*n)).sum()
    }
}

/// One message record located through the index: connection, record time, and payload.
#[derive(Debug, Clone)]
pub struct RawMessage<'a> {
    pub conn: u32,
    pub time: TimeNs,
    pub data: &'a [u8],
}

/// One index_data entry: a message's time and where to find it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexEntry {
    pub time: TimeNs,
    /// Connection id. `BagSet` rewrites this to a set-wide id so several files can share one index.
    pub conn: u32,
    /// Which file of the set holds it (always 0 for a single bag).
    pub file: u16,
    /// Position in that file's `chunks` of the chunk holding this message.
    pub chunk: u32,
    /// Byte offset of the message record from the start of the decompressed chunk data.
    pub offset: u32,
}

/// Chunk compression as recorded in the chunk header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Compression {
    None,
    Lz4,
}

/// `OpenStorage` for ROS 1 bags: what the `bag` source descriptor hands to `BagHandle::spawn`.
pub fn open_storage(
    path: &Path,
    cache: usize,
    cancel: &AtomicBool,
) -> Result<Box<dyn Storage>, BagError> {
    if is_cancelled(cancel) {
        return Err(BagError::Cancelled);
    }
    Ok(Box::new(BagReader::with_cache_capacity(path, cache)?))
}

/// Reads records from an open bag file and hands out decompressed chunk bytes.
pub struct BagReader {
    file: BufReader<File>,
    path: PathBuf,
    /// Total file size, reported by `BagInfo` and used to bound offsets.
    size_bytes: u64,
    header: BagHeader,
    connections: Vec<Connection>,
    chunks: Vec<ChunkInfo>,
    /// Compression of the chunks, taken from the first chunk actually read (bags are written with a single setting).
    compression: Option<&'static str>,
    cache: ChunkCache,
}

impl BagReader {
    /// Open a bag with the default cache size.
    pub fn open(path: &Path) -> Result<Self, BagError> {
        Self::with_cache_capacity(path, CHUNK_CACHE_TOTAL)
    }

    /// Open a bag and read its index section (connections + chunk infos); chunk bodies are left untouched.
    pub fn with_cache_capacity(path: &Path, cache: usize) -> Result<Self, BagError> {
        let file = File::open(path)?;
        let size_bytes = file.metadata()?.len();
        let mut file = BufReader::new(file);
        let mut magic = vec![0u8; MAGIC.len()];
        file.read_exact(&mut magic).map_err(|_| {
            BagError::UnsupportedVersion(String::from_utf8_lossy(&magic).trim().to_owned())
        })?;
        if magic != MAGIC {
            return Err(BagError::UnsupportedVersion(
                String::from_utf8_lossy(&magic).trim().to_owned(),
            ));
        }
        let (fields, _) = read_record(&mut file)?;
        let op = fields.op()?;
        if op != OP_BAG_HEADER {
            return Err(BagError::Malformed(format!(
                "expected bag header op 0x03, found 0x{op:02x}"
            )));
        }
        let index_pos = fields.u64_field("index_pos")?;
        if index_pos == 0 || index_pos >= size_bytes {
            return Err(BagError::MissingIndex);
        }
        let header = BagHeader {
            index_pos,
            conn_count: fields.u32_field("conn_count")?,
            chunk_count: fields.u32_field("chunk_count")?,
        };
        let mut reader = Self {
            file,
            path: path.to_path_buf(),
            size_bytes,
            header,
            connections: Vec::new(),
            chunks: Vec::new(),
            compression: None,
            cache: ChunkCache::new(cache.max(1)),
        };
        reader.read_index_section()?;
        Ok(reader)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    pub fn header(&self) -> BagHeader {
        self.header
    }

    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    pub fn chunks(&self) -> &[ChunkInfo] {
        &self.chunks
    }

    /// Chunk compression seen so far; None until a chunk has been read.
    pub fn compression(&self) -> Option<&'static str> {
        self.compression
    }

    /// Total messages across all chunks (from the index alone, no chunk bodies read).
    pub fn message_count(&self) -> u64 {
        self.chunks.iter().map(ChunkInfo::message_count).sum()
    }

    /// Read the trailing index section: connection records then chunk_info records.
    fn read_index_section(&mut self) -> Result<(), BagError> {
        self.file.seek(SeekFrom::Start(self.header.index_pos))?;
        loop {
            let (fields, data) = match read_record(&mut self.file) {
                Ok(r) => r,
                // A clean end-of-file terminates the section; a torn tail is reported as a missing index.
                Err(BagError::Io(_)) => break,
                Err(e) => return Err(e),
            };
            match fields.op()? {
                OP_CONNECTION => self.connections.push(parse_connection(&fields, &data)?),
                OP_CHUNK_INFO => self.chunks.push(parse_chunk_info(&fields, &data)?),
                // Ignore anything else so a writer extension does not make the bag unreadable.
                _ => {}
            }
        }
        if self.connections.is_empty() || self.chunks.is_empty() {
            return Err(BagError::MissingIndex);
        }
        // Play order depends on this, and rosbag writes chunk_info in file order rather than time order.
        self.chunks.sort_by_key(|c| c.start);
        Ok(())
    }

    /// Build the full message index by reading the index_data records that trail each chunk (chunk bodies stay compressed).
    pub fn read_message_index(&mut self, cancel: &AtomicBool) -> Result<Vec<IndexEntry>, BagError> {
        let mut entries = Vec::with_capacity(self.message_count() as usize);
        let chunks: Vec<(u64, usize)> = self
            .chunks
            .iter()
            .map(|c| (c.pos, c.conn_counts.len()))
            .collect();
        for (chunk_index, (pos, conn_count)) in chunks.into_iter().enumerate() {
            if is_cancelled(cancel) {
                return Err(BagError::Cancelled);
            }
            self.file.seek(SeekFrom::Start(pos))?;
            let (fields, _) = read_record_skipping_data(&mut self.file)?;
            let op = fields.op()?;
            if op != OP_CHUNK {
                return Err(BagError::Malformed(format!(
                    "chunk_pos {pos} points at op 0x{op:02x}, not a chunk"
                )));
            }
            self.note_compression(&fields)?;
            for _ in 0..conn_count {
                let (fields, data) = read_record(&mut self.file)?;
                let op = fields.op()?;
                if op != OP_INDEX_DATA {
                    return Err(BagError::Malformed(format!(
                        "expected index_data after chunk at {pos}, found op 0x{op:02x}"
                    )));
                }
                let conn = fields.u32_field("conn")?;
                let count = fields.u32_field("count")? as usize;
                if data.len() != count * INDEX_ENTRY_LEN {
                    return Err(BagError::Malformed(format!(
                        "index_data for conn {conn} has {} bytes for {count} entries",
                        data.len()
                    )));
                }
                for raw in data.chunks_exact(INDEX_ENTRY_LEN) {
                    entries.push(IndexEntry {
                        time: time_from_bytes(&raw[0..8]),
                        conn,
                        file: 0,
                        chunk: chunk_index as u32,
                        offset: u32::from_le_bytes(raw[8..12].try_into().expect("4 bytes")),
                    });
                }
            }
        }
        Ok(entries)
    }

    /// Earliest record time in this bag, used to order a set of files.
    pub fn start(&self) -> TimeNs {
        self.chunks.first().map_or(0, |c| c.start)
    }

    /// Fetch one message by index entry; the chunk is decompressed on demand and cached.
    pub fn message_at(&mut self, entry: &IndexEntry) -> Result<RawMessage<'_>, BagError> {
        let slot = self.load_chunk(entry.chunk)?;
        let body = &self.cache.entries[slot].1;
        let offset = entry.offset as usize;
        let (fields, data) = read_record_from_slice(body, offset)?;
        let op = fields.op()?;
        if op != OP_MSG_DATA {
            return Err(BagError::Malformed(format!(
                "index offset {offset} in chunk {} points at op 0x{op:02x}, not a message",
                entry.chunk
            )));
        }
        Ok(RawMessage {
            conn: fields.u32_field("conn")?,
            time: fields.time_field("time")?,
            data,
        })
    }

    /// Ensure chunk `index` is decompressed and return its cache slot.
    fn load_chunk(&mut self, index: u32) -> Result<usize, BagError> {
        if let Some(slot) = self.cache.touch(index) {
            return Ok(slot);
        }
        let info = self
            .chunks
            .get(index as usize)
            .ok_or_else(|| BagError::Malformed(format!("chunk index {index} out of range")))?;
        let pos = info.pos;
        self.file.seek(SeekFrom::Start(pos))?;
        let (fields, data) = read_record(&mut self.file)?;
        let op = fields.op()?;
        if op != OP_CHUNK {
            return Err(BagError::Malformed(format!(
                "chunk_pos {pos} points at op 0x{op:02x}, not a chunk"
            )));
        }
        let compression = self.note_compression(&fields)?;
        let size = fields.u32_field("size")? as usize;
        let body = decompress_chunk(compression, &data, size)?;
        self.cache.insert(index, body);
        Ok(self.cache.touch(index).expect("just inserted"))
    }

    /// Record (and validate) the chunk compression declared in a chunk header.
    fn note_compression(&mut self, fields: &HeaderFields) -> Result<Compression, BagError> {
        let raw = fields.str_field("compression")?;
        let compression = match raw {
            "none" => Compression::None,
            "lz4" => Compression::Lz4,
            other => return Err(BagError::UnsupportedCompression(other.to_owned())),
        };
        self.compression = Some(match compression {
            Compression::None => "none",
            Compression::Lz4 => "lz4",
        });
        Ok(compression)
    }
}

impl Storage for BagReader {
    fn path(&self) -> &Path {
        BagReader::path(self)
    }

    fn size_bytes(&self) -> u64 {
        BagReader::size_bytes(self)
    }

    fn connections(&self) -> &[Connection] {
        BagReader::connections(self)
    }

    fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    fn compression(&self) -> Option<&str> {
        BagReader::compression(self)
    }

    fn message_count(&self) -> u64 {
        BagReader::message_count(self)
    }

    fn start(&self) -> TimeNs {
        BagReader::start(self)
    }

    fn read_message_index(&mut self, cancel: &AtomicBool) -> Result<Vec<IndexEntry>, BagError> {
        BagReader::read_message_index(self, cancel)
    }

    fn message_at(&mut self, entry: &IndexEntry) -> Result<RawMessage<'_>, BagError> {
        BagReader::message_at(self, entry)
    }

    /// A ROS 1 bag either has its index or is refused (`MissingIndex`), so there is never a partial success to report.
    fn warnings(&self) -> &[String] {
        &[]
    }
}

/// Decompress chunk bytes and check the result against the chunk header's uncompressed `size`.
fn decompress_chunk(
    compression: Compression,
    data: &[u8],
    size: usize,
) -> Result<Vec<u8>, BagError> {
    let body = match compression {
        Compression::None => data.to_vec(),
        Compression::Lz4 => {
            let mut out = Vec::with_capacity(size);
            lz4_flex::frame::FrameDecoder::new(data)
                .read_to_end(&mut out)
                .map_err(|e| BagError::Decompress(e.to_string()))?;
            out
        }
    };
    if body.len() != size {
        return Err(BagError::Decompress(format!(
            "decompressed {} bytes but the chunk header says {size}",
            body.len()
        )));
    }
    Ok(body)
}

/// Parsed record header: `name=value` pairs, values kept as raw bytes since `message_definition` is large.
pub struct HeaderFields {
    fields: HashMap<String, Vec<u8>>,
}

impl HeaderFields {
    /// Split a record header into fields, each stored as `len:u32 | name=value`.
    fn parse(bytes: &[u8]) -> Result<Self, BagError> {
        let mut fields = HashMap::new();
        let mut at = 0usize;
        while at < bytes.len() {
            if bytes.len() - at < 4 {
                return Err(BagError::Malformed(
                    "record header ends mid length prefix".to_owned(),
                ));
            }
            let len = u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes")) as usize;
            at += 4;
            if bytes.len() - at < len {
                return Err(BagError::Malformed(format!(
                    "record header field claims {len} bytes but {} remain",
                    bytes.len() - at
                )));
            }
            let field = &bytes[at..at + len];
            at += len;
            let eq = field
                .iter()
                .position(|b| *b == b'=')
                .ok_or_else(|| BagError::Malformed("record header field has no `=`".to_owned()))?;
            let name = String::from_utf8_lossy(&field[..eq]).into_owned();
            fields.insert(name, field[eq + 1..].to_vec());
        }
        Ok(Self { fields })
    }

    fn raw(&self, name: &'static str, op: u8) -> Result<&[u8], BagError> {
        self.fields
            .get(name)
            .map(Vec::as_slice)
            .ok_or(BagError::MissingField { op, field: name })
    }

    /// The record's op code.
    pub fn op(&self) -> Result<u8, BagError> {
        let raw = self.raw("op", 0)?;
        raw.first()
            .copied()
            .ok_or(BagError::MissingField { op: 0, field: "op" })
    }

    pub fn u32_field(&self, name: &'static str) -> Result<u32, BagError> {
        let op = self.op().unwrap_or(0);
        let raw = self.raw(name, op)?;
        let bytes: [u8; 4] = raw
            .try_into()
            .map_err(|_| BagError::Malformed(format!("field `{name}` is not 4 bytes")))?;
        Ok(u32::from_le_bytes(bytes))
    }

    pub fn u64_field(&self, name: &'static str) -> Result<u64, BagError> {
        let op = self.op().unwrap_or(0);
        let raw = self.raw(name, op)?;
        let bytes: [u8; 8] = raw
            .try_into()
            .map_err(|_| BagError::Malformed(format!("field `{name}` is not 8 bytes")))?;
        Ok(u64::from_le_bytes(bytes))
    }

    /// A ROS 1 time field: `uint32 secs` + `uint32 nsecs`, converted to the viewer's nanosecond scale.
    pub fn time_field(&self, name: &'static str) -> Result<TimeNs, BagError> {
        let op = self.op().unwrap_or(0);
        let raw = self.raw(name, op)?;
        if raw.len() != 8 {
            return Err(BagError::Malformed(format!(
                "time field `{name}` is not 8 bytes"
            )));
        }
        Ok(time_from_bytes(raw))
    }

    pub fn str_field(&self, name: &'static str) -> Result<&str, BagError> {
        let op = self.op().unwrap_or(0);
        let raw = self.raw(name, op)?;
        std::str::from_utf8(raw)
            .map_err(|_| BagError::Malformed(format!("field `{name}` is not UTF-8")))
    }
}

/// ROS 1 `secs`/`nsecs` pair (little-endian) to nanoseconds since the epoch.
fn time_from_bytes(raw: &[u8]) -> TimeNs {
    let secs = u32::from_le_bytes(raw[0..4].try_into().expect("4 bytes")) as i64;
    let nsecs = u32::from_le_bytes(raw[4..8].try_into().expect("4 bytes")) as i64;
    secs * 1_000_000_000 + nsecs
}

/// Read one record (`header_len | header | data_len | data`) from a stream.
fn read_record<R: Read>(reader: &mut R) -> Result<(HeaderFields, Vec<u8>), BagError> {
    let header_len = read_len(reader)?;
    let mut header = vec![0u8; header_len];
    reader.read_exact(&mut header)?;
    let fields = HeaderFields::parse(&header)?;
    let data_len = read_len(reader)?;
    let mut data = vec![0u8; data_len];
    reader.read_exact(&mut data)?;
    Ok((fields, data))
}

/// Read one record's header but seek past its data; used to reach the index_data records after a chunk.
fn read_record_skipping_data<R: Read + Seek>(
    reader: &mut R,
) -> Result<(HeaderFields, u64), BagError> {
    let header_len = read_len(reader)?;
    let mut header = vec![0u8; header_len];
    reader.read_exact(&mut header)?;
    let fields = HeaderFields::parse(&header)?;
    let data_len = read_len(reader)? as u64;
    reader.seek(SeekFrom::Current(data_len as i64))?;
    Ok((fields, data_len))
}

/// Read one record out of an in-memory (decompressed) chunk at `offset`, borrowing its data.
fn read_record_from_slice(body: &[u8], offset: usize) -> Result<(HeaderFields, &[u8]), BagError> {
    let at = |pos: usize, n: usize| -> Result<&[u8], BagError> {
        body.get(pos..pos + n).ok_or_else(|| {
            BagError::Malformed(format!(
                "record at offset {offset} runs past the end of a {} byte chunk",
                body.len()
            ))
        })
    };
    let header_len = u32::from_le_bytes(at(offset, 4)?.try_into().expect("4 bytes")) as usize;
    let fields = HeaderFields::parse(at(offset + 4, header_len)?)?;
    let data_start = offset + 4 + header_len;
    let data_len = u32::from_le_bytes(at(data_start, 4)?.try_into().expect("4 bytes")) as usize;
    Ok((fields, at(data_start + 4, data_len)?))
}

/// Read a u32 length prefix, rejecting absurd values before they become an allocation.
fn read_len<R: Read>(reader: &mut R) -> Result<usize, BagError> {
    let mut raw = [0u8; 4];
    reader.read_exact(&mut raw)?;
    let len = u32::from_le_bytes(raw) as usize;
    if len > MAX_RECORD_LEN {
        return Err(BagError::Malformed(format!(
            "record length {len} exceeds the {MAX_RECORD_LEN} byte limit"
        )));
    }
    Ok(len)
}

/// A connection record: topic in the header, type/md5/definition in the data (itself a field block).
fn parse_connection(fields: &HeaderFields, data: &[u8]) -> Result<Connection, BagError> {
    let sub = HeaderFields::parse(data)?;
    Ok(Connection {
        id: fields.u32_field("conn")?,
        topic_raw: fields.str_field("topic")?.to_owned(),
        type_raw: sub.str_field("type")?.to_owned(),
        type_hash: sub.str_field("md5sum")?.to_owned(),
        // Lossy so one odd byte in a comment cannot cost us the whole connection.
        definition: String::from_utf8_lossy(sub.raw("message_definition", OP_CONNECTION)?)
            .into_owned(),
        message_encoding: ROS1_MESSAGE_ENCODING.to_owned(),
        definition_encoding: ROS1_DEFINITION_ENCODING.to_owned(),
    })
}

/// A chunk_info record: position and time span in the header, per-connection counts in the data.
fn parse_chunk_info(fields: &HeaderFields, data: &[u8]) -> Result<ChunkInfo, BagError> {
    let declared = fields.u32_field("count")? as usize;
    let mut conn_counts = Vec::with_capacity(declared);
    for raw in data.chunks_exact(8) {
        conn_counts.push((
            u32::from_le_bytes(raw[0..4].try_into().expect("4 bytes")),
            u32::from_le_bytes(raw[4..8].try_into().expect("4 bytes")),
        ));
    }
    if conn_counts.len() != declared {
        return Err(BagError::Malformed(format!(
            "chunk_info declares {declared} connections but carries {}",
            conn_counts.len()
        )));
    }
    Ok(ChunkInfo {
        pos: fields.u64_field("chunk_pos")?,
        start: fields.time_field("start_time")?,
        end: fields.time_field("end_time")?,
        conn_counts,
    })
}

/// Builders shared by this module's tests and `set.rs`'s (synthetic bags, so no real file is needed).
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// Build a record header field block from `name=value` pairs.
    pub fn header(pairs: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        for (name, value) in pairs {
            let mut field = name.as_bytes().to_vec();
            field.push(b'=');
            field.extend_from_slice(value);
            out.extend_from_slice(&(field.len() as u32).to_le_bytes());
            out.extend_from_slice(&field);
        }
        out
    }

    /// Build a complete record from a header field block and data.
    pub fn record(pairs: &[(&str, &[u8])], data: &[u8]) -> Vec<u8> {
        let h = header(pairs);
        let mut out = Vec::new();
        out.extend_from_slice(&(h.len() as u32).to_le_bytes());
        out.extend_from_slice(&h);
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    /// ROS 1 time bytes for `secs.nsecs`.
    pub fn time(secs: u32, nsecs: u32) -> Vec<u8> {
        [secs.to_le_bytes(), nsecs.to_le_bytes()].concat()
    }

    /// Assemble a whole in-memory bag: magic, header, one chunk with `messages`, index_data, connection, chunk_info.
    pub fn synthetic_bag(lz4: bool, messages: &[(u32, u32, &[u8])]) -> Vec<u8> {
        let compression = if lz4 {
            Compression::Lz4
        } else {
            Compression::None
        };
        let mut chunk_body = Vec::new();
        let mut offsets = Vec::new();
        for (conn, secs, payload) in messages {
            offsets.push(chunk_body.len() as u32);
            chunk_body.extend_from_slice(&record(
                &[
                    ("op", &[OP_MSG_DATA]),
                    ("conn", &conn.to_le_bytes()),
                    ("time", &time(*secs, 0)),
                ],
                payload,
            ));
        }
        let stored = match compression {
            Compression::None => chunk_body.clone(),
            Compression::Lz4 => {
                let mut enc = lz4_flex::frame::FrameEncoder::new(Vec::new());
                std::io::Write::write_all(&mut enc, &chunk_body).unwrap();
                enc.finish().unwrap()
            }
        };
        let label: &[u8] = match compression {
            Compression::None => b"none",
            Compression::Lz4 => b"lz4",
        };
        // rosbag writes one index_data record per connection holding all of that connection's entries.
        let mut index_data = Vec::new();
        let mut conns: Vec<u32> = messages.iter().map(|(conn, _, _)| *conn).collect();
        conns.dedup();
        for conn in conns {
            let mut entries = Vec::new();
            let mut count = 0u32;
            for ((c, secs, _), offset) in messages.iter().zip(&offsets) {
                if *c != conn {
                    continue;
                }
                entries.extend_from_slice(&time(*secs, 0));
                entries.extend_from_slice(&offset.to_le_bytes());
                count += 1;
            }
            index_data.extend_from_slice(&record(
                &[
                    ("op", &[OP_INDEX_DATA]),
                    ("conn", &conn.to_le_bytes()),
                    ("count", &count.to_le_bytes()),
                ],
                &entries,
            ));
        }
        let mut out = MAGIC.to_vec();
        // The bag header is written before index_pos is known, so reserve its size and patch it afterwards.
        let header_record = |index_pos: u64| {
            record(
                &[
                    ("op", &[OP_BAG_HEADER]),
                    ("index_pos", &index_pos.to_le_bytes()),
                    ("conn_count", &1u32.to_le_bytes()),
                    ("chunk_count", &1u32.to_le_bytes()),
                ],
                &[],
            )
        };
        let header_len = header_record(0).len();
        let chunk_pos = (out.len() + header_len) as u64;
        let chunk_record = record(
            &[
                ("op", &[OP_CHUNK]),
                ("compression", label),
                ("size", &(chunk_body.len() as u32).to_le_bytes()),
            ],
            &stored,
        );
        let index_pos = chunk_pos + chunk_record.len() as u64 + index_data.len() as u64;
        out.extend_from_slice(&header_record(index_pos));
        out.extend_from_slice(&chunk_record);
        out.extend_from_slice(&index_data);
        let conn_data = header(&[
            ("type", b"std_msgs/Empty"),
            ("md5sum", b"d41d8cd98f00b204e9800998ecf8427e"),
            ("message_definition", b""),
        ]);
        out.extend_from_slice(&record(
            &[
                ("op", &[OP_CONNECTION]),
                ("conn", &messages[0].0.to_le_bytes()),
                ("topic", b"chatter"),
            ],
            &conn_data,
        ));
        let mut counts = Vec::new();
        counts.extend_from_slice(&messages[0].0.to_le_bytes());
        counts.extend_from_slice(&(messages.len() as u32).to_le_bytes());
        out.extend_from_slice(&record(
            &[
                ("op", &[OP_CHUNK_INFO]),
                ("chunk_pos", &chunk_pos.to_le_bytes()),
                ("start_time", &time(messages[0].1, 0)),
                ("end_time", &time(messages[messages.len() - 1].1, 0)),
                ("count", &1u32.to_le_bytes()),
            ],
            &counts,
        ));
        out
    }

    /// Write bytes to a scratch file and return its path.
    pub fn write_temp(name: &str, bytes: &[u8]) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("visor_bag_test_{}_{name}", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    #[test]
    fn t1_header_fields_parse_by_type() {
        let bytes = header(&[
            ("op", &[OP_MSG_DATA]),
            ("conn", &7u32.to_le_bytes()),
            ("index_pos", &1234u64.to_le_bytes()),
            ("time", &time(5, 250)),
            ("compression", b"lz4"),
        ]);
        let fields = HeaderFields::parse(&bytes).unwrap();
        assert_eq!(fields.op().unwrap(), OP_MSG_DATA);
        assert_eq!(fields.u32_field("conn").unwrap(), 7);
        assert_eq!(fields.u64_field("index_pos").unwrap(), 1234);
        assert_eq!(fields.time_field("time").unwrap(), 5_000_000_250);
        assert_eq!(fields.str_field("compression").unwrap(), "lz4");
    }

    #[test]
    fn t1_header_field_problems_are_reported() {
        let bytes = header(&[("op", &[OP_CHUNK])]);
        let fields = HeaderFields::parse(&bytes).unwrap();
        assert!(matches!(
            fields.u32_field("conn"),
            Err(BagError::MissingField {
                field: "conn",
                op: OP_CHUNK
            })
        ));
        // A value of the wrong width is malformed rather than silently truncated.
        let bytes = header(&[("op", &[OP_CHUNK]), ("size", &[1, 2])]);
        let fields = HeaderFields::parse(&bytes).unwrap();
        assert!(matches!(
            fields.u32_field("size"),
            Err(BagError::Malformed(_))
        ));
        // A field with no `=` cannot be split into a name and value.
        let mut broken = Vec::new();
        broken.extend_from_slice(&3u32.to_le_bytes());
        broken.extend_from_slice(b"abc");
        assert!(matches!(
            HeaderFields::parse(&broken),
            Err(BagError::Malformed(_))
        ));
        // A length prefix longer than the block that follows it is malformed too.
        let mut short = Vec::new();
        short.extend_from_slice(&99u32.to_le_bytes());
        short.extend_from_slice(b"op=");
        assert!(matches!(
            HeaderFields::parse(&short),
            Err(BagError::Malformed(_))
        ));
    }

    #[test]
    fn t2_connection_record_carries_its_definition() {
        let data = header(&[
            ("type", b"nav_msgs/Odometry"),
            ("md5sum", b"cd5e73d190d741a2f92e81eda573aca7"),
            (
                "message_definition",
                b"Header header\nstring child_frame_id",
            ),
        ]);
        let fields = HeaderFields::parse(&header(&[
            ("op", &[OP_CONNECTION]),
            ("conn", &3u32.to_le_bytes()),
            ("topic", b"odom"),
        ]))
        .unwrap();
        let conn = parse_connection(&fields, &data).unwrap();
        assert_eq!(conn.id, 3);
        assert_eq!(conn.topic_raw, "odom");
        assert_eq!(conn.type_raw, "nav_msgs/Odometry");
        assert_eq!(conn.type_hash, "cd5e73d190d741a2f92e81eda573aca7");
        assert!(conn.definition.contains("child_frame_id"));
        // A ROS 1 bag always carries ROS 1 payloads and `.msg` text, named the way MCAP names them.
        assert_eq!(conn.message_encoding, "ros1");
        assert_eq!(conn.definition_encoding, "ros1msg");
    }

    #[test]
    fn t2_chunk_info_counts_connections_not_messages() {
        let data = [
            [1u32.to_le_bytes(), 10u32.to_le_bytes()].concat(),
            [2u32.to_le_bytes(), 5u32.to_le_bytes()].concat(),
        ]
        .concat();
        let fields = HeaderFields::parse(&header(&[
            ("op", &[OP_CHUNK_INFO]),
            ("chunk_pos", &4096u64.to_le_bytes()),
            ("start_time", &time(10, 0)),
            ("end_time", &time(11, 500)),
            ("count", &2u32.to_le_bytes()),
        ]))
        .unwrap();
        let info = parse_chunk_info(&fields, &data).unwrap();
        assert_eq!(info.pos, 4096);
        assert_eq!(info.start, 10_000_000_000);
        assert_eq!(info.end, 11_000_000_500);
        // `count` is 2 (connections) while the chunk holds 15 messages.
        assert_eq!(info.conn_counts.len(), 2);
        assert_eq!(info.message_count(), 15);
    }

    #[test]
    fn t2_chunk_info_with_a_count_mismatch_is_rejected() {
        let data = [1u32.to_le_bytes(), 10u32.to_le_bytes()].concat();
        let fields = HeaderFields::parse(&header(&[
            ("op", &[OP_CHUNK_INFO]),
            ("chunk_pos", &0u64.to_le_bytes()),
            ("start_time", &time(0, 0)),
            ("end_time", &time(0, 0)),
            ("count", &4u32.to_le_bytes()),
        ]))
        .unwrap();
        assert!(matches!(
            parse_chunk_info(&fields, &data),
            Err(BagError::Malformed(_))
        ));
    }

    #[test]
    fn t3_chunk_decompression_round_trips_both_codecs() {
        let body: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        assert_eq!(
            decompress_chunk(Compression::None, &body, body.len()).unwrap(),
            body
        );
        let compressed = lz4_flex::frame::FrameEncoder::new(Vec::new());
        let mut compressed = compressed;
        std::io::Write::write_all(&mut compressed, &body).unwrap();
        let compressed = compressed.finish().unwrap();
        // Standard LZ4 frame magic, matching what real bags store.
        assert_eq!(&compressed[..4], &[0x04, 0x22, 0x4d, 0x18]);
        assert_eq!(
            decompress_chunk(Compression::Lz4, &compressed, body.len()).unwrap(),
            body
        );
    }

    #[test]
    fn t3_size_mismatch_and_bad_lz4_are_distinct_failures() {
        let body = vec![7u8; 16];
        assert!(matches!(
            decompress_chunk(Compression::None, &body, 17),
            Err(BagError::Decompress(_))
        ));
        assert!(matches!(
            decompress_chunk(Compression::Lz4, b"not an lz4 frame", 16),
            Err(BagError::Decompress(_))
        ));
    }

    #[test]
    fn t3_message_record_is_read_out_of_a_decompressed_chunk() {
        let mut body = vec![0xAAu8; 5];
        let offset = body.len();
        body.extend_from_slice(&record(
            &[
                ("op", &[OP_MSG_DATA]),
                ("conn", &2u32.to_le_bytes()),
                ("time", &time(3, 7)),
            ],
            b"payload",
        ));
        let (fields, data) = read_record_from_slice(&body, offset).unwrap();
        assert_eq!(fields.op().unwrap(), OP_MSG_DATA);
        assert_eq!(fields.u32_field("conn").unwrap(), 2);
        assert_eq!(fields.time_field("time").unwrap(), 3_000_000_007);
        assert_eq!(data, b"payload");
        // An offset past the end reports where it ran out instead of panicking.
        assert!(matches!(
            read_record_from_slice(&body, body.len() - 2),
            Err(BagError::Malformed(_))
        ));
    }

    #[test]
    fn t4_synthetic_bag_reads_end_to_end() {
        for lz4 in [false, true] {
            let bytes = synthetic_bag(
                lz4,
                &[(4, 100, b"first"), (4, 101, b"second"), (4, 102, b"third")],
            );
            let path = write_temp(&format!("codec{lz4}.bag"), &bytes);
            let mut reader = BagReader::open(&path).unwrap();
            assert_eq!(reader.connections().len(), 1);
            assert_eq!(reader.connections()[0].topic_raw, "chatter");
            assert_eq!(reader.chunks().len(), 1);
            assert_eq!(reader.message_count(), 3);
            let index = reader.read_message_index(&AtomicBool::new(false)).unwrap();
            assert_eq!(index.len(), 3);
            assert_eq!(index[0].time, 100_000_000_000);
            assert_eq!(reader.compression(), Some(if lz4 { "lz4" } else { "none" }));
            let payloads: Vec<Vec<u8>> = index
                .iter()
                .map(|e| reader.message_at(e).unwrap().data.to_vec())
                .collect();
            assert_eq!(
                payloads,
                vec![b"first".to_vec(), b"second".to_vec(), b"third".to_vec()]
            );
            std::fs::remove_file(&path).ok();
        }
    }

    #[test]
    fn t4_each_kind_of_broken_bag_gets_its_own_error() {
        let path = write_temp("v1.bag", b"#ROSBAG V1.2\nrest of file");
        assert!(matches!(
            BagReader::open(&path),
            Err(BagError::UnsupportedVersion(_))
        ));
        std::fs::remove_file(&path).ok();

        // A bag still being recorded has index_pos = 0 (Q10 = A).
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&record(
            &[
                ("op", &[OP_BAG_HEADER]),
                ("index_pos", &0u64.to_le_bytes()),
                ("conn_count", &0u32.to_le_bytes()),
                ("chunk_count", &0u32.to_le_bytes()),
            ],
            &[],
        ));
        let path = write_temp("active.bag", &bytes);
        assert!(matches!(
            BagReader::open(&path),
            Err(BagError::MissingIndex)
        ));
        std::fs::remove_file(&path).ok();

        // Truncated right after the magic line: no header record to read.
        let path = write_temp("torn.bag", MAGIC);
        assert!(matches!(BagReader::open(&path), Err(BagError::Io(_))));
        std::fs::remove_file(&path).ok();

        // bz2 is refused by name rather than being misread (Q3 = A).
        let mut bytes = synthetic_bag(false, &[(4, 1, b"x")]);
        let at = bytes
            .windows(4)
            .position(|w| w == b"none")
            .expect("compression label present");
        bytes[at..at + 4].copy_from_slice(b"bzip");
        let path = write_temp("bz2.bag", &bytes);
        let mut reader = BagReader::open(&path).unwrap();
        assert!(matches!(
            reader.read_message_index(&AtomicBool::new(false)),
            Err(BagError::UnsupportedCompression(_))
        ));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_raised_cancel_flag_stops_opening_and_indexing() {
        let path = write_temp("cancel.bag", &synthetic_bag(false, &[(4, 1, b"x")]));
        let cancelled = AtomicBool::new(true);
        assert!(matches!(
            open_storage(&path, 1, &cancelled),
            Err(BagError::Cancelled)
        ));
        let mut reader = BagReader::open(&path).unwrap();
        assert!(matches!(
            reader.read_message_index(&cancelled),
            Err(BagError::Cancelled)
        ));
        // The flag is consulted per chunk, so a clear flag reads the whole index as before.
        assert_eq!(
            reader
                .read_message_index(&AtomicBool::new(false))
                .unwrap()
                .len(),
            1
        );
        std::fs::remove_file(&path).ok();
    }
}
