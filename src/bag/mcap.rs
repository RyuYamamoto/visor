//! rosbag2 MCAP storage: one top-level record walk that tolerates a torn tail (no Summary / Footer), chunk decompression (`""` / `lz4` / `zstd`) through the shared LRU cache, and a message index built from MessageIndex records or by scanning chunks.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use super::reader::{BagError, Connection, IndexEntry, RawMessage};
use super::storage::{ChunkCache, Storage, is_cancelled};
use crate::tf::buffer::TimeNs;

/// Opening and closing magic of every MCAP file (spec: `0x89 M C A P 0x30 \r \n`).
const MAGIC: [u8; 8] = [0x89, b'M', b'C', b'A', b'P', 0x30, b'\r', b'\n'];

/// Record opcodes (MCAP spec "Records").
const OP_HEADER: u8 = 0x01;
const OP_FOOTER: u8 = 0x02;
const OP_SCHEMA: u8 = 0x03;
const OP_CHANNEL: u8 = 0x04;
const OP_MESSAGE: u8 = 0x05;
const OP_CHUNK: u8 = 0x06;
const OP_MESSAGE_INDEX: u8 = 0x07;
const OP_CHUNK_INDEX: u8 = 0x08;
const OP_ATTACHMENT: u8 = 0x09;
const OP_ATTACHMENT_INDEX: u8 = 0x0A;
const OP_STATISTICS: u8 = 0x0B;
const OP_METADATA: u8 = 0x0C;
const OP_METADATA_INDEX: u8 = 0x0D;
const OP_SUMMARY_OFFSET: u8 = 0x0E;
const OP_DATA_END: u8 = 0x0F;

/// Every record starts with `opcode u8 | length u64`.
const RECORD_PREFIX: u64 = 9;
/// Fixed part of a Chunk record before the compression string: start, end, uncompressed_size (u64 each) and uncompressed_crc (u32).
const CHUNK_FIXED_HEAD: u64 = 28;
/// Guard against a corrupt length demanding a huge allocation; rosbag2 chunks are well under this.
const MAX_BODY_LEN: u64 = 1 << 31;
/// Largest decompressed chunk accepted (rosbag2 writes 768 KiB); a header claiming more is corruption, not something to allocate for.
const MAX_CHUNK_SIZE: u64 = 1 << 30;
/// Channel metadata key rosbag2 (Jazzy) uses for the RIHS type hash.
const TYPE_HASH_KEY: &str = "topic_type_hash";

struct Schema {
    name: String,
    encoding: String,
    data: Vec<u8>,
}

struct Channel {
    schema_id: u16,
    topic: String,
    message_encoding: String,
    metadata: Vec<(String, String)>,
}

/// One Chunk record: where its compressed records are and which MessageIndex records follow it.
struct ChunkInfo {
    start: TimeNs,
    compression: String,
    uncompressed_size: u64,
    /// File offset and length of the compressed `records` bytes.
    records_pos: u64,
    records_len: u64,
    /// File offset and body length of every MessageIndex record that followed this chunk; empty means "scan the chunk".
    message_indexes: Vec<(u64, u64)>,
}

/// Everything the top-level walk collects before the reader is assembled.
#[derive(Default)]
struct Walk {
    schemas: BTreeMap<u16, Schema>,
    channels: BTreeMap<u16, Channel>,
    chunks: Vec<ChunkInfo>,
    statistics_count: Option<u64>,
    chunk_index_count: usize,
    /// DataEnd was reached, so every chunk's MessageIndex records were written in full.
    data_end: bool,
    footer: bool,
    warnings: Vec<String>,
}

/// Reads an MCAP file written by rosbag2 (chunked; Summary optional) and hands out decompressed chunk bytes.
pub struct McapReader {
    file: BufReader<File>,
    path: PathBuf,
    size_bytes: u64,
    connections: Vec<Connection>,
    chunks: Vec<ChunkInfo>,
    /// `message_count` from the Statistics record, when the Summary was present.
    statistics_count: Option<u64>,
    /// Entries produced by `read_message_index`, the count to report when there are no Statistics.
    indexed: Option<u64>,
    warnings: Vec<String>,
    cache: ChunkCache,
}

/// `OpenStorage` for `.mcap` files.
pub fn open_storage(
    path: &Path,
    cache: usize,
    cancel: &AtomicBool,
) -> Result<Box<dyn Storage>, BagError> {
    Ok(Box::new(McapReader::open(path, cache, cancel)?))
}

impl McapReader {
    /// Open the file and walk its top-level records; chunk bodies are only decompressed when the Summary is missing.
    pub fn open(path: &Path, cache: usize, cancel: &AtomicBool) -> Result<Self, BagError> {
        let file = File::open(path)?;
        let size_bytes = file.metadata()?.len();
        let mut file = BufReader::new(file);
        let mut magic = [0u8; MAGIC.len()];
        if file.read_exact(&mut magic).is_err() || magic != MAGIC {
            return Err(BagError::UnsupportedVersion(format!("{magic:02x?}")));
        }
        let mut walk = Walk::default();
        walk_records(&mut file, size_bytes, cancel, &mut walk)?;
        // Play order depends on this, and a chunk holding only late-arriving topics can start after the next one.
        walk.chunks.sort_by_key(|c| c.start);
        let mut reader = Self {
            file,
            path: path.to_path_buf(),
            size_bytes,
            connections: Vec::new(),
            chunks: walk.chunks,
            statistics_count: walk.statistics_count,
            indexed: None,
            warnings: walk.warnings,
            cache: ChunkCache::new(cache.max(1)),
        };
        // rosbag2 writes Schema / Channel records inside the chunks as well as in the Summary, so a torn file still has them.
        if !walk.footer || walk.channels.is_empty() {
            reader.scan_chunks_for_channels(&mut walk.schemas, &mut walk.channels, cancel)?;
        }
        reader.connections = connections_from(&walk.schemas, &walk.channels);
        Ok(reader)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Chunk compression as the first chunk declares it, with MCAP's empty string shown as `none` like a ROS 1 bag.
    pub fn compression(&self) -> Option<&str> {
        self.chunks.first().map(|c| match c.compression.as_str() {
            "" => "none",
            other => other,
        })
    }

    pub fn message_count(&self) -> u64 {
        self.statistics_count.or(self.indexed).unwrap_or(0)
    }

    pub fn start(&self) -> TimeNs {
        self.chunks.first().map_or(0, |c| c.start)
    }

    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Build the message index from the MessageIndex records, scanning any chunk that has none.
    pub fn read_message_index(&mut self, cancel: &AtomicBool) -> Result<Vec<IndexEntry>, BagError> {
        let mut entries = Vec::new();
        for chunk in 0..self.chunks.len() {
            if is_cancelled(cancel) {
                return Err(BagError::Cancelled);
            }
            let indexes = self.chunks[chunk].message_indexes.clone();
            if indexes.is_empty() {
                let body = self.read_chunk(chunk)?;
                scan_messages(&body, chunk as u32, &mut entries)?;
                continue;
            }
            for (pos, len) in indexes {
                let body = read_at(&mut self.file, pos, len)?;
                parse_message_index(&body, chunk as u32, &mut entries)?;
            }
        }
        if let Some(count) = self.statistics_count
            && count != entries.len() as u64
        {
            self.warnings.push(format!(
                "statistics say {count} messages but the index has {}",
                entries.len()
            ));
        }
        self.indexed = Some(entries.len() as u64);
        Ok(entries)
    }

    /// Fetch one message by index entry; the chunk is decompressed on demand and cached.
    pub fn message_at(&mut self, entry: &IndexEntry) -> Result<RawMessage<'_>, BagError> {
        let slot = self.load_chunk(entry.chunk)?;
        let body = &self.cache.entries[slot].1;
        let offset = entry.offset as usize;
        let (op, record) = record_at(body, offset)?;
        if op != OP_MESSAGE {
            return Err(BagError::Malformed(format!(
                "index offset {offset} in chunk {} points at opcode 0x{op:02x}, not a message",
                entry.chunk
            )));
        }
        let mut cursor = Cursor::new(record);
        let conn = u32::from(cursor.u16()?);
        let _sequence = cursor.u32()?;
        let time = cursor.u64()? as TimeNs;
        let _publish_time = cursor.u64()?;
        Ok(RawMessage {
            conn,
            time,
            data: cursor.rest(),
        })
    }

    /// Ensure chunk `index` is decompressed and return its cache slot.
    fn load_chunk(&mut self, index: u32) -> Result<usize, BagError> {
        if let Some(slot) = self.cache.touch(index) {
            return Ok(slot);
        }
        let body = self.read_chunk(index as usize)?;
        self.cache.insert(index, body);
        Ok(self.cache.touch(index).expect("just inserted"))
    }

    /// Read and decompress one chunk's records without touching the cache (index scans and channel scans read every chunk once).
    fn read_chunk(&mut self, index: usize) -> Result<Vec<u8>, BagError> {
        let info = self
            .chunks
            .get(index)
            .ok_or_else(|| BagError::Malformed(format!("chunk index {index} out of range")))?;
        let (pos, len, size) = (info.records_pos, info.records_len, info.uncompressed_size);
        let compression = info.compression.clone();
        let data = read_at(&mut self.file, pos, len)?;
        decompress_chunk(&compression, &data, size)
    }

    /// Decompress every chunk and pick up the Schema / Channel records rosbag2 wrote inside them.
    fn scan_chunks_for_channels(
        &mut self,
        schemas: &mut BTreeMap<u16, Schema>,
        channels: &mut BTreeMap<u16, Channel>,
        cancel: &AtomicBool,
    ) -> Result<(), BagError> {
        for index in 0..self.chunks.len() {
            if is_cancelled(cancel) {
                return Err(BagError::Cancelled);
            }
            let body = self.read_chunk(index)?;
            let mut at = 0usize;
            while at < body.len() {
                let (op, record) = record_at(&body, at)?;
                match op {
                    OP_SCHEMA => {
                        let (id, schema) = parse_schema(record)?;
                        schemas.entry(id).or_insert(schema);
                    }
                    OP_CHANNEL => {
                        let (id, channel) = parse_channel(record)?;
                        channels.entry(id).or_insert(channel);
                    }
                    _ => {}
                }
                at += RECORD_PREFIX as usize + record.len();
            }
        }
        if !self.chunks.is_empty() {
            self.warnings.push(format!(
                "no summary section; scanned {} chunk(s) for topics",
                self.chunks.len()
            ));
        }
        Ok(())
    }
}

impl Storage for McapReader {
    fn path(&self) -> &Path {
        McapReader::path(self)
    }

    fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    fn connections(&self) -> &[Connection] {
        McapReader::connections(self)
    }

    fn chunk_count(&self) -> usize {
        McapReader::chunk_count(self)
    }

    fn compression(&self) -> Option<&str> {
        McapReader::compression(self)
    }

    fn message_count(&self) -> u64 {
        McapReader::message_count(self)
    }

    fn start(&self) -> TimeNs {
        McapReader::start(self)
    }

    fn read_message_index(&mut self, cancel: &AtomicBool) -> Result<Vec<IndexEntry>, BagError> {
        McapReader::read_message_index(self, cancel)
    }

    fn message_at(&mut self, entry: &IndexEntry) -> Result<RawMessage<'_>, BagError> {
        McapReader::message_at(self, entry)
    }

    fn warnings(&self) -> &[String] {
        McapReader::warnings(self)
    }
}

/// Walk the top-level records from just after the opening magic, collecting chunk positions and summary information; a torn tail ends the walk with a warning, an unknown opcode or a top-level Message rejects the file.
fn walk_records(
    file: &mut BufReader<File>,
    size: u64,
    cancel: &AtomicBool,
    walk: &mut Walk,
) -> Result<(), BagError> {
    let mut pos = MAGIC.len() as u64;
    loop {
        if is_cancelled(cancel) {
            return Err(BagError::Cancelled);
        }
        let remaining = size.saturating_sub(pos);
        if remaining == 0 {
            break;
        }
        if remaining < RECORD_PREFIX {
            walk.warnings.push(format!(
                "{remaining} trailing byte(s) after the last record were discarded"
            ));
            break;
        }
        let prefix = read_at(file, pos, RECORD_PREFIX)?;
        let op = prefix[0];
        let len = u64::from_le_bytes(prefix[1..9].try_into().expect("8 bytes"));
        let body_start = pos + RECORD_PREFIX;
        if len > remaining - RECORD_PREFIX {
            walk.warnings.push(format!(
                "record 0x{op:02x} at byte {pos} runs {len} bytes past the end of the file; the torn tail was discarded"
            ));
            break;
        }
        let body_end = body_start + len;
        match op {
            OP_MESSAGE => {
                return Err(BagError::Unsupported(
                    "unchunked MCAP (top-level Message records, as rosbag2's `fastwrite` preset writes)".to_owned(),
                ));
            }
            OP_SCHEMA => {
                let (id, schema) = parse_schema(&read_at(file, body_start, len)?)?;
                walk.schemas.insert(id, schema);
            }
            OP_CHANNEL => {
                let (id, channel) = parse_channel(&read_at(file, body_start, len)?)?;
                walk.channels.insert(id, channel);
            }
            OP_CHUNK => walk
                .chunks
                .push(parse_chunk_head(file, body_start, body_end)?),
            OP_MESSAGE_INDEX => match walk.chunks.last_mut() {
                Some(chunk) => chunk.message_indexes.push((body_start, len)),
                None => {
                    return Err(BagError::Malformed(format!(
                        "MessageIndex at byte {pos} precedes any Chunk"
                    )));
                }
            },
            OP_CHUNK_INDEX => walk.chunk_index_count += 1,
            OP_STATISTICS => {
                let body = read_at(file, body_start, len)?;
                walk.statistics_count = Some(Cursor::new(&body).u64()?);
            }
            OP_FOOTER => {
                walk.footer = true;
                let closing = read_at(file, body_end, MAGIC.len() as u64).ok();
                if closing.as_deref() != Some(&MAGIC[..]) {
                    walk.warnings
                        .push("footer present but the closing magic is missing".to_owned());
                }
                break;
            }
            OP_DATA_END => walk.data_end = true,
            OP_HEADER | OP_ATTACHMENT | OP_ATTACHMENT_INDEX | OP_METADATA | OP_METADATA_INDEX
            | OP_SUMMARY_OFFSET => {}
            other => {
                return Err(BagError::Malformed(format!(
                    "unknown record opcode 0x{other:02x} at byte {pos}"
                )));
            }
        }
        pos = body_end;
    }
    // A tail torn before DataEnd may have cut the last chunk's MessageIndex records short (one per channel); that chunk is scanned instead of trusting a partial index.
    if !walk.data_end
        && let Some(last) = walk.chunks.last_mut()
        && !last.message_indexes.is_empty()
    {
        last.message_indexes.clear();
        walk.warnings.push(
            "the last chunk's message index may be incomplete; that chunk is scanned".to_owned(),
        );
    }
    if !walk.footer {
        walk.warnings.push(
            "no footer: the file was cut short (still recording, or the writer crashed); the data section was scanned instead of the summary".to_owned(),
        );
    }
    if walk.chunk_index_count > 0 && walk.chunk_index_count != walk.chunks.len() {
        walk.warnings.push(format!(
            "summary lists {} chunk index record(s) but the data section holds {} chunk(s)",
            walk.chunk_index_count,
            walk.chunks.len()
        ));
    }
    Ok(())
}

/// Read the fixed header of a Chunk record, leaving its compressed records where they are.
fn parse_chunk_head(
    file: &mut BufReader<File>,
    body_start: u64,
    body_end: u64,
) -> Result<ChunkInfo, BagError> {
    let head = read_at(file, body_start, CHUNK_FIXED_HEAD + 4)?;
    let mut cursor = Cursor::new(&head);
    let start = cursor.u64()? as TimeNs;
    let _end = cursor.u64()?;
    let uncompressed_size = cursor.u64()?;
    if uncompressed_size > MAX_CHUNK_SIZE {
        return Err(BagError::Malformed(format!(
            "chunk at byte {body_start} claims {uncompressed_size} uncompressed bytes, over the {MAX_CHUNK_SIZE} byte limit"
        )));
    }
    let _crc = cursor.u32()?;
    let compression_len = u64::from(cursor.u32()?);
    let compression = read_at(file, body_start + CHUNK_FIXED_HEAD + 4, compression_len)?;
    let compression = String::from_utf8(compression)
        .map_err(|_| BagError::Malformed("chunk compression name is not UTF-8".to_owned()))?;
    let records_len_pos = body_start + CHUNK_FIXED_HEAD + 4 + compression_len;
    let records_len = Cursor::new(&read_at(file, records_len_pos, 8)?).u64()?;
    let records_pos = records_len_pos + 8;
    if records_pos + records_len != body_end {
        return Err(BagError::Malformed(format!(
            "chunk at byte {body_start} declares {records_len} record bytes but its record length says {}",
            body_end.saturating_sub(records_pos)
        )));
    }
    Ok(ChunkInfo {
        start,
        compression,
        uncompressed_size,
        records_pos,
        records_len,
        message_indexes: Vec::new(),
    })
}

/// Schema body: `id u16 | name str | encoding str | data (u32 len + bytes)`.
fn parse_schema(body: &[u8]) -> Result<(u16, Schema), BagError> {
    let mut cursor = Cursor::new(body);
    let id = cursor.u16()?;
    let name = cursor.str()?;
    let encoding = cursor.str()?;
    let data = cursor.prefixed_bytes()?.to_vec();
    Ok((
        id,
        Schema {
            name,
            encoding,
            data,
        },
    ))
}

/// Channel body: `id u16 | schema_id u16 | topic str | message_encoding str | metadata map<str, str>`.
fn parse_channel(body: &[u8]) -> Result<(u16, Channel), BagError> {
    let mut cursor = Cursor::new(body);
    let id = cursor.u16()?;
    let schema_id = cursor.u16()?;
    let topic = cursor.str()?;
    let message_encoding = cursor.str()?;
    let mut metadata = Vec::new();
    let mut map = Cursor::new(cursor.prefixed_bytes()?);
    while !map.is_empty() {
        let key = map.str()?;
        let value = map.str()?;
        metadata.push((key, value));
    }
    Ok((
        id,
        Channel {
            schema_id,
            topic,
            message_encoding,
            metadata,
        },
    ))
}

/// MessageIndex body: `channel_id u16 | records (u32 byte length + (log_time u64, offset u64)…)`.
fn parse_message_index(
    body: &[u8],
    chunk: u32,
    entries: &mut Vec<IndexEntry>,
) -> Result<(), BagError> {
    let mut cursor = Cursor::new(body);
    let conn = u32::from(cursor.u16()?);
    let records = cursor.prefixed_bytes()?;
    if records.len() % 16 != 0 {
        return Err(BagError::Malformed(format!(
            "message index for channel {conn} has {} bytes, not a multiple of 16",
            records.len()
        )));
    }
    for pair in records.chunks_exact(16) {
        let time = u64::from_le_bytes(pair[0..8].try_into().expect("8 bytes")) as TimeNs;
        let offset = u64::from_le_bytes(pair[8..16].try_into().expect("8 bytes"));
        entries.push(IndexEntry {
            time,
            conn,
            file: 0,
            chunk,
            offset: offset_u32(offset)?,
        });
    }
    Ok(())
}

/// Enumerate the Message records of a decompressed chunk (the fallback when the chunk has no MessageIndex).
fn scan_messages(body: &[u8], chunk: u32, entries: &mut Vec<IndexEntry>) -> Result<(), BagError> {
    let mut at = 0usize;
    while at < body.len() {
        let (op, record) = record_at(body, at)?;
        if op == OP_MESSAGE {
            let mut cursor = Cursor::new(record);
            let conn = u32::from(cursor.u16()?);
            let _sequence = cursor.u32()?;
            let time = cursor.u64()? as TimeNs;
            entries.push(IndexEntry {
                time,
                conn,
                file: 0,
                chunk,
                offset: offset_u32(at as u64)?,
            });
        }
        at += RECORD_PREFIX as usize + record.len();
    }
    Ok(())
}

/// `IndexEntry.offset` is 32 bits; a chunk large enough to overflow it is not something rosbag2 writes.
fn offset_u32(offset: u64) -> Result<u32, BagError> {
    u32::try_from(offset).map_err(|_| {
        BagError::Malformed(format!(
            "message offset {offset} exceeds the 4 GiB chunk limit"
        ))
    })
}

/// Turn every channel into a `Connection`, taking the type name and definition from its schema (schema 0 means none).
fn connections_from(
    schemas: &BTreeMap<u16, Schema>,
    channels: &BTreeMap<u16, Channel>,
) -> Vec<Connection> {
    channels
        .iter()
        .map(|(id, channel)| {
            let schema = schemas.get(&channel.schema_id);
            Connection {
                id: u32::from(*id),
                topic_raw: channel.topic.clone(),
                type_raw: schema.map(|s| s.name.clone()).unwrap_or_default(),
                type_hash: channel
                    .metadata
                    .iter()
                    .find(|(key, _)| key == TYPE_HASH_KEY)
                    .map(|(_, value)| value.clone())
                    .unwrap_or_default(),
                definition: schema
                    .map(|s| String::from_utf8_lossy(&s.data).into_owned())
                    .unwrap_or_default(),
                message_encoding: channel.message_encoding.clone(),
                definition_encoding: schema.map(|s| s.encoding.clone()).unwrap_or_default(),
            }
        })
        .collect()
}

/// Decompress chunk records and check the result against the chunk header's `uncompressed_size`. The declared size caps both the allocation and how far a decoder is read, so a corrupt header cannot make either unbounded.
fn decompress_chunk(compression: &str, data: &[u8], size: u64) -> Result<Vec<u8>, BagError> {
    if size > MAX_CHUNK_SIZE {
        return Err(BagError::Decompress(format!(
            "chunk header claims {size} uncompressed bytes, over the {MAX_CHUNK_SIZE} byte limit"
        )));
    }
    // One byte beyond the declared size is enough to notice an oversized stream without decoding all of it.
    let limit = size + 1;
    let body = match compression {
        "" | "none" => data.to_vec(),
        "lz4" => {
            let mut out = Vec::with_capacity(size as usize);
            lz4_flex::frame::FrameDecoder::new(data)
                .take(limit)
                .read_to_end(&mut out)
                .map_err(|e| BagError::Decompress(e.to_string()))?;
            out
        }
        "zstd" => {
            let mut out = Vec::with_capacity(size as usize);
            ruzstd::decoding::StreamingDecoder::new(data)
                .map_err(|e| BagError::Decompress(e.to_string()))?
                .take(limit)
                .read_to_end(&mut out)
                .map_err(|e| BagError::Decompress(e.to_string()))?;
            out
        }
        other => return Err(BagError::UnsupportedCompression(other.to_owned())),
    };
    if body.len() as u64 != size {
        return Err(BagError::Decompress(format!(
            "decompressed {} bytes but the chunk header says {size}",
            body.len()
        )));
    }
    Ok(body)
}

/// One record out of an in-memory record stream at `offset`: `(opcode, body)`.
fn record_at(buf: &[u8], offset: usize) -> Result<(u8, &[u8]), BagError> {
    let overrun = || {
        BagError::Malformed(format!(
            "record at offset {offset} runs past the end of a {} byte chunk",
            buf.len()
        ))
    };
    let prefix = buf
        .get(offset..offset + RECORD_PREFIX as usize)
        .ok_or_else(overrun)?;
    let len = u64::from_le_bytes(prefix[1..9].try_into().expect("8 bytes"));
    let start = offset + RECORD_PREFIX as usize;
    let end = usize::try_from(len)
        .ok()
        .and_then(|len| start.checked_add(len))
        .ok_or_else(overrun)?;
    Ok((prefix[0], buf.get(start..end).ok_or_else(overrun)?))
}

/// Read `len` bytes at absolute file offset `pos`.
fn read_at(file: &mut BufReader<File>, pos: u64, len: u64) -> Result<Vec<u8>, BagError> {
    if len > MAX_BODY_LEN {
        return Err(BagError::Malformed(format!(
            "record length {len} exceeds the {MAX_BODY_LEN} byte limit"
        )));
    }
    file.seek(SeekFrom::Start(pos))?;
    let mut out = vec![0u8; len as usize];
    file.read_exact(&mut out)?;
    Ok(out)
}

/// Little-endian field reader over a record body; every overrun is a `Malformed` error rather than a panic.
struct Cursor<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, at: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], BagError> {
        let end = self.at.checked_add(n).filter(|end| *end <= self.buf.len());
        let end = end.ok_or_else(|| {
            BagError::Malformed(format!(
                "record body ends after {} bytes where {n} more were expected",
                self.buf.len()
            ))
        })?;
        let out = &self.buf[self.at..end];
        self.at = end;
        Ok(out)
    }

    fn u16(&mut self) -> Result<u16, BagError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("2 bytes"),
        ))
    }

    fn u32(&mut self) -> Result<u32, BagError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    fn u64(&mut self) -> Result<u64, BagError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }

    /// `u32 length + bytes`, the shape of every string, byte array and map in MCAP.
    fn prefixed_bytes(&mut self) -> Result<&'a [u8], BagError> {
        let len = self.u32()? as usize;
        self.take(len)
    }

    fn str(&mut self) -> Result<String, BagError> {
        String::from_utf8(self.prefixed_bytes()?.to_vec())
            .map_err(|_| BagError::Malformed("string field is not UTF-8".to_owned()))
    }

    fn rest(&self) -> &'a [u8] {
        &self.buf[self.at..]
    }

    fn is_empty(&self) -> bool {
        self.at >= self.buf.len()
    }
}

/// Builders shared by this module's tests and `set.rs`'s: whole MCAP files assembled in memory the way rosbag2 lays them out.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// Test schema every synthetic channel points at.
    pub const SCHEMA_NAME: &str = "std_msgs/msg/String";
    pub const SCHEMA_TEXT: &str = "string data";

    /// Shape of a synthetic file; `chunks` holds `(channel, log_time, payload)` per message, one inner list per chunk.
    pub struct SyntheticMcap {
        pub compression: String,
        pub chunks: Vec<Vec<(u16, u64, Vec<u8>)>>,
        pub message_index: bool,
        pub summary: bool,
        /// Append one Message record at top level (what an unchunked writer produces).
        pub top_level_message: bool,
    }

    impl Default for SyntheticMcap {
        fn default() -> Self {
            Self {
                compression: String::new(),
                chunks: Vec::new(),
                message_index: true,
                summary: true,
                top_level_message: false,
            }
        }
    }

    pub fn record(op: u8, body: &[u8]) -> Vec<u8> {
        let mut out = vec![op];
        out.extend_from_slice(&(body.len() as u64).to_le_bytes());
        out.extend_from_slice(body);
        out
    }

    pub fn str_field(s: &str) -> Vec<u8> {
        let mut out = (s.len() as u32).to_le_bytes().to_vec();
        out.extend_from_slice(s.as_bytes());
        out
    }

    fn schema_record() -> Vec<u8> {
        let mut body = 1u16.to_le_bytes().to_vec();
        body.extend(str_field(SCHEMA_NAME));
        body.extend(str_field("ros2msg"));
        body.extend(str_field(SCHEMA_TEXT));
        record(OP_SCHEMA, &body)
    }

    fn channel_record(id: u16) -> Vec<u8> {
        let mut body = id.to_le_bytes().to_vec();
        body.extend_from_slice(&1u16.to_le_bytes());
        body.extend(str_field(&format!("/topic{id}")));
        body.extend(str_field("cdr"));
        let mut map = str_field(TYPE_HASH_KEY);
        map.extend(str_field("RIHS01_test"));
        body.extend_from_slice(&(map.len() as u32).to_le_bytes());
        body.extend(map);
        record(OP_CHANNEL, &body)
    }

    fn message_record(channel: u16, log_time: u64, sequence: u32, payload: &[u8]) -> Vec<u8> {
        let mut body = channel.to_le_bytes().to_vec();
        body.extend_from_slice(&sequence.to_le_bytes());
        body.extend_from_slice(&log_time.to_le_bytes());
        body.extend_from_slice(&log_time.to_le_bytes());
        body.extend_from_slice(payload);
        record(OP_MESSAGE, &body)
    }

    pub fn compress(compression: &str, records: &[u8]) -> Vec<u8> {
        match compression {
            "" => records.to_vec(),
            "lz4" => {
                let mut enc = lz4_flex::frame::FrameEncoder::new(Vec::new());
                std::io::Write::write_all(&mut enc, records).unwrap();
                enc.finish().unwrap()
            }
            "zstd" => ruzstd::encoding::compress_to_vec(
                records,
                ruzstd::encoding::CompressionLevel::Fastest,
            ),
            other => panic!("no synthetic encoder for {other}"),
        }
    }

    /// Assemble a whole in-memory MCAP: magic, Header, chunks (with Schema / Channel records in the first), MessageIndex, DataEnd, Summary, Footer, magic.
    pub fn synthetic_mcap(spec: &SyntheticMcap) -> Vec<u8> {
        let mut channels: Vec<u16> = spec
            .chunks
            .iter()
            .flatten()
            .map(|(channel, _, _)| *channel)
            .collect();
        channels.sort_unstable();
        channels.dedup();
        let mut out = MAGIC.to_vec();
        let mut header = str_field("ros2");
        header.extend(str_field("visor-test"));
        out.extend(record(OP_HEADER, &header));
        let mut chunk_indexes = Vec::new();
        let mut sequence = 0u32;
        let mut total = 0u64;
        for (n, messages) in spec.chunks.iter().enumerate() {
            let mut records = Vec::new();
            if n == 0 {
                records.extend(schema_record());
                for channel in &channels {
                    records.extend(channel_record(*channel));
                }
            }
            // One MessageIndex per channel, exactly as the mcap writer emits them.
            let mut per_channel: BTreeMap<u16, Vec<(u64, u64)>> = BTreeMap::new();
            for (channel, log_time, payload) in messages {
                per_channel
                    .entry(*channel)
                    .or_default()
                    .push((*log_time, records.len() as u64));
                records.extend(message_record(*channel, *log_time, sequence, payload));
                sequence += 1;
                total += 1;
            }
            let start = messages.iter().map(|m| m.1).min().unwrap_or(0);
            let end = messages.iter().map(|m| m.1).max().unwrap_or(0);
            let stored = compress(&spec.compression, &records);
            let mut body = start.to_le_bytes().to_vec();
            body.extend_from_slice(&end.to_le_bytes());
            body.extend_from_slice(&(records.len() as u64).to_le_bytes());
            body.extend_from_slice(&0u32.to_le_bytes());
            body.extend(str_field(&spec.compression));
            body.extend_from_slice(&(stored.len() as u64).to_le_bytes());
            body.extend(stored);
            let chunk_start = out.len() as u64;
            let chunk_record = record(OP_CHUNK, &body);
            let chunk_len = chunk_record.len() as u64;
            out.extend(chunk_record);
            let mut index_offsets = Vec::new();
            let index_start = out.len() as u64;
            if spec.message_index {
                for (channel, entries) in &per_channel {
                    index_offsets.push((*channel, out.len() as u64));
                    let mut body = channel.to_le_bytes().to_vec();
                    body.extend_from_slice(&((entries.len() * 16) as u32).to_le_bytes());
                    for (time, offset) in entries {
                        body.extend_from_slice(&time.to_le_bytes());
                        body.extend_from_slice(&offset.to_le_bytes());
                    }
                    out.extend(record(OP_MESSAGE_INDEX, &body));
                }
            }
            let index_len = out.len() as u64 - index_start;
            chunk_indexes.push((
                start,
                end,
                chunk_start,
                chunk_len,
                index_offsets,
                index_len,
                records.len() as u64,
            ));
        }
        if spec.top_level_message {
            out.extend(message_record(channels[0], 1, sequence, b"loose"));
        }
        out.extend(record(OP_DATA_END, &0u32.to_le_bytes()));
        if !spec.summary {
            return out;
        }
        let summary_start = out.len() as u64;
        out.extend(schema_record());
        for channel in &channels {
            out.extend(channel_record(*channel));
        }
        let mut stats = total.to_le_bytes().to_vec();
        stats.extend_from_slice(&1u16.to_le_bytes());
        stats.extend_from_slice(&(channels.len() as u32).to_le_bytes());
        stats.extend_from_slice(&0u32.to_le_bytes());
        stats.extend_from_slice(&0u32.to_le_bytes());
        stats.extend_from_slice(&(spec.chunks.len() as u32).to_le_bytes());
        stats.extend_from_slice(&0u64.to_le_bytes());
        stats.extend_from_slice(&0u64.to_le_bytes());
        stats.extend_from_slice(&0u32.to_le_bytes());
        out.extend(record(OP_STATISTICS, &stats));
        for (start, end, chunk_start, chunk_len, offsets, index_len, uncompressed) in &chunk_indexes
        {
            let mut body = start.to_le_bytes().to_vec();
            body.extend_from_slice(&end.to_le_bytes());
            body.extend_from_slice(&chunk_start.to_le_bytes());
            body.extend_from_slice(&chunk_len.to_le_bytes());
            body.extend_from_slice(&((offsets.len() * 10) as u32).to_le_bytes());
            for (channel, offset) in offsets {
                body.extend_from_slice(&channel.to_le_bytes());
                body.extend_from_slice(&offset.to_le_bytes());
            }
            body.extend_from_slice(&index_len.to_le_bytes());
            body.extend(str_field(&spec.compression));
            body.extend_from_slice(&(chunk_len - RECORD_PREFIX).to_le_bytes());
            body.extend_from_slice(&uncompressed.to_le_bytes());
            out.extend(record(OP_CHUNK_INDEX, &body));
        }
        let mut footer = summary_start.to_le_bytes().to_vec();
        footer.extend_from_slice(&0u64.to_le_bytes());
        footer.extend_from_slice(&0u32.to_le_bytes());
        out.extend(record(OP_FOOTER, &footer));
        out.extend_from_slice(&MAGIC);
        out
    }

    /// Bytes from the DataEnd record's end to the end of the file: what a crash before the summary loses.
    pub fn summary_len(bytes: &[u8]) -> usize {
        let mut at = MAGIC.len();
        while at < bytes.len() {
            let (op, body) = record_at(bytes, at).unwrap();
            at += RECORD_PREFIX as usize + body.len();
            if op == OP_DATA_END {
                return bytes.len() - at;
            }
        }
        0
    }

    /// Byte offset of the `n`th top-level record with opcode `op`.
    pub fn offset_of(bytes: &[u8], op: u8, n: usize) -> usize {
        let mut at = MAGIC.len();
        let mut seen = 0;
        while at < bytes.len() {
            let (found, body) = record_at(bytes, at).unwrap();
            if found == op {
                if seen == n {
                    return at;
                }
                seen += 1;
            }
            at += RECORD_PREFIX as usize + body.len();
        }
        panic!("no record 0x{op:02x} #{n}");
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use crate::bag::reader::test_support::write_temp;

    fn no_cancel() -> AtomicBool {
        AtomicBool::new(false)
    }

    fn three_messages() -> Vec<Vec<(u16, u64, Vec<u8>)>> {
        vec![
            vec![(1, 100, b"first".to_vec()), (2, 101, b"second".to_vec())],
            vec![(1, 102, b"third".to_vec())],
        ]
    }

    fn payloads(reader: &mut McapReader, index: &[IndexEntry]) -> Vec<Vec<u8>> {
        let mut sorted = index.to_vec();
        sorted.sort_by_key(|e| e.time);
        sorted
            .iter()
            .map(|e| reader.message_at(e).unwrap().data.to_vec())
            .collect()
    }

    #[test]
    fn synthetic_mcap_reads_end_to_end_with_every_codec() {
        for compression in ["", "lz4", "zstd"] {
            let bytes = synthetic_mcap(&SyntheticMcap {
                compression: compression.to_owned(),
                chunks: three_messages(),
                ..SyntheticMcap::default()
            });
            let path = write_temp(&format!("codec_{compression}.mcap"), &bytes);
            let mut reader = McapReader::open(&path, 2, &no_cancel()).unwrap();
            assert!(reader.warnings().is_empty(), "{:?}", reader.warnings());
            assert_eq!(reader.chunk_count(), 2);
            assert_eq!(reader.message_count(), 3);
            assert_eq!(reader.start(), 100);
            assert_eq!(
                reader.compression(),
                Some(if compression.is_empty() {
                    "none"
                } else {
                    compression
                })
            );
            let conns = reader.connections();
            assert_eq!(conns.len(), 2);
            assert_eq!(conns[0].id, 1);
            assert_eq!(conns[0].topic_raw, "/topic1");
            assert_eq!(conns[0].type_raw, SCHEMA_NAME);
            assert_eq!(conns[0].type_hash, "RIHS01_test");
            assert_eq!(conns[0].definition, SCHEMA_TEXT);
            assert_eq!(conns[0].message_encoding, "cdr");
            assert_eq!(conns[0].definition_encoding, "ros2msg");
            let index = reader.read_message_index(&no_cancel()).unwrap();
            assert_eq!(index.len(), 3);
            assert_eq!(
                index[0],
                IndexEntry {
                    time: 100,
                    conn: 1,
                    file: 0,
                    chunk: 0,
                    offset: index[0].offset
                }
            );
            assert_eq!(index[2].chunk, 1);
            let message = reader.message_at(&index[1]).unwrap();
            assert_eq!(message.conn, 2);
            assert_eq!(message.time, 101);
            assert_eq!(
                payloads(&mut reader, &index),
                vec![b"first".to_vec(), b"second".to_vec(), b"third".to_vec()]
            );
            std::fs::remove_file(&path).ok();
        }
    }

    #[test]
    fn a_file_cut_before_the_summary_reads_the_same_index_with_a_warning() {
        let full = synthetic_mcap(&SyntheticMcap {
            compression: "lz4".to_owned(),
            chunks: three_messages(),
            ..SyntheticMcap::default()
        });
        let mut torn = full.clone();
        torn.truncate(full.len() - summary_len(&full));
        let path = write_temp("torn_summary.mcap", &torn);
        let mut reader = McapReader::open(&path, 2, &no_cancel()).unwrap();
        // Channels came out of the chunks, so the topic list is complete.
        assert_eq!(reader.connections().len(), 2);
        assert!(
            reader.warnings().iter().any(|w| w.contains("no footer")),
            "{:?}",
            reader.warnings()
        );
        assert!(
            reader
                .warnings()
                .iter()
                .any(|w| w.contains("scanned 2 chunk")),
            "{:?}",
            reader.warnings()
        );
        let index = reader.read_message_index(&no_cancel()).unwrap();
        assert_eq!(index.len(), 3);
        // Without Statistics the count comes from the index itself.
        assert_eq!(reader.message_count(), 3);
        assert_eq!(
            payloads(&mut reader, &index),
            vec![b"first".to_vec(), b"second".to_vec(), b"third".to_vec()]
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_file_cut_inside_a_message_index_still_yields_every_message_of_that_chunk() {
        // One chunk, two channels, so the writer emits two MessageIndex records; cut inside the second one.
        let full = synthetic_mcap(&SyntheticMcap {
            chunks: vec![vec![
                (1, 100, b"a".to_vec()),
                (2, 101, b"b".to_vec()),
                (1, 102, b"c".to_vec()),
            ]],
            ..SyntheticMcap::default()
        });
        let second_index = offset_of(&full, OP_MESSAGE_INDEX, 1);
        let mut torn = full.clone();
        torn.truncate(second_index + 12);
        let path = write_temp("torn_index.mcap", &torn);
        let mut reader = McapReader::open(&path, 2, &no_cancel()).unwrap();
        assert!(
            reader
                .warnings()
                .iter()
                .any(|w| w.contains("message index may be incomplete")),
            "{:?}",
            reader.warnings()
        );
        let index = reader.read_message_index(&no_cancel()).unwrap();
        // The complete first MessageIndex alone would have given two of the three; the chunk scan gives all of them.
        assert_eq!(index.len(), 3);
        assert_eq!(
            payloads(&mut reader, &index),
            vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn an_absurd_uncompressed_size_is_refused_before_anything_is_allocated() {
        let mut bytes = synthetic_mcap(&SyntheticMcap {
            compression: "lz4".to_owned(),
            chunks: three_messages(),
            ..SyntheticMcap::default()
        });
        let at = offset_of(&bytes, OP_CHUNK, 0) + RECORD_PREFIX as usize + 16;
        bytes[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        let path = write_temp("hugesize.mcap", &bytes);
        assert!(matches!(
            McapReader::open(&path, 2, &no_cancel()),
            Err(BagError::Malformed(_))
        ));
        std::fs::remove_file(&path).ok();
        // The decoder path has the same guard, and an oversized stream is cut off one byte past the declared size instead of being decoded in full.
        assert!(matches!(
            decompress_chunk("lz4", b"x", u64::MAX),
            Err(BagError::Decompress(_))
        ));
        let frame = compress("lz4", &[7u8; 4096]);
        assert!(matches!(
            decompress_chunk("lz4", &frame, 16),
            Err(BagError::Decompress(_))
        ));
    }

    #[test]
    fn a_file_cut_inside_the_last_chunk_drops_that_chunk_only() {
        let full = synthetic_mcap(&SyntheticMcap {
            chunks: three_messages(),
            ..SyntheticMcap::default()
        });
        let second_chunk = offset_of(&full, OP_CHUNK, 1);
        let mut torn = full.clone();
        torn.truncate(second_chunk + 20);
        let path = write_temp("torn_chunk.mcap", &torn);
        let mut reader = McapReader::open(&path, 2, &no_cancel()).unwrap();
        assert_eq!(reader.chunk_count(), 1);
        assert!(
            reader.warnings().iter().any(|w| w.contains("torn tail")),
            "{:?}",
            reader.warnings()
        );
        let index = reader.read_message_index(&no_cancel()).unwrap();
        assert_eq!(index.len(), 2);
        assert_eq!(
            payloads(&mut reader, &index),
            vec![b"first".to_vec(), b"second".to_vec()]
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn chunks_without_a_message_index_are_scanned() {
        let bytes = synthetic_mcap(&SyntheticMcap {
            compression: "zstd".to_owned(),
            chunks: three_messages(),
            message_index: false,
            ..SyntheticMcap::default()
        });
        let path = write_temp("no_index.mcap", &bytes);
        let mut reader = McapReader::open(&path, 2, &no_cancel()).unwrap();
        assert!(reader.warnings().is_empty(), "{:?}", reader.warnings());
        let index = reader.read_message_index(&no_cancel()).unwrap();
        assert_eq!(index.len(), 3);
        assert_eq!(
            payloads(&mut reader, &index),
            vec![b"first".to_vec(), b"second".to_vec(), b"third".to_vec()]
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_statistics_mismatch_is_a_warning_not_a_failure() {
        let mut bytes = synthetic_mcap(&SyntheticMcap {
            chunks: three_messages(),
            ..SyntheticMcap::default()
        });
        let stats = offset_of(&bytes, OP_STATISTICS, 0) + RECORD_PREFIX as usize;
        bytes[stats..stats + 8].copy_from_slice(&9u64.to_le_bytes());
        let path = write_temp("stats.mcap", &bytes);
        let mut reader = McapReader::open(&path, 2, &no_cancel()).unwrap();
        let index = reader.read_message_index(&no_cancel()).unwrap();
        assert_eq!(index.len(), 3);
        assert!(
            reader
                .warnings()
                .iter()
                .any(|w| w.contains("statistics say 9")),
            "{:?}",
            reader.warnings()
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn corruption_is_rejected_with_a_distinct_error_each() {
        let clean = synthetic_mcap(&SyntheticMcap {
            chunks: three_messages(),
            ..SyntheticMcap::default()
        });
        // Not an MCAP at all.
        let path = write_temp("not.mcap", b"#ROSBAG V2.0\nrest");
        assert!(matches!(
            McapReader::open(&path, 2, &no_cancel()),
            Err(BagError::UnsupportedVersion(_))
        ));
        std::fs::remove_file(&path).ok();
        // An opcode nothing defines, at top level.
        let mut bytes = clean.clone();
        let at = offset_of(&bytes, OP_DATA_END, 0);
        bytes[at] = 0x7F;
        let path = write_temp("badop.mcap", &bytes);
        assert!(matches!(
            McapReader::open(&path, 2, &no_cancel()),
            Err(BagError::Malformed(_))
        ));
        std::fs::remove_file(&path).ok();
        // A chunk whose uncompressed_size disagrees with what comes out.
        let mut bytes = clean.clone();
        let at = offset_of(&bytes, OP_CHUNK, 0) + RECORD_PREFIX as usize + 16;
        bytes[at..at + 8].copy_from_slice(&5u64.to_le_bytes());
        let path = write_temp("badsize.mcap", &bytes);
        let mut reader = McapReader::open(&path, 2, &no_cancel()).unwrap();
        let index = reader.read_message_index(&no_cancel()).unwrap();
        assert!(matches!(
            reader.message_at(&index[0]),
            Err(BagError::Decompress(_))
        ));
        std::fs::remove_file(&path).ok();
        // An index entry pointing at the Schema record that opens the chunk.
        let path = write_temp("badoffset.mcap", &clean);
        let mut reader = McapReader::open(&path, 2, &no_cancel()).unwrap();
        let index = reader.read_message_index(&no_cancel()).unwrap();
        let wrong = IndexEntry {
            offset: 0,
            ..index[0]
        };
        assert!(matches!(
            reader.message_at(&wrong),
            Err(BagError::Malformed(_))
        ));
        std::fs::remove_file(&path).ok();
        // A top-level Message, even with a complete Summary behind it.
        let bytes = synthetic_mcap(&SyntheticMcap {
            chunks: three_messages(),
            top_level_message: true,
            ..SyntheticMcap::default()
        });
        let path = write_temp("unchunked.mcap", &bytes);
        assert!(matches!(
            McapReader::open(&path, 2, &no_cancel()),
            Err(BagError::Unsupported(_))
        ));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn zstd_frames_from_the_reference_tool_decompress() {
        let frame = include_bytes!("../../tests/fixtures/rosbag2/hello.zst");
        let text = b"hello mcap zstd\n";
        assert_eq!(
            decompress_chunk("zstd", frame, text.len() as u64).unwrap(),
            text
        );
        assert!(matches!(
            decompress_chunk("zstd", frame, 3),
            Err(BagError::Decompress(_))
        ));
        assert!(matches!(
            decompress_chunk("zstd", b"not a frame", 3),
            Err(BagError::Decompress(_))
        ));
    }

    #[test]
    fn an_unknown_compression_is_refused_by_name() {
        assert!(matches!(
            decompress_chunk("bz2", b"x", 1),
            Err(BagError::UnsupportedCompression(_))
        ));
        let mut bytes = synthetic_mcap(&SyntheticMcap {
            compression: "lz4".to_owned(),
            chunks: three_messages(),
            ..SyntheticMcap::default()
        });
        let at =
            offset_of(&bytes, OP_CHUNK, 0) + RECORD_PREFIX as usize + CHUNK_FIXED_HEAD as usize + 4;
        bytes[at..at + 3].copy_from_slice(b"bz2");
        let path = write_temp("bz2.mcap", &bytes);
        let mut reader = McapReader::open(&path, 2, &no_cancel()).unwrap();
        let index = reader.read_message_index(&no_cancel()).unwrap();
        assert!(matches!(
            reader.message_at(&index[0]),
            Err(BagError::UnsupportedCompression(_))
        ));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_raised_cancel_flag_stops_opening_and_indexing_before_any_chunk_is_read() {
        let bytes = synthetic_mcap(&SyntheticMcap {
            chunks: three_messages(),
            message_index: false,
            ..SyntheticMcap::default()
        });
        let path = write_temp("cancel.mcap", &bytes);
        let cancelled = AtomicBool::new(true);
        assert!(matches!(
            open_storage(&path, 2, &cancelled),
            Err(BagError::Cancelled)
        ));
        let mut reader = McapReader::open(&path, 2, &no_cancel()).unwrap();
        assert!(matches!(
            reader.read_message_index(&cancelled),
            Err(BagError::Cancelled)
        ));
        assert!(reader.cache.entries.is_empty());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn cursor_reports_overruns_instead_of_panicking() {
        let mut cursor = Cursor::new(&[1, 0, 2, 0, 0, 0]);
        assert_eq!(cursor.u16().unwrap(), 1);
        assert!(matches!(cursor.u64(), Err(BagError::Malformed(_))));
        assert!(matches!(
            record_at(&[OP_MESSAGE, 9, 0, 0, 0, 0, 0, 0, 0, 1], 0),
            Err(BagError::Malformed(_))
        ));
    }
}
