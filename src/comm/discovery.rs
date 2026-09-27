//! Enumerate nodes/topics/types via liveliness token subscription (aggregation lives in the I/O-free `GraphState`).

use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use crossbeam_channel::Sender;
use zenoh::Session;
use zenoh::sample::SampleKind;

use super::keyexpr::{EntityKind, dds_type_to_ros, parse_liveliness_token};
use super::session::{Notify, TopicRow};

/// Timeout for enumerating existing tokens via liveliness get (same as probe).
const LIVELINESS_GET_TIMEOUT: Duration = Duration::from_secs(3);

struct TopicEntity {
    topic: String,
    type_name_dds: String,
    type_hash: String,
    kind: EntityKind,
}

/// Graph state of liveliness tokens (dedup by token key, holding pub/sub entities).
#[derive(Default)]
pub struct GraphState {
    tokens: HashMap<String, TopicEntity>,
}

impl GraphState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a token put. Returns true only if the topic list changed (false for NN/SS/SC, unparsable, or duplicate).
    pub fn apply_put(&mut self, token_key: &str) -> bool {
        if self.tokens.contains_key(token_key) {
            return false;
        }
        let Ok(token) = parse_liveliness_token(token_key) else {
            return false;
        };
        if !matches!(token.kind, EntityKind::Publisher | EntityKind::Subscription) {
            return false;
        }
        let Some(info) = token.topic else {
            return false;
        };
        self.tokens.insert(
            token_key.to_owned(),
            TopicEntity {
                topic: info.name,
                type_name_dds: info.type_name,
                type_hash: info.type_hash,
                kind: token.kind,
            },
        );
        true
    }

    /// Apply a token delete. Returns true only if the topic list changed.
    pub fn apply_delete(&mut self, token_key: &str) -> bool {
        self.tokens.remove(token_key).is_some()
    }

    /// Return the list aggregated by (topic name, type), sorted by topic name.
    pub fn snapshot(&self) -> Vec<TopicRow> {
        let mut rows: BTreeMap<(String, String), TopicRow> = BTreeMap::new();
        for entity in self.tokens.values() {
            let row = rows
                .entry((entity.topic.clone(), entity.type_name_dds.clone()))
                .or_insert_with(|| TopicRow {
                    name: entity.topic.clone(),
                    ros_type: dds_type_to_ros(&entity.type_name_dds)
                        .unwrap_or_else(|_| entity.type_name_dds.clone()),
                    type_name_dds: entity.type_name_dds.clone(),
                    type_hash: entity.type_hash.clone(),
                    publisher_count: 0,
                    subscriber_count: 0,
                });
            match entity.kind {
                EntityKind::Publisher => row.publisher_count += 1,
                EntityKind::Subscription => row.subscriber_count += 1,
                _ => {}
            }
        }
        rows.into_values().collect()
    }
}

/// Declare a liveliness subscriber, get existing tokens, then subscribe to changes, sending a snapshot on every change.
pub async fn run_discovery(
    session: Session,
    domain_id: u32,
    graph_tx: Sender<Vec<TopicRow>>,
    notify: Notify,
) {
    let key = format!("@ros2_lv/{domain_id}/**");
    let sub = match session.liveliness().declare_subscriber(&key).await {
        Ok(sub) => sub,
        Err(e) => {
            eprintln!("visor: failed to declare liveliness subscriber `{key}`: {e}");
            return;
        }
    };
    let mut state = GraphState::new();
    match session
        .liveliness()
        .get(&key)
        .timeout(LIVELINESS_GET_TIMEOUT)
        .await
    {
        Ok(replies) => {
            while let Ok(reply) = replies.recv_async().await {
                if let Ok(sample) = reply.result()
                    && state.apply_put(sample.key_expr().as_str())
                {
                    send_snapshot(&state, &graph_tx, &notify);
                }
            }
        }
        Err(e) => eprintln!("visor: liveliness get failed: {e}"),
    }
    while let Ok(sample) = sub.recv_async().await {
        let token_key = sample.key_expr().as_str();
        let changed = match sample.kind() {
            SampleKind::Put => state.apply_put(token_key),
            SampleKind::Delete => state.apply_delete(token_key),
        };
        if changed {
            send_snapshot(&state, &graph_tx, &notify);
        }
    }
}

fn send_snapshot(state: &GraphState, graph_tx: &Sender<Vec<TopicRow>>, notify: &Notify) {
    let _ = graph_tx.send(state.snapshot());
    notify();
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZID_A: &str = "aaaabbbbccccddddeeeeffff00001111";
    const ZID_B: &str = "11110000ffffeeeeddddccccbbbbaaaa";

    fn mp_chatter(zid: &str, id: u32) -> String {
        format!(
            "@ros2_lv/0/{zid}/0/{id}/MP/%/%/talker/%chatter/std_msgs::msg::dds_::String_/RIHS01_df66/::,:,:,,"
        )
    }

    fn ms_chatter(zid: &str, id: u32) -> String {
        format!(
            "@ros2_lv/0/{zid}/0/{id}/MS/%/%/listener/%chatter/std_msgs::msg::dds_::String_/RIHS01_df66/::,:,:,,"
        )
    }

    #[test]
    fn single_publisher_becomes_one_row() {
        let mut state = GraphState::new();
        assert!(state.apply_put(&mp_chatter(ZID_A, 10)));
        let rows = state.snapshot();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "/chatter");
        assert_eq!(rows[0].ros_type, "std_msgs/msg/String");
        assert_eq!(rows[0].type_name_dds, "std_msgs::msg::dds_::String_");
        assert_eq!(rows[0].publisher_count, 1);
        assert_eq!(rows[0].subscriber_count, 0);
    }

    #[test]
    fn multiple_entities_aggregate_into_one_row() {
        let mut state = GraphState::new();
        assert!(state.apply_put(&mp_chatter(ZID_A, 10)));
        assert!(state.apply_put(&mp_chatter(ZID_B, 11)));
        assert!(state.apply_put(&ms_chatter(ZID_B, 12)));
        let rows = state.snapshot();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].publisher_count, 2);
        assert_eq!(rows[0].subscriber_count, 1);
    }

    #[test]
    fn duplicate_put_is_not_double_counted() {
        let mut state = GraphState::new();
        assert!(state.apply_put(&mp_chatter(ZID_A, 10)));
        assert!(!state.apply_put(&mp_chatter(ZID_A, 10)));
        assert_eq!(state.snapshot()[0].publisher_count, 1);
    }

    #[test]
    fn partial_delete_keeps_row_and_full_delete_removes_it() {
        let mut state = GraphState::new();
        state.apply_put(&mp_chatter(ZID_A, 10));
        state.apply_put(&ms_chatter(ZID_B, 12));
        assert!(state.apply_delete(&mp_chatter(ZID_A, 10)));
        let rows = state.snapshot();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].publisher_count, 0);
        assert_eq!(rows[0].subscriber_count, 1);
        assert!(state.apply_delete(&ms_chatter(ZID_B, 12)));
        assert!(state.snapshot().is_empty());
        assert!(!state.apply_delete(&ms_chatter(ZID_B, 12)));
    }

    #[test]
    fn node_service_client_tokens_are_ignored() {
        let mut state = GraphState::new();
        assert!(!state.apply_put(&format!("@ros2_lv/0/{ZID_A}/0/0/NN/%/%/talker")));
        assert!(!state.apply_put(&format!(
            "@ros2_lv/0/{ZID_A}/0/20/SS/%/%/adder/%add_two_ints/example_interfaces::srv::dds_::AddTwoInts_Request_/RIHS01_zzz/::,:,:,,"
        )));
        assert!(!state.apply_put(&format!(
            "@ros2_lv/0/{ZID_A}/0/21/SC/%/%/caller/%add_two_ints/example_interfaces::srv::dds_::AddTwoInts_Request_/RIHS01_zzz/::,:,:,,"
        )));
        assert!(state.snapshot().is_empty());
    }

    #[test]
    fn unparsable_token_is_ignored() {
        let mut state = GraphState::new();
        assert!(!state.apply_put("not/a/liveliness/token"));
        assert!(!state.apply_put(""));
        assert!(state.snapshot().is_empty());
    }

    #[test]
    fn unconvertible_type_name_falls_back_to_dds_name() {
        let mut state = GraphState::new();
        assert!(state.apply_put(&format!(
            "@ros2_lv/0/{ZID_A}/0/30/MP/%/%/weird/%odd/BrokenTypeName/RIHS01_www/::,:,:,,"
        )));
        let rows = state.snapshot();
        assert_eq!(rows[0].ros_type, "BrokenTypeName");
        assert_eq!(rows[0].type_name_dds, "BrokenTypeName");
    }

    #[test]
    fn snapshot_is_sorted_by_topic_name() {
        let mut state = GraphState::new();
        state.apply_put(&format!(
            "@ros2_lv/0/{ZID_A}/0/40/MP/%/%/n/%tf/tf2_msgs::msg::dds_::TFMessage_/RIHS01_a/::,:,:,,"
        ));
        state.apply_put(&mp_chatter(ZID_A, 41));
        state.apply_put(&format!(
            "@ros2_lv/0/{ZID_A}/0/42/MP/%/%/n/%amap/nav_msgs::msg::dds_::OccupancyGrid_/RIHS01_b/::,:,:,,"
        ));
        let rows = state.snapshot();
        let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, ["/amap", "/chatter", "/tf"]);
    }

    #[test]
    fn subscriber_only_topic_is_listed() {
        let mut state = GraphState::new();
        assert!(state.apply_put(&ms_chatter(ZID_A, 50)));
        let rows = state.snapshot();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].publisher_count, 0);
        assert_eq!(rows[0].subscriber_count, 1);
    }
}
