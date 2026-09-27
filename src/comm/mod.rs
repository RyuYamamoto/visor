pub mod discovery;
pub mod keyexpr;
pub mod session;

/// Default zenoh router endpoint (works with no args on a local Linux sim setup).
pub const DEFAULT_ENDPOINT: &str = "tcp/localhost:7447";

/// zenoh connection config, shared by the GUI and probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommConfig {
    pub endpoint: String,
    pub domain_id: u32,
}

impl CommConfig {
    /// Resolve config in order: CLI args, then env (injected as a closure to isolate tests from the process env), then defaults.
    pub fn resolve<I, F>(args: I, env: F) -> Result<Self, String>
    where
        I: IntoIterator<Item = String>,
        F: Fn(&str) -> Option<String>,
    {
        let mut endpoint: Option<String> = None;
        let mut domain_id: Option<String> = None;
        let mut it = args.into_iter();
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--endpoint" => {
                    endpoint = Some(it.next().ok_or("--endpoint requires a value")?);
                }
                "--domain-id" => {
                    domain_id = Some(it.next().ok_or("--domain-id requires a value")?);
                }
                other => return Err(format!("unknown argument `{other}`")),
            }
        }
        let endpoint = endpoint
            .or_else(|| env("ZENOH_ENDPOINT"))
            .or_else(|| {
                env("ROS_STATIC_PEERS")
                    .as_deref()
                    .and_then(endpoint_from_static_peers)
            })
            .unwrap_or_else(|| DEFAULT_ENDPOINT.to_owned());
        let domain_id = match domain_id.or_else(|| env("ROS_DOMAIN_ID")) {
            Some(s) => s
                .parse::<u32>()
                .map_err(|_| format!("invalid domain id `{s}`"))?,
            None => 0,
        };
        Ok(Self {
            endpoint,
            domain_id,
        })
    }
}

/// Normalize the first entry of ROS_STATIC_PEERS (`;`-separated, set by robot-connection shell functions) into a zenoh endpoint.
fn endpoint_from_static_peers(value: &str) -> Option<String> {
    let first = value.split(';').map(str::trim).find(|s| !s.is_empty())?;
    if first.contains('/') {
        Some(first.to_owned())
    } else if first.contains(':') {
        // Also accept the DDS-style `host:port` form.
        Some(format!("tcp/{first}"))
    } else {
        Some(format!("tcp/{first}:7447"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn defaults_when_nothing_given() {
        let c = CommConfig::resolve(args(&[]), no_env).unwrap();
        assert_eq!(c.endpoint, DEFAULT_ENDPOINT);
        assert_eq!(c.domain_id, 0);
    }

    #[test]
    fn env_overrides_default() {
        let env = |key: &str| match key {
            "ZENOH_ENDPOINT" => Some("tcp/robot:7447".to_owned()),
            "ROS_DOMAIN_ID" => Some("7".to_owned()),
            _ => None,
        };
        let c = CommConfig::resolve(args(&[]), env).unwrap();
        assert_eq!(c.endpoint, "tcp/robot:7447");
        assert_eq!(c.domain_id, 7);
    }

    #[test]
    fn cli_overrides_env() {
        let env = |key: &str| match key {
            "ZENOH_ENDPOINT" => Some("tcp/robot:7447".to_owned()),
            "ROS_DOMAIN_ID" => Some("7".to_owned()),
            _ => None,
        };
        let c = CommConfig::resolve(
            args(&["--endpoint", "tcp/sim:7447", "--domain-id", "3"]),
            env,
        )
        .unwrap();
        assert_eq!(c.endpoint, "tcp/sim:7447");
        assert_eq!(c.domain_id, 3);
    }

    #[test]
    fn static_peers_used_when_zenoh_endpoint_missing() {
        // Some shell setups set ROS_STATIC_PEERS, not ZENOH_ENDPOINT.
        let env = |key: &str| match key {
            "ROS_STATIC_PEERS" => Some("tcp/192.0.2.10:7447".to_owned()),
            "ROS_DOMAIN_ID" => Some("32".to_owned()),
            _ => None,
        };
        let c = CommConfig::resolve(args(&[]), env).unwrap();
        assert_eq!(c.endpoint, "tcp/192.0.2.10:7447");
        assert_eq!(c.domain_id, 32);
    }

    #[test]
    fn zenoh_endpoint_wins_over_static_peers() {
        let env = |key: &str| match key {
            "ZENOH_ENDPOINT" => Some("tcp/sim:7447".to_owned()),
            "ROS_STATIC_PEERS" => Some("tcp/robot:7447".to_owned()),
            _ => None,
        };
        let c = CommConfig::resolve(args(&[]), env).unwrap();
        assert_eq!(c.endpoint, "tcp/sim:7447");
    }

    #[test]
    fn static_peers_normalization() {
        assert_eq!(
            endpoint_from_static_peers("tcp/robot:7447"),
            Some("tcp/robot:7447".to_owned())
        );
        assert_eq!(
            endpoint_from_static_peers("192.168.1.10:8447"),
            Some("tcp/192.168.1.10:8447".to_owned())
        );
        assert_eq!(
            endpoint_from_static_peers("robot.local"),
            Some("tcp/robot.local:7447".to_owned())
        );
        assert_eq!(
            endpoint_from_static_peers("tcp/a:7447;tcp/b:7447"),
            Some("tcp/a:7447".to_owned())
        );
        assert_eq!(endpoint_from_static_peers("  ;  "), None);
        assert_eq!(endpoint_from_static_peers(""), None);
    }

    #[test]
    fn rejects_invalid_domain_id() {
        assert!(CommConfig::resolve(args(&["--domain-id", "abc"]), no_env).is_err());
        let env = |key: &str| (key == "ROS_DOMAIN_ID").then(|| "-1".to_owned());
        assert!(CommConfig::resolve(args(&[]), env).is_err());
    }

    #[test]
    fn rejects_missing_value_and_unknown_argument() {
        assert!(CommConfig::resolve(args(&["--endpoint"]), no_env).is_err());
        assert!(CommConfig::resolve(args(&["--domain-id"]), no_env).is_err());
        assert!(CommConfig::resolve(args(&["--bogus"]), no_env).is_err());
    }
}
