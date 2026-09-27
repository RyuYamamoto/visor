//! A plugin-supplied .msg definition must reach the live decode path (AC-2), verified without a ROS environment.

use visor::decode::cdr::decode_message;
use visor::decode::value::Value;
use visor::plugin::registry::Registry;
use visor_plugin_sample::{FLEET_STATE_TYPE, SamplePlugin};

/// Minimal CDR_LE writer: tracks the offset so each field lands on its natural alignment, as the wire format requires.
#[derive(Default)]
struct Cdr {
    body: Vec<u8>,
}

impl Cdr {
    fn align(&mut self, to: usize) {
        while !self.body.len().is_multiple_of(to) {
            self.body.push(0);
        }
    }

    fn u32(&mut self, v: u32) -> &mut Self {
        self.align(4);
        self.body.extend_from_slice(&v.to_le_bytes());
        self
    }

    fn i32(&mut self, v: i32) -> &mut Self {
        self.align(4);
        self.body.extend_from_slice(&v.to_le_bytes());
        self
    }

    fn f32(&mut self, v: f32) -> &mut Self {
        self.align(4);
        self.body.extend_from_slice(&v.to_le_bytes());
        self
    }

    fn f64(&mut self, v: f64) -> &mut Self {
        self.align(8);
        self.body.extend_from_slice(&v.to_le_bytes());
        self
    }

    /// Length-prefixed, NUL-terminated string (the length counts the NUL).
    fn string(&mut self, s: &str) -> &mut Self {
        self.u32(s.len() as u32 + 1);
        self.body.extend_from_slice(s.as_bytes());
        self.body.push(0);
        self
    }

    /// Prepend the 4-byte encapsulation header rmw_zenoh payloads carry.
    fn finish(&self) -> Vec<u8> {
        let mut out = vec![0x00, 0x01, 0x00, 0x00];
        out.extend_from_slice(&self.body);
        out
    }
}

/// One FleetState with a single robot at (1, 2, 3), identity orientation, 55.5% battery.
fn fleet_state_payload() -> Vec<u8> {
    let mut cdr = Cdr::default();
    cdr.i32(7).u32(8).string("map");
    cdr.u32(1);
    cdr.string("amr_1");
    cdr.f64(1.0).f64(2.0).f64(3.0);
    cdr.f64(0.0).f64(0.0).f64(0.0).f64(1.0);
    cdr.f32(55.5);
    cdr.finish()
}

/// The registry a build with the sample plugin produces.
fn registry_with_sample() -> std::sync::Arc<Registry> {
    let mut registry = Registry::builtin();
    registry.add_plugin(&SamplePlugin::default());
    registry.finish(&|_| None)
}

#[test]
fn a_builtin_only_build_cannot_decode_the_plugin_type() {
    let registry = Registry::builtin().finish(&|_| None);
    let (types, problems) = registry.build_type_registry();
    assert!(problems.is_empty(), "{problems:?}");
    assert!(types.get(FLEET_STATE_TYPE).is_none());
    assert!(decode_message(&types, FLEET_STATE_TYPE, &fleet_state_payload()).is_err());
}

#[test]
fn a_plugin_definition_decodes_a_real_cdr_payload_of_its_own_type() {
    let registry = registry_with_sample();
    let (types, problems) = registry.build_type_registry();
    assert!(problems.is_empty(), "{problems:?}");
    let value = decode_message(&types, FLEET_STATE_TYPE, &fleet_state_payload())
        .expect("plugin definition decodes its own payload");
    let Some(Value::Array(robots)) = value.get("robots") else {
        panic!("robots should decode as a sequence: {value:?}");
    };
    assert_eq!(robots.len(), 1);
    assert_eq!(
        robots[0].get("name"),
        Some(&Value::String("amr_1".to_owned()))
    );
    assert_eq!(robots[0].get("battery"), Some(&Value::F32(55.5)));
    let pose = robots[0].get("pose").expect("pose");
    assert_eq!(
        pose.get("position").and_then(|p| p.get("y")),
        Some(&Value::F64(2.0))
    );
    // The plugin's type references std_msgs/Header, which only the builtin set provides.
    assert_eq!(
        value.get("header").and_then(|h| h.get("frame_id")),
        Some(&Value::String("map".to_owned()))
    );
}

#[test]
fn the_plugin_renderer_consumes_what_the_decoder_produced() {
    let registry = registry_with_sample();
    let (types, _) = registry.build_type_registry();
    let value = decode_message(&types, FLEET_STATE_TYPE, &fleet_state_payload()).expect("decodes");
    let entry = registry
        .find_renderer("/fleet_state", FLEET_STATE_TYPE)
        .expect("the plugin registered a renderer for its own type");
    assert_eq!(entry.key, "sample::FleetState");
    let mut renderer = entry.make().expect("factory");
    renderer.on_message(&value);
    // With no TF samples the map frame resolves to itself, so this bakes rather than reporting TfUnavailable.
    let buffer = visor::tf::buffer::TfBuffer::new();
    let tf = visor::plugin::TfContext {
        buffer: &buffer,
        fixed_frame: "map",
    };
    let batches = renderer.scene(&tf).expect("baked");
    assert!(!batches.is_empty());
}
