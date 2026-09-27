//! Regression tests over real-environment captures (taken 2026-07-21 with rmw_zenoh_cpp 0.2.4 + ros2 topic pub)

use visor::decode::cdr::decode_message;
use visor::decode::msg_parser::TypeRegistry;
use visor::decode::value::Value;

fn f(name: &str, v: Value) -> (String, Value) {
    (name.to_owned(), v)
}

#[test]
fn decodes_real_chatter_payload() {
    let payload = include_bytes!("fixtures/chatter_le.bin");
    let reg = TypeRegistry::with_embedded().unwrap();
    let v = decode_message(&reg, "std_msgs/msg/String", payload).unwrap();
    assert_eq!(
        v,
        Value::Struct(vec![f(
            "data",
            Value::String("hello ros2_viewer".to_owned())
        )])
    );
}

#[test]
fn decodes_real_tf_message_payload() {
    let payload = include_bytes!("fixtures/tf_message_le.bin");
    let reg = TypeRegistry::with_embedded().unwrap();
    let v = decode_message(&reg, "tf2_msgs/msg/TFMessage", payload).unwrap();
    let expected = Value::Struct(vec![f(
        "transforms",
        Value::Array(vec![Value::Struct(vec![
            f(
                "header",
                Value::Struct(vec![
                    f(
                        "stamp",
                        Value::Struct(vec![f("sec", Value::I32(100)), f("nanosec", Value::U32(5))]),
                    ),
                    f("frame_id", Value::String("map".to_owned())),
                ]),
            ),
            f("child_frame_id", Value::String("base_link".to_owned())),
            f(
                "transform",
                Value::Struct(vec![
                    f(
                        "translation",
                        Value::Struct(vec![
                            f("x", Value::F64(1.5)),
                            f("y", Value::F64(-0.5)),
                            f("z", Value::F64(0.0)),
                        ]),
                    ),
                    f(
                        "rotation",
                        Value::Struct(vec![
                            f("x", Value::F64(0.0)),
                            f("y", Value::F64(0.0)),
                            f("z", Value::F64(0.0)),
                            f("w", Value::F64(1.0)),
                        ]),
                    ),
                ]),
            ),
        ])]),
    )]);
    assert_eq!(v, expected);
}
