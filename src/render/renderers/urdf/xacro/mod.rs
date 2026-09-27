//! xacro expansion inside visor (include / property / macro / if-unless / `$(find)` / `$(env)` / `$(optenv)` / `$(arg)` plus the `${}` subset of expr.rs), so a `*.urdf.xacro` loads on a machine that has neither the `xacro` command nor a ROS install; anything outside the subset is a located error, never a silently empty robot.

pub mod expr;

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use super::resolve::{self, MeshRoots};
use expr::Value;

/// Namespace URI every xacro file declares as `xmlns:xacro`.
pub const XACRO_NS: &str = "http://www.ros.org/wiki/xacro";
/// Nested include limit; an include cycle is the only way to reach it.
const MAX_INCLUDE_DEPTH: usize = 32;
/// Nested macro call limit; a macro calling itself is the only way to reach it.
const MAX_CALL_DEPTH: usize = 64;
/// xacro's deprecated unprefixed spellings of its own elements; rejected rather than passed through as URDF (where urdf-rs would silently drop them).
const LEGACY_TAGS: [&str; 7] = [
    "include",
    "macro",
    "property",
    "if",
    "unless",
    "arg",
    "insert_block",
];

/// `path` made absolute against the current directory (left as-is only when the current directory itself is unavailable).
fn absolute(path: &Path) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}

/// xacro CLI style `name:=value` pairs: they define `$(arg name)` and, as a visor extension, override the process environment for `$(env)` / `$(optenv)` (robot descriptions pick the model through env vars, which a GUI has no other way to set).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Mappings(BTreeMap<String, String>);

impl Mappings {
    /// Parse whitespace-separated `name:=value` words, the way the xacro command line takes them.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut map = BTreeMap::new();
        for word in text.split_whitespace() {
            let (name, value) = word
                .split_once(":=")
                .ok_or_else(|| format!("`{word}` is not a `name:=value` mapping"))?;
            if name.is_empty() {
                return Err(format!("`{word}` has an empty name"));
            }
            map.insert(name.to_owned(), value.to_owned());
        }
        Ok(Self(map))
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.0.get(name).map(String::as_str)
    }

    pub fn set(&mut self, name: &str, value: &str) {
        self.0.insert(name.to_owned(), value.to_owned());
    }

    pub fn remove(&mut self, name: &str) {
        self.0.remove(name);
    }

    /// Back to the command-line form, `name:=value` words in name order (what the settings field stores).
    pub fn to_text(&self) -> String {
        self.0
            .iter()
            .map(|(name, value)| format!("{name}:={value}"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Which substitution read a variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableKind {
    /// `$(env NAME)` or `$(optenv NAME default)`.
    Env,
    /// `$(arg NAME)`.
    Arg,
}

/// Where a variable's value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableSource {
    Mapping,
    Environment,
    /// The default written in the file (`optenv`'s trailing words or `<xacro:arg default>`).
    Default,
    /// Nothing supplied a value; the expansion failed right there.
    Undefined,
}

/// One variable the expansion read through `$(env)` / `$(optenv)` / `$(arg)`: what it resolved to and, when the variable picks an include file, which values name a file that exists. The settings panel turns these into a form so nobody has to know the variable names by heart.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Variable {
    pub name: String,
    pub kind: VariableKind,
    /// Effective value (empty when undefined).
    pub value: String,
    pub source: VariableSource,
    /// Default as written in the file, if it had one.
    pub default: Option<String>,
    /// Values for which the include path this variable appears in names an existing file, sorted; empty when the variable is not part of an include path.
    pub candidates: Vec<String>,
}

/// Read-only inputs of one expansion.
pub struct XacroContext<'a> {
    /// Entry file, the base of relative includes and of the `$(find)` ancestor search; None for text off a topic.
    pub file: Option<&'a Path>,
    /// Package search roots shared with `package://` mesh resolution.
    pub roots: &'a MeshRoots,
    pub mappings: &'a Mappings,
    /// Process environment, injected so tests need no real variables.
    pub env: &'a dyn Fn(&str) -> Option<String>,
}

/// Where in which file something went wrong (1-based row and column, as editors show them).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Location {
    pub file: Option<Rc<PathBuf>>,
    pub row: u32,
    pub col: u32,
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.file {
            Some(file) => write!(f, "{}:{}:{}", file.display(), self.row, self.col),
            None => write!(f, "line {}:{}", self.row, self.col),
        }
    }
}

/// Why an expansion failed; `location` is None only when the failure has no place in any file (a bad mapping, for example).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XacroError {
    pub message: String,
    pub location: Option<Location>,
    /// Variables read before the failure, so the form can still offer a fix (a mapping that names a missing model file, for example).
    pub variables: Vec<Variable>,
}

impl fmt::Display for XacroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.location {
            Some(at) => write!(f, "{at}: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for XacroError {}

/// Result of a successful expansion: plain URDF text plus how many files were read (the entry included).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expanded {
    pub xml: String,
    pub files: usize,
    /// Macros defined but never called, sorted. A file that only defines macros is an include meant for another entry file; the renderer says so instead of "no links".
    pub unused_macros: Vec<String>,
    /// Variables read, in the order first encountered.
    pub variables: Vec<Variable>,
}

/// Expand xacro `text` into URDF text. Only the subset described in the module docs is understood; the rest fails with the offending element's file and line.
pub fn expand(text: &str, ctx: &XacroContext<'_>) -> Result<Expanded, XacroError> {
    // Absolute from the start, so `$(find)` (which searches this file's ancestors) and relative includes never join two relative paths.
    let file = ctx.file.map(absolute);
    let root = parse_document(text, file.clone().map(Rc::new))?;
    if root.xacro.is_some() {
        return Err(root.at.error(format!(
            "the root element must be <robot>, not <{}>",
            root.name
        )));
    }
    let base_dir = file
        .as_deref()
        .and_then(Path::parent)
        .map(Path::to_path_buf);
    let mut expander = Expander {
        ctx,
        scope: Scope::new(),
        macros: HashMap::new(),
        called: HashSet::new(),
        args: HashMap::new(),
        evaluating_args: RefCell::new(Vec::new()),
        variables: RefCell::new(Vec::new()),
        files: 1,
        include_depth: 0,
        call_depth: 0,
        out: String::with_capacity(text.len()),
    };
    let result = expander
        .collect_args(&root)
        .and_then(|()| expander.write_element(&root, base_dir.as_deref()));
    let variables = expander.variables.into_inner();
    if let Err(mut error) = result {
        error.variables = variables;
        return Err(error);
    }
    let mut unused_macros: Vec<String> = expander
        .macros
        .keys()
        .filter(|name| !expander.called.contains(*name))
        .cloned()
        .collect();
    unused_macros.sort();
    Ok(Expanded {
        xml: expander.out,
        files: expander.files,
        unused_macros,
        variables,
    })
}

/// Stand-in for a variable's value while working out which files an include pattern could name; never appears in a real path.
const WILDCARD: char = '\u{1}';

/// Names of the variables an attribute text reads through `$(env)` / `$(optenv)` / `$(arg)`, in order of first appearance.
fn referenced_variables(text: &str) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("$(") {
        let body = &rest[start + 2..];
        let Some(end) = body.find(')') else {
            break;
        };
        let mut words = body[..end].split_whitespace();
        if let (Some("env" | "optenv" | "arg"), Some(name)) = (words.next(), words.next())
            && !names.iter().any(|n| n == name)
        {
            names.push(name.to_owned());
        }
        rest = &body[end + 1..];
    }
    names
}

/// A value has to satisfy every include the variable takes part in, so a second include narrows the list to the intersection (an include with no enumerable pattern contributes nothing and leaves the list alone).
fn merge_candidates(existing: &mut Vec<String>, incoming: Vec<String>) {
    if incoming.is_empty() {
        return;
    }
    if existing.is_empty() {
        *existing = incoming;
    } else {
        existing.retain(|candidate| incoming.contains(candidate));
    }
}

/// Values of the wildcard for which `pattern` (a path whose last component contains [`WILDCARD`] once) names an existing file.
fn wildcard_candidates(pattern: &Path) -> Vec<String> {
    let Some(file_name) = pattern.file_name().and_then(|n| n.to_str()) else {
        return Vec::new();
    };
    let Some(dir) = pattern.parent() else {
        return Vec::new();
    };
    // A wildcard inside a directory component would need a recursive walk; those includes get no candidates.
    if dir.to_string_lossy().contains(WILDCARD) {
        return Vec::new();
    }
    let Some((prefix, suffix)) = file_name.split_once(WILDCARD) else {
        return Vec::new();
    };
    if suffix.contains(WILDCARD) {
        return Vec::new();
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut candidates: Vec<String> = entries
        .flatten()
        .filter_map(|entry| entry.file_name().to_str().map(str::to_owned))
        .filter_map(|name| {
            let middle = name.strip_prefix(prefix)?.strip_suffix(suffix)?;
            (!middle.is_empty()).then(|| middle.to_owned())
        })
        .collect();
    candidates.sort();
    candidates.dedup();
    candidates
}

#[derive(Debug, Clone)]
enum Node {
    Element(Element),
    Text(String),
}

/// Owned copy of one XML element, so included documents and macro bodies outlive the parser that read them.
#[derive(Debug, Clone)]
struct Element {
    /// Local name when the element is in the xacro namespace (`include`, `macro`, a macro call name, ...).
    xacro: Option<String>,
    /// Qualified name as written, used verbatim in the output.
    name: String,
    attrs: Vec<(String, String)>,
    children: Vec<Node>,
    at: Location,
}

impl Element {
    fn attr(&self, name: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    fn required_attr(&self, name: &str) -> Result<&str, XacroError> {
        self.attr(name).ok_or_else(|| {
            self.at
                .error(format!("<{}> needs a `{name}` attribute", self.name))
        })
    }

    fn has_element_children(&self) -> bool {
        self.children.iter().any(|c| matches!(c, Node::Element(_)))
    }
}

impl Location {
    fn error(&self, message: impl Into<String>) -> XacroError {
        XacroError {
            message: message.into(),
            location: Some(self.clone()),
            variables: Vec::new(),
        }
    }
}

/// Parse one file into the owned tree; the root carries the file's namespace declarations (minus xacro's) so they survive into the output.
fn parse_document(text: &str, file: Option<Rc<PathBuf>>) -> Result<Element, XacroError> {
    let options = roxmltree::ParsingOptions {
        allow_dtd: true,
        ..Default::default()
    };
    let doc = roxmltree::Document::parse_with_options(text, options).map_err(|e| {
        let hint = match e {
            roxmltree::Error::UnknownNamespace(..) => {
                " (a file that uses `xacro:` elements must declare xmlns:xacro=\"http://www.ros.org/wiki/xacro\" on its root)"
            }
            _ => "",
        };
        let pos = e.pos();
        XacroError {
            message: format!("not well-formed XML: {e}{hint}"),
            location: Some(Location {
                file: file.clone(),
                row: pos.row,
                col: pos.col,
            }),
            variables: Vec::new(),
        }
    })?;
    let root = doc.root_element();
    let mut element = convert(root, &doc, &file);
    let declarations: Vec<(String, String)> = root
        .namespaces()
        .filter(|ns| ns.uri() != XACRO_NS)
        .map(|ns| {
            let key = match ns.name() {
                Some(prefix) => format!("xmlns:{prefix}"),
                None => "xmlns".to_owned(),
            };
            (key, ns.uri().to_owned())
        })
        .collect();
    element.attrs.splice(0..0, declarations);
    Ok(element)
}

fn convert(
    node: roxmltree::Node<'_, '_>,
    doc: &roxmltree::Document<'_>,
    file: &Option<Rc<PathBuf>>,
) -> Element {
    let tag = node.tag_name();
    let xacro = (tag.namespace() == Some(XACRO_NS)).then(|| tag.name().to_owned());
    let attrs = node
        .attributes()
        .map(|a| {
            (
                qualified(node, a.namespace(), a.name()),
                a.value().to_owned(),
            )
        })
        .collect();
    let children = node
        .children()
        .filter_map(|child| {
            if child.is_element() {
                Some(Node::Element(convert(child, doc, file)))
            } else if child.is_text() {
                child.text().map(|t| Node::Text(t.to_owned()))
            } else {
                None
            }
        })
        .collect();
    let pos = doc.text_pos_at(node.range().start);
    Element {
        xacro,
        name: qualified(node, tag.namespace(), tag.name()),
        attrs,
        children,
        at: Location {
            file: file.clone(),
            row: pos.row,
            col: pos.col,
        },
    }
}

/// `prefix:local` as the file wrote it, or the bare local name when the name is unprefixed (default namespace included).
fn qualified(node: roxmltree::Node<'_, '_>, namespace: Option<&str>, local: &str) -> String {
    match namespace.and_then(|uri| node.lookup_prefix(uri)) {
        Some(prefix) if !prefix.is_empty() => format!("{prefix}:{local}"),
        _ => local.to_owned(),
    }
}

/// A `<xacro:macro>` definition, kept as its unevaluated body plus the directory its file lives in (relative includes inside the body resolve against that, not the call site).
#[derive(Debug, Clone)]
struct Macro {
    params: Vec<Param>,
    body: Vec<Node>,
    base_dir: Option<PathBuf>,
    at: Location,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Param {
    name: String,
    default: ParamDefault,
}

/// The `params="a b:=1 c:=^ d:=^|2"` forms: required, literal default, inherit from the calling scope, inherit or default.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ParamDefault {
    Required,
    Value(String),
    Inherit,
    InheritOr(String),
}

fn parse_params(text: &str) -> Result<Vec<Param>, String> {
    text.split_whitespace()
        .map(|word| {
            if word.starts_with('*') {
                return Err(format!(
                    "block parameter `{word}` is not supported (visor's xacro subset has no insert_block)"
                ));
            }
            let (name, default) = match word.split_once(":=") {
                None => (word, ParamDefault::Required),
                Some((name, "^")) => (name, ParamDefault::Inherit),
                Some((name, default)) => match default.strip_prefix("^|") {
                    Some(rest) => (name, ParamDefault::InheritOr(rest.to_owned())),
                    None => (name, ParamDefault::Value(default.to_owned())),
                },
            };
            if name.is_empty() {
                return Err(format!("parameter `{word}` has an empty name"));
            }
            Ok(Param {
                name: name.to_owned(),
                default,
            })
        })
        .collect()
}

/// Property / parameter frames, innermost last: a macro call pushes one, so a body sees its params, then its caller's names, then the globals (xacro's dynamic scoping).
struct Scope {
    frames: Vec<HashMap<String, Value>>,
}

impl Scope {
    fn new() -> Self {
        Self {
            frames: vec![HashMap::new()],
        }
    }

    fn lookup(&self, name: &str) -> Option<Value> {
        self.frames
            .iter()
            .rev()
            .find_map(|frame| frame.get(name).cloned())
    }

    fn define(&mut self, name: &str, value: Value) {
        self.frames
            .last_mut()
            .expect("the global frame is never popped")
            .insert(name.to_owned(), value);
    }

    fn define_in_parent(&mut self, name: &str, value: Value) -> Result<(), String> {
        let depth = self.frames.len();
        if depth < 2 {
            return Err("scope=\"parent\" is only allowed inside a macro".to_owned());
        }
        self.frames[depth - 2].insert(name.to_owned(), value);
        Ok(())
    }

    fn define_global(&mut self, name: &str, value: Value) {
        self.frames[0].insert(name.to_owned(), value);
    }
}

struct Expander<'c> {
    ctx: &'c XacroContext<'c>,
    scope: Scope,
    macros: HashMap<String, Macro>,
    /// Names of the macros called at least once (for the unused-macro hint).
    called: HashSet<String>,
    /// `<xacro:arg>` defaults as written (evaluated when `$(arg)` asks); mappings take precedence at lookup time.
    args: HashMap<String, String>,
    /// Args whose default is being evaluated right now, innermost last; a name showing up twice is a reference cycle.
    evaluating_args: RefCell<Vec<String>>,
    /// Variables read so far, first encounter first (a RefCell because substitution runs behind `&self`).
    variables: RefCell<Vec<Variable>>,
    files: usize,
    include_depth: usize,
    call_depth: usize,
    out: String,
}

impl Expander<'_> {
    /// Register the `<xacro:arg>` declarations directly under the root up front, so `<robot name="$(arg name)">` works with the declaration inside (as it does in xacro). Declarations nested in conditionals or macro bodies register only when they are actually reached.
    fn collect_args(&mut self, root: &Element) -> Result<(), XacroError> {
        for child in &root.children {
            if let Node::Element(child) = child
                && child.xacro.as_deref() == Some("arg")
            {
                self.declare_arg(child)?;
            }
        }
        Ok(())
    }

    fn declare_arg(&mut self, element: &Element) -> Result<(), XacroError> {
        let name = element.required_attr("name")?.to_owned();
        if let Some(default) = element.attr("default") {
            self.args.entry(name).or_insert_with(|| default.to_owned());
        }
        Ok(())
    }

    fn process_nodes(
        &mut self,
        nodes: &[Node],
        base_dir: Option<&Path>,
        parent: &Location,
    ) -> Result<(), XacroError> {
        for node in nodes {
            match node {
                Node::Text(text) => {
                    let value = self.eval_text(text, base_dir, parent)?;
                    escape_text(&value.to_string(), &mut self.out);
                }
                Node::Element(element) => self.process_element(element, base_dir)?,
            }
        }
        Ok(())
    }

    fn process_element(
        &mut self,
        element: &Element,
        base_dir: Option<&Path>,
    ) -> Result<(), XacroError> {
        match element.xacro.as_deref() {
            None if LEGACY_TAGS.contains(&element.name.as_str()) => Err(element.at.error(format!(
                "<{0}> without the xacro prefix is xacro's deprecated legacy syntax, which visor does not expand; write <xacro:{0}>",
                element.name
            ))),
            None => self.write_element(element, base_dir),
            Some("include") => self.include(element, base_dir),
            Some("property") => self.property(element, base_dir),
            Some("macro") => self.define_macro(element, base_dir),
            Some("arg") => self.declare_arg(element),
            Some("if") => self.conditional(element, base_dir, true),
            Some("unless") => self.conditional(element, base_dir, false),
            Some(name @ ("insert_block" | "element" | "attribute")) => Err(element.at.error(format!(
                "<xacro:{name}> is not supported (visor's xacro subset has include, property, macro, arg, if, unless and macro calls)"
            ))),
            Some(name) => self.call_macro(name, element, base_dir),
        }
    }

    /// Emit a plain element with its attributes and text evaluated, and its children processed.
    fn write_element(
        &mut self,
        element: &Element,
        base_dir: Option<&Path>,
    ) -> Result<(), XacroError> {
        self.out.push('<');
        self.out.push_str(&element.name);
        for (key, raw) in &element.attrs {
            let value = self.eval_text(raw, base_dir, &element.at)?.to_string();
            self.out.push(' ');
            self.out.push_str(key);
            self.out.push_str("=\"");
            escape_attr(&value, &mut self.out);
            self.out.push('"');
        }
        if element.children.is_empty() {
            self.out.push_str("/>");
            return Ok(());
        }
        self.out.push('>');
        self.process_nodes(&element.children, base_dir, &element.at)?;
        self.out.push_str("</");
        self.out.push_str(&element.name);
        self.out.push('>');
        Ok(())
    }

    fn include(&mut self, element: &Element, base_dir: Option<&Path>) -> Result<(), XacroError> {
        let raw = element.required_attr("filename")?;
        let filename = self.eval_text(raw, base_dir, &element.at)?.to_string();
        for name in referenced_variables(raw) {
            let candidates = self.include_candidates(raw, &name, base_dir);
            self.add_candidates(&name, candidates);
        }
        let path = Path::new(&filename);
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            match base_dir {
                Some(dir) => dir.join(path),
                None => {
                    return Err(element.at.error(format!(
                        "cannot include the relative path `{filename}`: this xacro did not come from a file"
                    )));
                }
            }
        };
        if self.include_depth >= MAX_INCLUDE_DEPTH {
            return Err(element.at.error(format!(
                "includes nested deeper than {MAX_INCLUDE_DEPTH} levels (an include cycle?) at `{}`",
                path.display()
            )));
        }
        let text = std::fs::read_to_string(&path).map_err(|e| {
            element
                .at
                .error(format!("cannot include `{}`: {e}", path.display()))
        })?;
        let root = parse_document(&text, Some(Rc::new(path.clone())))?;
        self.files += 1;
        self.include_depth += 1;
        let included_dir = path.parent().map(Path::to_path_buf);
        self.process_nodes(&root.children, included_dir.as_deref(), &root.at)?;
        self.include_depth -= 1;
        Ok(())
    }

    fn property(&mut self, element: &Element, base_dir: Option<&Path>) -> Result<(), XacroError> {
        let name = element.required_attr("name")?;
        if element.has_element_children() {
            return Err(element.at.error(format!(
                "block property `{name}` is not supported (visor's xacro subset has value properties only)"
            )));
        }
        let (text, only_if_unset) = match (element.attr("value"), element.attr("default")) {
            (Some(value), _) => (value, false),
            (None, Some(default)) => (default, true),
            (None, None) => {
                return Err(element.at.error(format!(
                    "<xacro:property name=\"{name}\"> needs a `value` or `default` attribute"
                )));
            }
        };
        if only_if_unset && self.scope.lookup(name).is_some() {
            return Ok(());
        }
        let value = literal(self.eval_text(text, base_dir, &element.at)?);
        match element.attr("scope") {
            None | Some("local") => self.scope.define(name, value),
            Some("global") => self.scope.define_global(name, value),
            Some("parent") => self
                .scope
                .define_in_parent(name, value)
                .map_err(|e| element.at.error(e))?,
            Some(other) => {
                return Err(element.at.error(format!(
                    "unknown property scope `{other}` (expected local, parent or global)"
                )));
            }
        }
        Ok(())
    }

    fn define_macro(
        &mut self,
        element: &Element,
        base_dir: Option<&Path>,
    ) -> Result<(), XacroError> {
        let name = element.required_attr("name")?;
        let params = parse_params(element.attr("params").unwrap_or_default())
            .map_err(|e| element.at.error(format!("macro `{name}`: {e}")))?;
        self.macros.insert(
            name.to_owned(),
            Macro {
                params,
                body: element.children.clone(),
                base_dir: base_dir.map(Path::to_path_buf),
                at: element.at.clone(),
            },
        );
        Ok(())
    }

    fn conditional(
        &mut self,
        element: &Element,
        base_dir: Option<&Path>,
        want: bool,
    ) -> Result<(), XacroError> {
        let value = self.eval_text(element.required_attr("value")?, base_dir, &element.at)?;
        let truth = value
            .truthy()
            .map_err(|e| element.at.error(format!("<{}>: {e}", element.name)))?;
        if truth == want {
            self.process_nodes(&element.children, base_dir, &element.at)?;
        }
        Ok(())
    }

    fn call_macro(
        &mut self,
        name: &str,
        call: &Element,
        base_dir: Option<&Path>,
    ) -> Result<(), XacroError> {
        let Some(macro_) = self.macros.get(name).cloned() else {
            return Err(call.at.error(format!(
                "<xacro:{name}> is neither a known xacro element nor a macro defined before this point"
            )));
        };
        self.called.insert(name.to_owned());
        if self.call_depth >= MAX_CALL_DEPTH {
            return Err(call.at.error(format!(
                "macro calls nested deeper than {MAX_CALL_DEPTH} levels (a macro calling itself?) at <xacro:{name}>"
            )));
        }
        if call.has_element_children() {
            return Err(call.at.error(format!(
                "<xacro:{name}> has block content, which is not supported (visor's xacro subset has no insert_block)"
            )));
        }
        for (key, _) in &call.attrs {
            if !macro_.params.iter().any(|p| p.name == *key) {
                return Err(call
                    .at
                    .error(format!("macro `{name}` has no parameter `{key}`")));
            }
        }
        let mut frame = HashMap::new();
        for param in &macro_.params {
            let value = match (call.attr(&param.name), &param.default) {
                (Some(text), _) => literal(self.eval_text(text, base_dir, &call.at)?),
                (None, ParamDefault::Value(default)) => literal(self.eval_text(default, base_dir, &call.at)?),
                (None, ParamDefault::Inherit) => self.scope.lookup(&param.name).ok_or_else(|| {
                    call.at.error(format!(
                        "macro `{name}`: parameter `{}` inherits (`^`) but nothing of that name is in scope",
                        param.name
                    ))
                })?,
                (None, ParamDefault::InheritOr(default)) => match self.scope.lookup(&param.name) {
                    Some(value) => value,
                    None => literal(self.eval_text(default, base_dir, &call.at)?),
                },
                (None, ParamDefault::Required) => {
                    return Err(call.at.error(format!(
                        "macro `{name}` needs the parameter `{}`",
                        param.name
                    )));
                }
            };
            frame.insert(param.name.clone(), value);
        }
        self.scope.frames.push(frame);
        self.call_depth += 1;
        self.process_nodes(&macro_.body, macro_.base_dir.as_deref(), &macro_.at)?;
        self.call_depth -= 1;
        self.scope.frames.pop();
        Ok(())
    }

    fn eval_text(
        &self,
        text: &str,
        base_dir: Option<&Path>,
        at: &Location,
    ) -> Result<Value, XacroError> {
        let lookup = |name: &str| self.scope.lookup(name);
        let subst = |inner: &str| self.substitute(inner, base_dir);
        expr::eval_text(text, &lookup, &subst).map_err(|e| at.error(format!("in `{text}`: {e}")))
    }

    /// One `$(...)`: `find`, `env`, `optenv`, `arg`.
    fn substitute(&self, inner: &str, base_dir: Option<&Path>) -> Result<String, String> {
        let mut words = inner.split_whitespace();
        let command = words.next().ok_or_else(|| "empty `$()`".to_owned())?;
        let mut one_word = |what: &str| {
            words
                .next()
                .ok_or_else(|| format!("$({command}) needs a {what}"))
                .map(str::to_owned)
        };
        match command {
            "find" => {
                let package = one_word("package name")?;
                if words.next().is_some() {
                    return Err(format!(
                        "$(find {package} ...) takes exactly one package name"
                    ));
                }
                resolve::find_package(&package, self.ctx.roots, base_dir, |p| p.is_dir())
                    .map(|dir| absolute(&dir).display().to_string())
                    .map_err(|e| e.to_string())
            }
            "env" => {
                let var = one_word("variable name")?;
                let (value, source) = self.variable(&var);
                self.record(&var, VariableKind::Env, value.as_deref(), source, None);
                value.ok_or_else(|| {
                    format!("environment variable `{var}` is not set (add `{var}:=value` to the mappings)")
                })
            }
            "optenv" => {
                let var = one_word("variable name")?;
                let default = words.collect::<Vec<&str>>().join(" ");
                let (value, source) = self.variable(&var);
                let (value, source) = match value {
                    Some(value) => (value, source),
                    None => (default.clone(), VariableSource::Default),
                };
                self.record(
                    &var,
                    VariableKind::Env,
                    Some(&value),
                    source,
                    Some(&default),
                );
                Ok(value)
            }
            "arg" => {
                let name = one_word("argument name")?;
                if let Some(value) = self.ctx.mappings.get(&name) {
                    self.record(
                        &name,
                        VariableKind::Arg,
                        Some(value),
                        VariableSource::Mapping,
                        self.args.get(&name).map(String::as_str),
                    );
                    return Ok(value.to_owned());
                }
                let Some(default) = self.args.get(&name).cloned() else {
                    self.record(
                        &name,
                        VariableKind::Arg,
                        None,
                        VariableSource::Undefined,
                        None,
                    );
                    return Err(format!(
                        "$(arg {name}) is undefined (declare it with <xacro:arg name=\"{name}\" default=\"...\"/> or add `{name}:=value` to the mappings)"
                    ));
                };
                if self.evaluating_args.borrow().contains(&name) {
                    let chain = self.evaluating_args.borrow().join(" -> ");
                    return Err(format!(
                        "$(arg {name}) refers to itself through its default value ({chain} -> {name})"
                    ));
                }
                self.evaluating_args.borrow_mut().push(name.clone());
                let lookup = |n: &str| self.scope.lookup(n);
                let subst = |inner: &str| self.substitute(inner, base_dir);
                let result =
                    expr::eval_text(&default, &lookup, &subst).map(|value| value.to_string());
                self.evaluating_args.borrow_mut().pop();
                if let Ok(value) = &result {
                    self.record(
                        &name,
                        VariableKind::Arg,
                        Some(value),
                        VariableSource::Default,
                        Some(&default),
                    );
                }
                result
            }
            other => Err(format!(
                "unsupported substitution $({other} ...) (visor's xacro subset has find, env, optenv, arg)"
            )),
        }
    }

    /// Mappings first, then the process environment, with which of the two answered.
    fn variable(&self, name: &str) -> (Option<String>, VariableSource) {
        if let Some(value) = self.ctx.mappings.get(name) {
            return (Some(value.to_owned()), VariableSource::Mapping);
        }
        match (self.ctx.env)(name) {
            Some(value) => (Some(value), VariableSource::Environment),
            None => (None, VariableSource::Undefined),
        }
    }

    /// Remember a variable the first time it is read (later reads resolve the same way, except for a differing `optenv` default, where the first one stands).
    fn record(
        &self,
        name: &str,
        kind: VariableKind,
        value: Option<&str>,
        source: VariableSource,
        default: Option<&str>,
    ) {
        let mut variables = self.variables.borrow_mut();
        if variables.iter().any(|v| v.name == name) {
            return;
        }
        variables.push(Variable {
            name: name.to_owned(),
            kind,
            value: value.unwrap_or_default().to_owned(),
            source,
            default: default.map(str::to_owned),
            candidates: Vec::new(),
        });
    }

    fn add_candidates(&self, name: &str, candidates: Vec<String>) {
        let mut variables = self.variables.borrow_mut();
        if let Some(variable) = variables.iter_mut().find(|v| v.name == name) {
            merge_candidates(&mut variable.candidates, candidates);
        }
    }

    /// Values of `name` for which the include `raw` would name an existing file: the path is evaluated with that one variable replaced by a wildcard and the directory is listed against the resulting pattern.
    fn include_candidates(&self, raw: &str, name: &str, base_dir: Option<&Path>) -> Vec<String> {
        let lookup = |n: &str| self.scope.lookup(n);
        let subst = |inner: &str| {
            let mut words = inner.split_whitespace();
            match (words.next(), words.next()) {
                (Some("env" | "optenv" | "arg"), Some(var)) if var == name => {
                    Ok(WILDCARD.to_string())
                }
                _ => self.substitute(inner, base_dir),
            }
        };
        let Ok(pattern) = expr::eval_text(raw, &lookup, &subst) else {
            return Vec::new();
        };
        let pattern = pattern.to_string();
        let pattern = Path::new(&pattern);
        let pattern = match (pattern.is_absolute(), base_dir) {
            (true, _) => pattern.to_path_buf(),
            (false, Some(dir)) => dir.join(pattern),
            (false, None) => return Vec::new(),
        };
        wildcard_candidates(&pattern)
    }
}

/// A string result of eval_text is what the file wrote, so it gets xacro's literal conversion; a typed result stays.
fn literal(value: Value) -> Value {
    match value {
        Value::Str(text) => Value::literal(&text),
        typed => typed,
    }
}

fn escape_text(text: &str, out: &mut String) {
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            other => out.push(other),
        }
    }
}

fn escape_attr(text: &str, out: &mut String) {
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '"' => out.push_str("&quot;"),
            other => out.push(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::renderers::urdf::model::{self, Shape};

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/xacro/robot_description/urdf")
            .join(name)
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn expand_file(name: &str, mappings: &str) -> Result<Expanded, XacroError> {
        let path = fixture(name);
        let text = std::fs::read_to_string(&path).expect("fixture readable");
        let mappings = Mappings::parse(mappings).expect("mappings parse");
        let roots = MeshRoots::default();
        expand(
            &text,
            &XacroContext {
                file: Some(&path),
                roots: &roots,
                mappings: &mappings,
                env: &no_env,
            },
        )
    }

    /// Expand inline text with no file behind it (the topic case), against an injectable environment.
    fn expand_text(
        text: &str,
        mappings: &str,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Expanded, XacroError> {
        let mappings = Mappings::parse(mappings).expect("mappings parse");
        let roots = MeshRoots::default();
        expand(
            text,
            &XacroContext {
                file: None,
                roots: &roots,
                mappings: &mappings,
                env,
            },
        )
    }

    fn wrap(body: &str) -> String {
        format!(r#"<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="r">{body}</robot>"#)
    }

    #[test]
    fn the_fixture_expands_like_a_real_robot_description() {
        let expanded = expand_file("robot.urdf.xacro", "").expect("expands");
        // Entry + materials.xacro + demo.xacro.
        assert_eq!(expanded.files, 3);
        assert!(!expanded.xml.contains("xacro"), "{}", expanded.xml);
        assert!(
            expanded.unused_macros.is_empty(),
            "{:?}",
            expanded.unused_macros
        );
        // Every variable the file read, in encounter order (the root attribute first), with the model selector's candidates listed from the directory.
        let summary: Vec<(&str, &str, VariableSource, &[String])> = expanded
            .variables
            .iter()
            .map(|v| {
                (
                    v.name.as_str(),
                    v.value.as_str(),
                    v.source,
                    v.candidates.as_slice(),
                )
            })
            .collect();
        let models = ["alt".to_owned(), "demo".to_owned()];
        assert_eq!(
            summary,
            [
                ("robot_name", "fixture", VariableSource::Default, &[][..]),
                ("ROBOT_MODEL", "demo", VariableSource::Default, &models[..]),
                ("USE_MESH", "true", VariableSource::Default, &[][..]),
            ]
        );
        assert_eq!(expanded.variables[0].kind, VariableKind::Arg);
        assert_eq!(expanded.variables[1].kind, VariableKind::Env);
        assert_eq!(expanded.variables[1].default.as_deref(), Some("demo"));
        // A file that only defines macros (a description file meant to be included) expands to an empty robot; the unused names let the caller explain that.
        let defs_only = expand_file("demo_model.xacro", "").expect("expands");
        assert_eq!(defs_only.unused_macros, ["robot"]);
        assert!(model::parse(&defs_only.xml).unwrap().visuals.is_empty());
        let parsed = model::parse(&expanded.xml).expect("expanded text is URDF");
        assert_eq!(parsed.robot_name, "fixture");
        assert_eq!(parsed.link_count, 3);
        let names: Vec<&str> = parsed.visuals.iter().map(|v| v.link.as_str()).collect();
        assert_eq!(names, ["base_link", "left_wheel_link", "right_wheel_link"]);
        assert_eq!(
            parsed.visuals[0].shape,
            Shape::Mesh {
                uri: "package://mesh/triangle_ascii.stl".to_owned(),
                scale: [1.0; 3],
            }
        );
        let left = &parsed.visuals[1];
        assert_eq!(
            left.shape,
            Shape::Cylinder {
                radius: 0.05,
                length: 0.03
            }
        );
        // `0 ${wheel_tread * offset / 2} ${wheel_radius}` with tread 0.4, offset 1, radius 0.05.
        let t = left.origin.translation.vector;
        assert!((t.x, t.y, t.z) == (0.0, 0.2, 0.05), "{t:?}");
        let (roll, pitch, yaw) = left.origin.rotation.euler_angles();
        assert!((roll - std::f64::consts::FRAC_PI_2).abs() < 1e-12 && pitch == 0.0 && yaw == 0.0);
        // The right wheel passed `length="0.04"` explicitly and `offset="-1"`.
        let right = &parsed.visuals[2];
        assert_eq!(
            right.shape,
            Shape::Cylinder {
                radius: 0.05,
                length: 0.04
            }
        );
        assert_eq!(right.origin.translation.vector.y, -0.2);
        // Colors came through the included materials file (black wheels).
        assert_eq!(left.color[..3], [0, 0, 0]);
    }

    #[test]
    fn mappings_switch_the_model_and_override_the_environment() {
        let expanded = expand_file(
            "robot.urdf.xacro",
            "ROBOT_MODEL:=alt robot_name:=other USE_MESH:=false",
        )
        .expect("expands");
        let parsed = model::parse(&expanded.xml).expect("URDF");
        assert_eq!(parsed.robot_name, "other");
        assert_eq!(parsed.link_count, 1);
        assert_eq!(parsed.visuals[0].shape, Shape::Sphere { radius: 0.1 });
        // USE_MESH:=false picks the <xacro:unless> branch of the demo model.
        let boxed = expand_file("robot.urdf.xacro", "USE_MESH:=false").expect("expands");
        let parsed = model::parse(&boxed.xml).expect("URDF");
        assert_eq!(
            parsed.visuals[0].shape,
            Shape::Box {
                size: [0.7, 0.53, 0.07]
            }
        );
        // The process environment is consulted when no mapping names the variable.
        let env = |name: &str| (name == "ROBOT_MODEL").then(|| "alt".to_owned());
        let path = fixture("robot.urdf.xacro");
        let text = std::fs::read_to_string(&path).unwrap();
        let expanded = expand(
            &text,
            &XacroContext {
                file: Some(&path),
                roots: &MeshRoots::default(),
                mappings: &Mappings::default(),
                env: &env,
            },
        )
        .expect("expands");
        assert_eq!(model::parse(&expanded.xml).unwrap().link_count, 1);
        assert!(Mappings::parse("a:=1 b:=x:=y").unwrap().get("b") == Some("x:=y"));
        assert!(Mappings::parse("novalue").is_err());
        assert!(Mappings::parse(":=1").is_err());
    }

    #[test]
    fn errors_name_the_file_and_line() {
        let error = expand_file("bad_insert_block.xacro", "").unwrap_err();
        let text = error.to_string();
        assert!(text.contains("bad_insert_block.xacro:4:"), "{text}");
        assert!(text.contains("insert_block"), "{text}");
        let missing = expand_text(
            &wrap(r#"<xacro:include filename="nowhere.xacro"/>"#),
            "",
            &no_env,
        )
        .unwrap_err();
        assert!(
            missing.message.contains("did not come from a file"),
            "{missing}"
        );
        assert!(missing.to_string().starts_with("line 1:"), "{missing}");
        let unresolved = expand_file("bad_include.xacro", "").unwrap_err();
        assert!(
            unresolved.message.contains("cannot include"),
            "{unresolved}"
        );
        assert!(
            unresolved.message.contains("does_not_exist.xacro"),
            "{unresolved}"
        );
        let no_ns = expand_text(
            r#"<robot><xacro:property name="a" value="1"/></robot>"#,
            "",
            &no_env,
        )
        .unwrap_err();
        assert!(no_ns.message.contains("xmlns:xacro"), "{no_ns}");
    }

    #[test]
    fn unsupported_constructs_are_rejected_not_ignored() {
        for (body, needle) in [
            (r#"<xacro:element xacro:name="link"/>"#, "xacro:element"),
            (
                r#"<xacro:property name="p"><a/></xacro:property>"#,
                "block property",
            ),
            (r#"<xacro:property name="p"/>"#, "`value` or `default`"),
            (
                r#"<xacro:macro name="m" params="*block"/>"#,
                "block parameter",
            ),
            (
                r#"<xacro:macro name="m" params="a"/><xacro:m/>"#,
                "needs the parameter `a`",
            ),
            (
                r#"<xacro:macro name="m"/><xacro:m x="1"/>"#,
                "no parameter `x`",
            ),
            (
                r#"<xacro:macro name="m"/><xacro:m><a/></xacro:m>"#,
                "block content",
            ),
            (
                r#"<xacro:nope/>"#,
                "neither a known xacro element nor a macro",
            ),
            (
                r#"<xacro:macro name="m"><xacro:m/></xacro:macro><xacro:m/>"#,
                "nested deeper",
            ),
            (
                r#"<link name="${undefined}"/>"#,
                "undefined name `undefined`",
            ),
            (r#"<link name="$(cwd)"/>"#, "unsupported substitution $(cwd"),
            (r#"<link name="$(arg x)"/>"#, "$(arg x) is undefined"),
            (r#"<link name="$(env NOPE)"/>"#, "`NOPE` is not set"),
            (r#"<link name="$(find)"/>"#, "needs a package name"),
            (r#"<link name="$(find pkg)"/>"#, "no mesh roots"),
            (r#"<xacro:if value="maybe"/>"#, "not a boolean"),
            (
                r#"<xacro:property name="p" value="1" scope="parent"/>"#,
                "only allowed inside a macro",
            ),
            (
                r#"<xacro:property name="p" value="1" scope="odd"/>"#,
                "unknown property scope",
            ),
            (r#"<link name="${1 +}"/>"#, "ends unexpectedly"),
        ] {
            let error = expand_text(&wrap(body), "", &no_env).unwrap_err();
            assert!(error.message.contains(needle), "{body} -> {error}");
            assert!(error.location.is_some(), "{body}");
        }
        let root = expand_text(
            r#"<xacro:macro xmlns:xacro="http://www.ros.org/wiki/xacro" name="m"/>"#,
            "",
            &no_env,
        )
        .unwrap_err();
        assert!(
            root.message.contains("root element must be <robot>"),
            "{root}"
        );
    }

    #[test]
    fn scoping_defaults_and_escapes_follow_xacro() {
        let body = r#"
            <xacro:property name="a" value="1"/>
            <xacro:property name="a" default="9"/>
            <xacro:property name="b" default="${a + 1}"/>
            <xacro:macro name="inner" params="x:=^ y:=^|7 z:=${a*10}">
              <i x="${x}" y="${y}" z="${z}"/>
              <xacro:property name="from_inner" value="in" scope="parent"/>
              <xacro:property name="glob" value="g" scope="global"/>
            </xacro:macro>
            <xacro:macro name="outer" params="x">
              <xacro:inner/>
              <o from_inner="${from_inner}"/>
            </xacro:macro>
            <xacro:outer x="5"/>
            <after glob="${glob}" text="a &amp; b &lt; c" quote='say "hi"'>${a}${b} $$ literal &amp; &lt;tag&gt;</after>
        "#;
        let expanded = expand_text(&wrap(body), "", &no_env).expect("expands");
        let xml = expanded.xml;
        assert!(xml.contains(r#"<i x="5" y="7" z="10"/>"#), "{xml}");
        assert!(xml.contains(r#"<o from_inner="in"/>"#), "{xml}");
        assert!(
            xml.contains(r#"<after glob="g" text="a &amp; b &lt; c" quote="say &quot;hi&quot;">12 $ literal &amp; &lt;tag&gt;</after>"#),
            "{xml}"
        );
        // Every xacro element is gone and the namespace declaration with it.
        assert!(!xml.contains("xacro"), "{xml}");
        assert!(xml.starts_with(r#"<robot name="r">"#), "{xml}");
        // Other namespace declarations on the root survive.
        let other_ns = expand_text(
            r#"<robot xmlns:xacro="http://www.ros.org/wiki/xacro" xmlns:g="urn:g" name="r"><g:x g:y="1"/></robot>"#,
            "",
            &no_env,
        )
        .expect("expands");
        assert_eq!(
            other_ns.xml,
            r#"<robot xmlns:g="urn:g" name="r"><g:x g:y="1"/></robot>"#
        );
        // The first <xacro:arg> default wins over later ones, but a mapping wins over both.
        let args = r#"<xacro:arg name="n" default="one"/><xacro:arg name="n" default="two"/><l name="$(arg n)"/>"#;
        assert!(
            expand_text(&wrap(args), "", &no_env)
                .unwrap()
                .xml
                .contains(r#"<l name="one"/>"#)
        );
        assert!(
            expand_text(&wrap(args), "n:=three", &no_env)
                .unwrap()
                .xml
                .contains(r#"<l name="three"/>"#)
        );
        // optenv joins a multi-word default with single spaces.
        let optenv = wrap(r#"<l name="$(optenv NOPE a   b)"/>"#);
        assert!(
            expand_text(&optenv, "", &no_env)
                .unwrap()
                .xml
                .contains(r#"<l name="a b"/>"#)
        );
        // A macro defined after its use site is an error (single pass, like xacro).
        let late = wrap(r#"<xacro:m/><xacro:macro name="m"/>"#);
        assert!(expand_text(&late, "", &no_env).is_err());
    }

    #[test]
    fn review_findings_arg_cycles_legacy_tags_conditional_args_and_escapes() {
        // An arg whose default refers back to itself (directly or through another arg) is an error, not a stack overflow.
        let cycle = wrap(
            r#"<xacro:arg name="a" default="$(arg b)"/><xacro:arg name="b" default="$(arg a)"/><l name="$(arg a)"/>"#,
        );
        let error = expand_text(&cycle, "", &no_env).unwrap_err();
        assert!(
            error.message.contains("refers to itself") && error.message.contains("a -> b -> a"),
            "{error}"
        );
        let direct = wrap(r#"<xacro:arg name="a" default="x$(arg a)"/><l name="$(arg a)"/>"#);
        assert!(expand_text(&direct, "", &no_env).is_err());
        // A mapping breaks the cycle because it wins before the default is looked at.
        assert!(
            expand_text(&cycle, "b:=ok", &no_env)
                .unwrap()
                .xml
                .contains(r#"<l name="ok"/>"#)
        );
        // Only the declarations directly under the root are collected up front; one inside a false branch never takes effect.
        let conditional = wrap(
            r#"<xacro:if value="false"><xacro:arg name="n" default="wrong"/></xacro:if><xacro:arg name="n" default="right"/><l name="$(arg n)"/>"#,
        );
        assert!(
            expand_text(&conditional, "", &no_env)
                .unwrap()
                .xml
                .contains(r#"<l name="right"/>"#)
        );
        let reached = wrap(
            r#"<xacro:if value="true"><xacro:arg name="n" default="inner"/></xacro:if><l name="$(arg n)"/>"#,
        );
        assert!(
            expand_text(&reached, "", &no_env)
                .unwrap()
                .xml
                .contains(r#"<l name="inner"/>"#)
        );
        // xacro's deprecated unprefixed elements are rejected instead of being emitted as URDF.
        for body in [
            r#"<macro name="m"><link name="a"/></macro>"#,
            r#"<include filename="x.xacro"/>"#,
            r#"<property name="p" value="1"/>"#,
            r#"<if value="true"/>"#,
        ] {
            let error = expand_text(&wrap(body), "", &no_env).unwrap_err();
            assert!(error.message.contains("legacy syntax"), "{body} -> {error}");
        }
        // `$${...}` is a literal `${...}` in the output and parses as plain URDF text.
        let escaped = wrap(
            r#"<link name="$${name}"><visual><geometry><box size="1 1 1"/></geometry></visual></link>"#,
        );
        let xml = expand_text(&escaped, "", &no_env).unwrap().xml;
        assert!(xml.contains(r#"<link name="${name}">"#), "{xml}");
        assert_eq!(
            model::parse(&xml)
                .expect("literal braces are URDF text")
                .visuals[0]
                .link,
            "${name}"
        );
    }

    #[test]
    fn a_relative_entry_path_resolves_find_and_includes_once() {
        // Tests run with the crate root as the working directory, so this is a real relative path to the fixture.
        let relative = Path::new("tests/fixtures/xacro/robot_description/urdf/robot.urdf.xacro");
        assert!(relative.exists(), "cwd is not the crate root");
        let text = std::fs::read_to_string(relative).unwrap();
        let roots = MeshRoots::default();
        let expanded = expand(
            &text,
            &XacroContext {
                file: Some(relative),
                roots: &roots,
                mappings: &Mappings::default(),
                env: &no_env,
            },
        )
        .expect("relative entry path expands");
        assert_eq!(expanded.files, 3);
        // Error locations name the absolute file, so they are unambiguous whatever the working directory was.
        let bad = Path::new("tests/fixtures/xacro/robot_description/urdf/bad_insert_block.xacro");
        let error = expand(
            &std::fs::read_to_string(bad).unwrap(),
            &XacroContext {
                file: Some(bad),
                roots: &roots,
                mappings: &Mappings::default(),
                env: &no_env,
            },
        )
        .unwrap_err();
        let file = error
            .location
            .as_ref()
            .and_then(|l| l.file.clone())
            .expect("has a file");
        assert!(file.is_absolute(), "{}", file.display());
    }

    #[test]
    fn variables_report_their_source_and_survive_a_failed_expansion() {
        // Mapping and environment show up as such; the mapping wins over the environment.
        let env = |name: &str| (name == "USE_MESH").then(|| "false".to_owned());
        let path = fixture("robot.urdf.xacro");
        let text = std::fs::read_to_string(&path).unwrap();
        let mappings = Mappings::parse("ROBOT_MODEL:=alt robot_name:=x").unwrap();
        let expanded = expand(
            &text,
            &XacroContext {
                file: Some(&path),
                roots: &MeshRoots::default(),
                mappings: &mappings,
                env: &env,
            },
        )
        .expect("expands");
        let by_name = |name: &str| {
            expanded
                .variables
                .iter()
                .find(|v| v.name == name)
                .unwrap_or_else(|| panic!("{name} recorded"))
        };
        assert_eq!(
            (
                by_name("ROBOT_MODEL").value.as_str(),
                by_name("ROBOT_MODEL").source
            ),
            ("alt", VariableSource::Mapping)
        );
        assert_eq!(
            (
                by_name("robot_name").value.as_str(),
                by_name("robot_name").source
            ),
            ("x", VariableSource::Mapping)
        );
        assert_eq!(
            (
                by_name("USE_MESH").value.as_str(),
                by_name("USE_MESH").source
            ),
            ("false", VariableSource::Environment)
        );
        // A mapping that names a model file which does not exist fails the include, but the error still carries the variables and the candidates, so the form can offer the fix.
        let error = expand_file("robot.urdf.xacro", "ROBOT_MODEL:=nope").unwrap_err();
        assert!(error.message.contains("cannot include"), "{error}");
        let model = error
            .variables
            .iter()
            .find(|v| v.name == "ROBOT_MODEL")
            .expect("recorded before the failure");
        assert_eq!(model.candidates, ["alt", "demo"]);
        assert_eq!(model.source, VariableSource::Mapping);
        // An undefined $(env) is recorded as such before the error.
        let unset = expand_text(&wrap(r#"<l name="$(env NOPE)"/>"#), "", &no_env).unwrap_err();
        assert_eq!(unset.variables[0].source, VariableSource::Undefined);
        assert_eq!(unset.variables[0].kind, VariableKind::Env);
        // Mappings round-trip through the text form the settings store.
        let mut m = Mappings::parse("b:=2 a:=1").unwrap();
        m.set("c", "3");
        m.remove("b");
        assert_eq!(m.to_text(), "a:=1 c:=3");
        assert_eq!(Mappings::default().to_text(), "");
    }

    #[test]
    fn include_candidates_come_from_the_last_path_component_only() {
        assert_eq!(
            referenced_variables(
                "$(find p)/urdf/$(optenv ROBOT_MODEL x)_description.urdf.xacro $(arg a) $(env B) $(optenv ROBOT_MODEL y)"
            ),
            ["ROBOT_MODEL", "a", "B"]
        );
        assert!(referenced_variables("plain $(find p)/x").is_empty());
        let dir = fixture("robot.urdf.xacro").parent().unwrap().to_path_buf();
        let pattern = dir.join(format!("{WILDCARD}_model.xacro"));
        assert_eq!(wildcard_candidates(&pattern), ["alt", "demo"]);
        let loose = dir.join(format!("{WILDCARD}.xacro"));
        assert!(wildcard_candidates(&loose).contains(&"materials".to_owned()));
        // A wildcard in a directory component, two wildcards, or a directory that does not exist all yield nothing.
        assert!(wildcard_candidates(&dir.join(format!("{WILDCARD}/x.xacro"))).is_empty());
        assert!(wildcard_candidates(&dir.join(format!("{WILDCARD}_{WILDCARD}.xacro"))).is_empty());
        assert!(wildcard_candidates(Path::new("/nonexistent/\u{1}.xacro")).is_empty());
        // The prefix has to match too: nothing in the fixture directory starts with `zz`.
        assert!(wildcard_candidates(&dir.join(format!("zz{WILDCARD}.xacro"))).is_empty());
        // Two includes on the same variable intersect; an include that could not be enumerated changes nothing.
        let mut merged = Vec::new();
        merge_candidates(&mut merged, vec!["a".into(), "b".into(), "c".into()]);
        merge_candidates(&mut merged, Vec::new());
        assert_eq!(merged, ["a", "b", "c"]);
        merge_candidates(&mut merged, vec!["b".into(), "c".into(), "d".into()]);
        assert_eq!(merged, ["b", "c"]);
    }

    #[test]
    fn param_syntax_parses_every_default_form() {
        assert_eq!(
            parse_params("a b:=1 c:=^ d:=^|2").unwrap(),
            vec![
                Param {
                    name: "a".into(),
                    default: ParamDefault::Required
                },
                Param {
                    name: "b".into(),
                    default: ParamDefault::Value("1".into())
                },
                Param {
                    name: "c".into(),
                    default: ParamDefault::Inherit
                },
                Param {
                    name: "d".into(),
                    default: ParamDefault::InheritOr("2".into())
                },
            ]
        );
        assert!(parse_params("").unwrap().is_empty());
        assert!(parse_params("*origin").is_err());
        assert!(parse_params(":=1").is_err());
    }
}
