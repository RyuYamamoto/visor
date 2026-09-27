//! `${...}` evaluation for xacro: the Python subset robot descriptions actually use (arithmetic, comparisons, `and` / `or` / `not`, string literals, a few math functions), plus the `${}` / `$()` / `$$` interpolation of attribute and text values. Parsing builds a small tree first so `and` / `or` and comparison chains short-circuit like Python.

use std::fmt;

/// A dynamically typed xacro value (Python's int / float, str and bool, collapsed to what URDF attributes need).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Number(f64),
    Str(String),
    Bool(bool),
}

impl Value {
    /// The literal a property or parameter text stands for: a number when the whole text parses as one, else the text itself (xacro's `_eval_literal`).
    pub fn literal(text: &str) -> Value {
        match text.trim().parse::<f64>() {
            Ok(n) if n.is_finite() => Value::Number(n),
            _ => Value::Str(text.to_owned()),
        }
    }

    /// Truth value as `xacro:if` reads it: bools as-is, numbers by non-zero, strings `true` / `false` (any case) or an integer; anything else is an error rather than a guess.
    pub fn truthy(&self) -> Result<bool, String> {
        match self {
            Value::Bool(b) => Ok(*b),
            Value::Number(n) => Ok(*n != 0.0),
            Value::Str(s) => match s.trim().to_ascii_lowercase().as_str() {
                "true" => Ok(true),
                "false" => Ok(false),
                other => other.parse::<i64>().map(|n| n != 0).map_err(|_| {
                    format!("`{s}` is not a boolean (expected true / false or an integer)")
                }),
            },
        }
    }

    /// Python truthiness, used by `and` / `or` / `not` (a non-empty string is true there, unlike in `xacro:if`).
    fn python_truth(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Number(n) => *n != 0.0,
            Value::Str(s) => !s.is_empty(),
        }
    }

    fn number(&self, what: &str) -> Result<f64, String> {
        match self {
            Value::Number(n) => Ok(*n),
            Value::Bool(b) => Ok(f64::from(u8::from(*b))),
            Value::Str(s) => Err(format!("{what} needs a number, got the string `{s}`")),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Number(n) => write!(f, "{n}"),
            Value::Str(s) => f.write_str(s),
            Value::Bool(true) => f.write_str("True"),
            Value::Bool(false) => f.write_str("False"),
        }
    }
}

/// Interpolate one attribute or text value: `${expr}` through [`eval_expr`], `$(cmd args)` through `subst` (given the text inside the parentheses), `$$` as a literal `$`. A value that is exactly one `${expr}` keeps its type; anything else becomes a string.
pub fn eval_text(
    text: &str,
    lookup: &dyn Fn(&str) -> Option<Value>,
    subst: &dyn Fn(&str) -> Result<String, String>,
) -> Result<Value, String> {
    if let Some(inner) = text.strip_prefix("${").and_then(|t| t.strip_suffix('}'))
        && !inner.contains('}')
    {
        return eval_expr(inner, lookup);
    }
    let mut out = String::new();
    let mut rest = text;
    while let Some(at) = rest.find('$') {
        out.push_str(&rest[..at]);
        let after = &rest[at + 1..];
        if let Some(tail) = after.strip_prefix('$') {
            out.push('$');
            rest = tail;
        } else if let Some(body) = after.strip_prefix('{') {
            let end = body
                .find('}')
                .ok_or_else(|| format!("unterminated `${{` in `{text}`"))?;
            out.push_str(&eval_expr(&body[..end], lookup)?.to_string());
            rest = &body[end + 1..];
        } else if let Some(body) = after.strip_prefix('(') {
            let end = body
                .find(')')
                .ok_or_else(|| format!("unterminated `$(` in `{text}`"))?;
            out.push_str(&subst(&body[..end])?);
            rest = &body[end + 1..];
        } else {
            out.push('$');
            rest = after;
        }
    }
    out.push_str(rest);
    Ok(Value::Str(out))
}

/// Evaluate the inside of one `${...}`; `lookup` resolves property and parameter names (builtins `pi`, `true` / `false` come after it).
pub fn eval_expr(src: &str, lookup: &dyn Fn(&str) -> Option<Value>) -> Result<Value, String> {
    let tokens = tokenize(src)?;
    if tokens.is_empty() {
        return Err("empty expression `${}`".to_owned());
    }
    let mut parser = Parser { tokens, pos: 0 };
    let expr = parser.or_expr()?;
    if let Some(extra) = parser.tokens.get(parser.pos) {
        return Err(format!("unexpected `{extra}` in `{src}`"));
    }
    eval(&expr, lookup)
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Str(String),
    Ident(String),
    Op(&'static str),
    LParen,
    RParen,
    Comma,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Number(n) => write!(f, "{n}"),
            Token::Str(s) => write!(f, "'{s}'"),
            Token::Ident(s) => f.write_str(s),
            Token::Op(op) => f.write_str(op),
            Token::LParen => f.write_str("("),
            Token::RParen => f.write_str(")"),
            Token::Comma => f.write_str(","),
        }
    }
}

/// Operators, longest first so `**` and `<=` win over their one-character prefixes.
const OPERATORS: [&str; 13] = [
    "**", "//", "==", "!=", "<=", ">=", "+", "-", "*", "/", "%", "<", ">",
];

fn tokenize(src: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut rest = src;
    while let Some(c) = rest.chars().next() {
        if c.is_whitespace() {
            rest = &rest[c.len_utf8()..];
        } else if c.is_ascii_digit()
            || (c == '.' && rest[1..].starts_with(|d: char| d.is_ascii_digit()))
        {
            let len = number_len(rest);
            let text = &rest[..len];
            let n: f64 = text
                .parse()
                .map_err(|_| format!("bad number `{text}` in `{src}`"))?;
            tokens.push(Token::Number(n));
            rest = &rest[len..];
        } else if c == '\'' || c == '"' {
            let body = &rest[1..];
            let end = body
                .find(c)
                .ok_or_else(|| format!("unterminated string literal in `{src}`"))?;
            tokens.push(Token::Str(body[..end].to_owned()));
            rest = &body[end + 1..];
        } else if c.is_alphabetic() || c == '_' {
            let len = rest
                .find(|d: char| !(d.is_alphanumeric() || d == '_'))
                .unwrap_or(rest.len());
            tokens.push(Token::Ident(rest[..len].to_owned()));
            rest = &rest[len..];
        } else if c == '(' {
            tokens.push(Token::LParen);
            rest = &rest[1..];
        } else if c == ')' {
            tokens.push(Token::RParen);
            rest = &rest[1..];
        } else if c == ',' {
            tokens.push(Token::Comma);
            rest = &rest[1..];
        } else if let Some(op) = OPERATORS.iter().find(|op| rest.starts_with(**op)) {
            tokens.push(Token::Op(op));
            rest = &rest[op.len()..];
        } else {
            return Err(format!("unexpected `{c}` in `{src}`"));
        }
    }
    Ok(tokens)
}

/// Length of the numeric literal at the start of `s` (digits, one dot, optional exponent).
fn number_len(s: &str) -> usize {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
        i += 1;
    }
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        let mut j = i + 1;
        if j < bytes.len() && (bytes[j] == b'+' || bytes[j] == b'-') {
            j += 1;
        }
        if j < bytes.len() && bytes[j].is_ascii_digit() {
            i = j;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
    }
    i
}

/// Parsed expression; evaluated afterwards so that `and` / `or` / comparison chains can stop early.
#[derive(Debug, Clone, PartialEq)]
enum Expr {
    Literal(Value),
    Name(String),
    Call(String, Vec<Expr>),
    Neg(Box<Expr>),
    Not(Box<Expr>),
    /// `+ - * / // % **`.
    Binary(&'static str, Box<Expr>, Box<Expr>),
    /// `a < b <= c`: the first operand, then (operator, operand) pairs.
    Compare(Box<Expr>, Vec<(&'static str, Expr)>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
}

/// Recursive-descent parser over the tokens, with Python's precedence: `or` < `and` < `not` < comparison < `+ -` < `* / // %` < unary `-` < `**` < atom.
struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn take_op(&mut self, ops: &[&str]) -> Option<&'static str> {
        let op = match self.peek() {
            Some(Token::Op(op)) if ops.contains(op) => *op,
            _ => return None,
        };
        self.pos += 1;
        Some(op)
    }

    fn take_ident(&mut self, name: &str) -> bool {
        let found = matches!(self.peek(), Some(Token::Ident(id)) if id == name);
        if found {
            self.pos += 1;
        }
        found
    }

    fn or_expr(&mut self) -> Result<Expr, String> {
        let mut left = self.and_expr()?;
        while self.take_ident("or") {
            left = Expr::Or(Box::new(left), Box::new(self.and_expr()?));
        }
        Ok(left)
    }

    fn and_expr(&mut self) -> Result<Expr, String> {
        let mut left = self.not_expr()?;
        while self.take_ident("and") {
            left = Expr::And(Box::new(left), Box::new(self.not_expr()?));
        }
        Ok(left)
    }

    fn not_expr(&mut self) -> Result<Expr, String> {
        if self.take_ident("not") {
            return Ok(Expr::Not(Box::new(self.not_expr()?)));
        }
        self.comparison()
    }

    fn comparison(&mut self) -> Result<Expr, String> {
        let first = self.arith()?;
        let mut rest = Vec::new();
        while let Some(op) = self.take_op(&["==", "!=", "<=", ">=", "<", ">"]) {
            rest.push((op, self.arith()?));
        }
        Ok(if rest.is_empty() {
            first
        } else {
            Expr::Compare(Box::new(first), rest)
        })
    }

    fn arith(&mut self) -> Result<Expr, String> {
        let mut left = self.term()?;
        while let Some(op) = self.take_op(&["+", "-"]) {
            left = Expr::Binary(op, Box::new(left), Box::new(self.term()?));
        }
        Ok(left)
    }

    fn term(&mut self) -> Result<Expr, String> {
        let mut left = self.factor()?;
        while let Some(op) = self.take_op(&["*", "/", "//", "%"]) {
            left = Expr::Binary(op, Box::new(left), Box::new(self.factor()?));
        }
        Ok(left)
    }

    fn factor(&mut self) -> Result<Expr, String> {
        if self.take_op(&["-"]).is_some() {
            return Ok(Expr::Neg(Box::new(self.factor()?)));
        }
        if self.take_op(&["+"]).is_some() {
            return self.factor();
        }
        self.power()
    }

    fn power(&mut self) -> Result<Expr, String> {
        let base = self.atom()?;
        if self.take_op(&["**"]).is_some() {
            return Ok(Expr::Binary("**", Box::new(base), Box::new(self.factor()?)));
        }
        Ok(base)
    }

    fn atom(&mut self) -> Result<Expr, String> {
        let token = self
            .peek()
            .cloned()
            .ok_or_else(|| "expression ends unexpectedly".to_owned())?;
        self.pos += 1;
        match token {
            Token::Number(n) => Ok(Expr::Literal(Value::Number(n))),
            Token::Str(s) => Ok(Expr::Literal(Value::Str(s))),
            Token::LParen => {
                let inner = self.or_expr()?;
                match self.peek() {
                    Some(Token::RParen) => {
                        self.pos += 1;
                        Ok(inner)
                    }
                    _ => Err("missing `)`".to_owned()),
                }
            }
            Token::Ident(name) if matches!(self.peek(), Some(Token::LParen)) => {
                self.pos += 1;
                let mut args = Vec::new();
                if !matches!(self.peek(), Some(Token::RParen)) {
                    args.push(self.or_expr()?);
                    while matches!(self.peek(), Some(Token::Comma)) {
                        self.pos += 1;
                        args.push(self.or_expr()?);
                    }
                }
                match self.peek() {
                    Some(Token::RParen) => self.pos += 1,
                    _ => {
                        return Err(format!("missing `)` after the arguments of {name}()"));
                    }
                }
                Ok(Expr::Call(name, args))
            }
            Token::Ident(name) => Ok(Expr::Name(name)),
            other => Err(format!("unexpected `{other}`")),
        }
    }
}

fn eval(expr: &Expr, lookup: &dyn Fn(&str) -> Option<Value>) -> Result<Value, String> {
    match expr {
        Expr::Literal(value) => Ok(value.clone()),
        Expr::Name(name) => {
            if let Some(value) = lookup(name) {
                return Ok(value);
            }
            match name.as_str() {
                "pi" => Ok(Value::Number(std::f64::consts::PI)),
                "true" | "True" => Ok(Value::Bool(true)),
                "false" | "False" => Ok(Value::Bool(false)),
                _ => Err(format!("undefined name `{name}`")),
            }
        }
        Expr::Call(name, args) => {
            let args: Vec<Value> = args
                .iter()
                .map(|arg| eval(arg, lookup))
                .collect::<Result<_, _>>()?;
            call(name, &args)
        }
        Expr::Neg(inner) => finite(-eval(inner, lookup)?.number("unary `-`")?),
        Expr::Not(inner) => Ok(Value::Bool(!eval(inner, lookup)?.python_truth())),
        Expr::Binary(op, left, right) => binary(op, eval(left, lookup)?, eval(right, lookup)?),
        Expr::Compare(first, rest) => {
            let mut left = eval(first, lookup)?;
            for (op, operand) in rest {
                let right = eval(operand, lookup)?;
                if !compare(op, &left, &right)? {
                    return Ok(Value::Bool(false));
                }
                left = right;
            }
            Ok(Value::Bool(true))
        }
        Expr::And(left, right) => {
            let left = eval(left, lookup)?;
            if left.python_truth() {
                eval(right, lookup)
            } else {
                Ok(left)
            }
        }
        Expr::Or(left, right) => {
            let left = eval(left, lookup)?;
            if left.python_truth() {
                Ok(left)
            } else {
                eval(right, lookup)
            }
        }
    }
}

fn binary(op: &str, left: Value, right: Value) -> Result<Value, String> {
    if let ("+", Value::Str(a), Value::Str(b)) = (op, &left, &right) {
        return Ok(Value::Str(format!("{a}{b}")));
    }
    let what = format!("`{op}`");
    let a = left.number(&what)?;
    let b = right.number(&what)?;
    if matches!(op, "/" | "//" | "%") && b == 0.0 {
        return Err(format!("division by zero in `{a} {op} {b}`"));
    }
    finite(match op {
        "+" => a + b,
        "-" => a - b,
        "*" => a * b,
        "/" => a / b,
        "//" => (a / b).floor(),
        "%" => a - b * (a / b).floor(),
        _ => a.powf(b),
    })
}

fn compare(op: &str, left: &Value, right: &Value) -> Result<bool, String> {
    use std::cmp::Ordering;
    let ordering = match (left, right) {
        (Value::Str(a), Value::Str(b)) => Some(a.cmp(b)),
        (Value::Str(_), _) | (_, Value::Str(_)) => None,
        (a, b) => a
            .number("comparison")?
            .partial_cmp(&b.number("comparison")?),
    };
    Ok(match (op, ordering) {
        ("==", o) => o == Some(Ordering::Equal),
        ("!=", o) => o != Some(Ordering::Equal),
        (_, None) => {
            return Err(format!("cannot order `{left}` and `{right}` with `{op}`"));
        }
        ("<", Some(o)) => o == Ordering::Less,
        ("<=", Some(o)) => o != Ordering::Greater,
        (">", Some(o)) => o == Ordering::Greater,
        (_, Some(o)) => o != Ordering::Less,
    })
}

/// The math functions xacro files commonly use; everything else is outside this subset by design.
fn call(name: &str, args: &[Value]) -> Result<Value, String> {
    let one = || match args {
        [value] => value.number(&format!("{name}()")),
        _ => Err(format!(
            "{name}() takes exactly one argument, got {}",
            args.len()
        )),
    };
    let result = match name {
        "radians" => one()?.to_radians(),
        "degrees" => one()?.to_degrees(),
        "sin" => one()?.sin(),
        "cos" => one()?.cos(),
        "sqrt" => one()?.sqrt(),
        "abs" => one()?.abs(),
        _ => {
            return Err(format!(
                "unsupported function {name}() (visor's xacro subset has radians, degrees, sin, cos, sqrt, abs)"
            ));
        }
    };
    finite(result)
}

fn finite(n: f64) -> Result<Value, String> {
    if n.is_finite() {
        Ok(Value::Number(n))
    } else {
        Err("result is not a finite number".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup(pairs: Vec<(&'static str, Value)>) -> impl Fn(&str) -> Option<Value> {
        move |name| {
            pairs
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| v.clone())
        }
    }

    fn no_subst(inner: &str) -> Result<String, String> {
        Err(format!("unexpected $({inner})"))
    }

    fn eval(src: &str) -> Value {
        eval_expr(src, &lookup(vec![])).unwrap_or_else(|e| panic!("{src}: {e}"))
    }

    #[test]
    fn arithmetic_follows_python_precedence() {
        assert_eq!(eval("1 + 2 * 3"), Value::Number(7.0));
        assert_eq!(eval("(1 + 2) * 3"), Value::Number(9.0));
        assert_eq!(eval("2 ** 3 ** 2"), Value::Number(512.0));
        assert_eq!(eval("-2 ** 2"), Value::Number(-4.0));
        assert_eq!(eval("7 // 2"), Value::Number(3.0));
        assert_eq!(eval("-7 // 2"), Value::Number(-4.0));
        assert_eq!(eval("-7 % 3"), Value::Number(2.0));
        assert_eq!(eval("255/255"), Value::Number(1.0));
        assert_eq!(eval("0.1 - 0.07 / 2"), Value::Number(0.1 - 0.07 / 2.0));
    }

    #[test]
    fn names_come_from_the_lookup_then_the_builtins() {
        let scope = lookup(vec![
            ("PI", Value::Number(3.0)),
            ("suffix", Value::Str("left".into())),
        ]);
        assert_eq!(eval_expr("PI / 2", &scope).unwrap(), Value::Number(1.5));
        assert_eq!(
            eval_expr("suffix", &scope).unwrap(),
            Value::Str("left".into())
        );
        assert_eq!(
            eval_expr("pi", &scope).unwrap(),
            Value::Number(std::f64::consts::PI)
        );
        assert_eq!(
            eval_expr("true and not False", &scope).unwrap(),
            Value::Bool(true)
        );
        let error = eval_expr("wheel_radius * 2", &scope).unwrap_err();
        assert!(error.contains("undefined name `wheel_radius`"), "{error}");
    }

    #[test]
    fn comparisons_and_boolean_operators() {
        let scope = lookup(vec![
            ("can_device", Value::Str("dummy".into())),
            ("n", Value::Number(3.0)),
        ]);
        assert_eq!(
            eval_expr("can_device == 'dummy'", &scope).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            eval_expr("can_device != \"dummy\"", &scope).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(eval_expr("1 < n <= 3", &scope).unwrap(), Value::Bool(true));
        assert_eq!(eval_expr("1 < n < 3", &scope).unwrap(), Value::Bool(false));
        assert_eq!(eval_expr("n == 'x'", &scope).unwrap(), Value::Bool(false));
        assert_eq!(eval_expr("n or 0", &scope).unwrap(), Value::Number(3.0));
        assert_eq!(eval_expr("0 or n", &scope).unwrap(), Value::Number(3.0));
        assert_eq!(
            eval_expr("'' or 'b'", &scope).unwrap(),
            Value::Str("b".into())
        );
        assert!(eval_expr("n < 'x'", &scope).is_err());
    }

    #[test]
    fn and_or_and_comparison_chains_short_circuit() {
        // The right operand is never evaluated once the result is known, so a guard can protect a division.
        assert_eq!(eval("false and 1 / 0"), Value::Bool(false));
        assert_eq!(eval("0 and undefined_name"), Value::Number(0.0));
        assert_eq!(eval("1 or 1 / 0"), Value::Number(1.0));
        assert_eq!(eval("'x' or undefined_name"), Value::Str("x".into()));
        assert_eq!(eval("2 < 1 < 1 / 0"), Value::Bool(false));
        assert_eq!(eval("not (1 / 1)"), Value::Bool(false));
        // Without a guard the error still surfaces.
        assert!(eval_expr("true and 1 / 0", &lookup(vec![])).is_err());
        assert!(eval_expr("0 or 1 / 0", &lookup(vec![])).is_err());
        assert!(eval_expr("1 < 2 < 1 / 0", &lookup(vec![])).is_err());
    }

    #[test]
    fn functions_strings_and_errors() {
        assert_eq!(eval("radians(180)"), Value::Number(std::f64::consts::PI));
        assert_eq!(eval("degrees(pi)"), Value::Number(180.0));
        assert_eq!(eval("abs(-2) + sqrt(16)"), Value::Number(6.0));
        assert_eq!(eval("'a' + 'b'"), Value::Str("ab".into()));
        for (src, needle) in [
            ("1 / 0", "division by zero"),
            ("1 // 0", "division by zero"),
            ("", "empty"),
            ("1 +", "ends unexpectedly"),
            ("(1", "missing `)`"),
            ("1 2", "unexpected `2`"),
            ("'abc", "unterminated"),
            ("foo(1)", "unsupported function foo()"),
            ("sin(1, 2)", "exactly one argument"),
            ("[1, 2]", "unexpected `[`"),
            ("x.y", "unexpected `.`"),
            ("'a' - 1", "needs a number"),
            ("sqrt(-1)", "not a finite"),
        ] {
            let error = eval_expr(src, &lookup(vec![])).unwrap_err();
            assert!(error.contains(needle), "{src}: {error}");
        }
    }

    #[test]
    fn numbers_print_without_a_trailing_zero_and_shortest() {
        assert_eq!(eval("6.0").to_string(), "6");
        assert_eq!(eval("pi / 2").to_string(), "1.5707963267948966");
        assert_eq!(eval("1e-5").to_string(), "0.00001");
        assert_eq!(eval(".5 + 1.").to_string(), "1.5");
        assert_eq!(Value::Bool(true).to_string(), "True");
    }

    #[test]
    fn literal_and_truthy_follow_xacro() {
        assert_eq!(Value::literal("0.0655"), Value::Number(0.0655));
        assert_eq!(Value::literal(" 3 "), Value::Number(3.0));
        assert_eq!(Value::literal("1e3"), Value::Number(1000.0));
        assert_eq!(Value::literal("left"), Value::Str("left".into()));
        assert_eq!(Value::literal("inf"), Value::Str("inf".into()));
        assert_eq!(Value::literal("nan"), Value::Str("nan".into()));
        assert_eq!(Value::literal(""), Value::Str(String::new()));
        for (value, expected) in [
            (Value::Bool(false), false),
            (Value::Number(2.0), true),
            (Value::Number(0.0), false),
            (Value::Str("True".into()), true),
            (Value::Str(" false ".into()), false),
            (Value::Str("1".into()), true),
            (Value::Str("0".into()), false),
        ] {
            assert_eq!(value.truthy().unwrap(), expected, "{value:?}");
        }
        assert!(Value::Str("yes".into()).truthy().is_err());
        assert!(Value::Str(String::new()).truthy().is_err());
    }

    #[test]
    fn text_interpolation_mixes_expressions_substitutions_and_escapes() {
        let scope = lookup(vec![
            ("suffix", Value::Str("left".into())),
            ("PI", Value::Number(3.0)),
        ]);
        let subst = |inner: &str| -> Result<String, String> {
            match inner {
                "find pkg" => Ok("/ws/pkg".to_owned()),
                "optenv ROBOT_MODEL model-a" => Ok("model-a".to_owned()),
                other => Err(format!("unknown $({other})")),
            }
        };
        let text = |t: &str| eval_text(t, &scope, &subst).unwrap_or_else(|e| panic!("{t}: {e}"));
        assert_eq!(
            text("${suffix}_wheel_link"),
            Value::Str("left_wheel_link".into())
        );
        assert_eq!(text("${PI / 2} 0 0"), Value::Str("1.5 0 0".into()));
        assert_eq!(
            text("$(find pkg)/urdf/$(optenv ROBOT_MODEL model-a).xacro"),
            Value::Str("/ws/pkg/urdf/model-a.xacro".into())
        );
        assert_eq!(
            text("cost $$5 and $ alone"),
            Value::Str("cost $5 and $ alone".into())
        );
        // `$${` and `$$(` are how a file writes a literal `${` / `$(` (no expansion happens on them).
        assert_eq!(
            text("$${suffix} $$(find pkg)"),
            Value::Str("${suffix} $(find pkg)".into())
        );
        assert_eq!(text("plain"), Value::Str("plain".into()));
        // A lone expression keeps its type, so a property can hold a number or a bool.
        assert_eq!(text("${PI / 2}"), Value::Number(1.5));
        assert_eq!(text("${suffix == 'left'}"), Value::Bool(true));
        assert_eq!(text("${PI}${PI}"), Value::Str("33".into()));
        for (t, needle) in [
            ("${PI", "unterminated `${`"),
            ("$(find pkg", "unterminated `$(`"),
            ("$(unknown)", "unknown $(unknown)"),
            ("${nope}", "undefined name `nope`"),
        ] {
            let error = eval_text(t, &scope, &subst).unwrap_err();
            assert!(error.contains(needle), "{t}: {error}");
        }
        assert!(eval_text("x", &scope, &no_subst).is_ok());
    }
}
