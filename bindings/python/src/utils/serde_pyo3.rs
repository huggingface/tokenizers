use serde::de::value::Error;
use serde::{ser, Serialize};
type Result<T> = ::std::result::Result<T, Error>;

pub struct Serializer {
    // This string starts empty and JSON is appended as values are serialized.
    output: String,
    /// Each levels remembers its own number of elements
    num_elements: Vec<usize>,
    max_elements: usize,
    level: usize,
    max_depth: usize,
    /// Maximum string representation
    /// Useful to ellipsis precompiled_charmap
    max_string: usize,
}

// By convention, the public API of a Serde serializer is one or more `to_abc`
// functions such as `to_string`, `to_bytes`, or `to_writer` depending on what
// Rust types the serializer is able to produce as output.
//
// This basic serializer supports only `to_string`.
pub fn to_string<T>(value: &T) -> Result<String>
where
    T: Serialize,
{
    let max_depth = 20;
    let max_elements = 6;
    let max_string = 100;
    let mut serializer = Serializer {
        output: String::new(),
        level: 0,
        max_depth,
        max_elements,
        num_elements: vec![0; max_depth],
        max_string,
    };
    value.serialize(&mut serializer)?;
    Ok(serializer.output)
}

pub fn repr<T>(value: &T) -> Result<String>
where
    T: Serialize,
{
    let max_depth = 200;
    let max_string = usize::MAX;
    let mut serializer = Serializer {
        output: String::new(),
        level: 0,
        max_depth,
        max_elements: 100,
        num_elements: vec![0; max_depth],
        max_string,
    };
    value.serialize(&mut serializer)?;
    Ok(serializer.output)
}

impl ser::Serializer for &mut Serializer {
    // The output type produced by this `Serializer` during successful
    // serialization. Most serializers that produce text or binary output should
    // set `Ok = ()` and serialize into an `io::Write` or buffer contained
    // within the `Serializer` instance, as happens here. Serializers that build
    // in-memory data structures may be simplified by using `Ok` to propagate
    // the data structure around.
    type Ok = ();

    // The error type when some error occurs during serialization.
    type Error = Error;

    // Associated types for keeping track of additional state while serializing
    // compound data structures like sequences and maps. In this case no
    // additional state is required beyond what is already stored in the
    // Serializer struct.
    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = Self;
    type SerializeMap = Self;
    type SerializeStruct = Self;
    type SerializeStructVariant = Self;

    // Here we go with the simple methods. The following 12 methods receive one
    // of the primitive types of the data model and map it to JSON by appending
    // into the output string.
    fn serialize_bool(self, v: bool) -> Result<()> {
        self.output += if v { "True" } else { "False" };
        Ok(())
    }

    // JSON does not distinguish between different sizes of integers, so all
    // signed integers will be serialized the same and all unsigned integers
    // will be serialized the same. Other formats, especially compact binary
    // formats, may need independent logic for the different sizes.
    fn serialize_i8(self, v: i8) -> Result<()> {
        self.serialize_i64(i64::from(v))
    }

    fn serialize_i16(self, v: i16) -> Result<()> {
        self.serialize_i64(i64::from(v))
    }

    fn serialize_i32(self, v: i32) -> Result<()> {
        self.serialize_i64(i64::from(v))
    }

    // Not particularly efficient but this is example code anyway. A more
    // performant approach would be to use the `itoa` crate.
    fn serialize_i64(self, v: i64) -> Result<()> {
        self.output += &v.to_string();
        Ok(())
    }

    fn serialize_u8(self, v: u8) -> Result<()> {
        self.serialize_u64(u64::from(v))
    }

    fn serialize_u16(self, v: u16) -> Result<()> {
        self.serialize_u64(u64::from(v))
    }

    fn serialize_u32(self, v: u32) -> Result<()> {
        self.serialize_u64(u64::from(v))
    }

    fn serialize_u64(self, v: u64) -> Result<()> {
        self.output += &v.to_string();
        Ok(())
    }

    fn serialize_f32(self, v: f32) -> Result<()> {
        self.serialize_f64(f64::from(v))
    }

    fn serialize_f64(self, v: f64) -> Result<()> {
        self.output += &v.to_string();
        Ok(())
    }

    // Serialize a char as a single-character string. Other formats may
    // represent this differently.
    fn serialize_char(self, v: char) -> Result<()> {
        self.serialize_str(&v.to_string())
    }

    // This only works for strings that don't require escape sequences but you
    // get the idea. For example it would emit invalid JSON if the input string
    // contains a '"' character.
    fn serialize_str(self, v: &str) -> Result<()> {
        self.output += "\"";
        if v.len() > self.max_string {
            self.output += &v[..self.max_string];
            self.output += "...";
        } else {
            self.output += v;
        }
        self.output += "\"";
        Ok(())
    }

    // Serialize a byte array as an array of bytes. Could also use a base64
    // string here. Binary formats will typically represent byte arrays more
    // compactly.
    fn serialize_bytes(self, v: &[u8]) -> Result<()> {
        use serde::ser::SerializeSeq;
        let mut seq = self.serialize_seq(Some(v.len()))?;
        for byte in v {
            seq.serialize_element(byte)?;
        }
        seq.end()
    }

    // An absent optional is represented as the JSON `null`.
    fn serialize_none(self) -> Result<()> {
        self.serialize_unit()
    }

    // A present optional is represented as just the contained value. Note that
    // this is a lossy representation. For example the values `Some(())` and
    // `None` both serialize as just `null`. Unfortunately this is typically
    // what people expect when working with JSON. Other formats are encouraged
    // to behave more intelligently if possible.
    fn serialize_some<T>(self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(self)
    }

    // In Serde, unit means an anonymous value containing no data. Map this to
    // JSON as `null`.
    fn serialize_unit(self) -> Result<()> {
        self.output += "None";
        Ok(())
    }

    // Unit struct means a named value containing no data. Again, since there is
    // no data, map this to JSON as `null`. There is no need to serialize the
    // name in most formats.
    fn serialize_unit_struct(self, _name: &'static str) -> Result<()> {
        self.serialize_unit()
    }

    // When serializing a unit variant (or any other kind of variant), formats
    // can choose whether to keep track of it by index or by name. Binary
    // formats typically use the index of the variant and human-readable formats
    // typically use the name.
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
    ) -> Result<()> {
        // self.serialize_str(variant)
        self.output += variant;
        Ok(())
    }

    // As is done here, serializers are encouraged to treat newtype structs as
    // insignificant wrappers around the data they contain.
    fn serialize_newtype_struct<T>(self, _name: &'static str, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(self)
    }

    // Note that newtype variant (and all of the other variant serialization
    // methods) refer exclusively to the "externally tagged" enum
    // representation.
    //
    // Serialize this to JSON in externally tagged form as `{ NAME: VALUE }`.
    fn serialize_newtype_variant<T>(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        // variant.serialize(&mut *self)?;
        self.output += variant;
        self.output += "(";
        value.serialize(&mut *self)?;
        self.output += ")";
        Ok(())
    }

    // Now we get to the serialization of compound types.
    //
    // The start of the sequence, each value, and the end are three separate
    // method calls. This one is responsible only for serializing the start,
    // which in JSON is `[`.
    //
    // The length of the sequence may or may not be known ahead of time. This
    // doesn't make a difference in JSON because the length is not represented
    // explicitly in the serialized form. Some serializers may only be able to
    // support sequences for which the length is known up front.
    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq> {
        self.output += "[";
        self.level = std::cmp::min(self.max_depth - 1, self.level + 1);
        self.num_elements[self.level] = 0;
        Ok(self)
    }

    // Tuples look just like sequences in JSON. Some formats may be able to
    // represent tuples more efficiently by omitting the length, since tuple
    // means that the corresponding `Deserialize implementation will know the
    // length without needing to look at the serialized data.
    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple> {
        self.output += "(";
        self.level = std::cmp::min(self.max_depth - 1, self.level + 1);
        self.num_elements[self.level] = 0;
        Ok(self)
    }

    // Tuple structs look just like sequences in JSON.
    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct> {
        self.serialize_tuple(len)
    }

    // Tuple variants are represented in JSON as `{ NAME: [DATA...] }`. Again
    // this method is only responsible for the externally tagged representation.
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant> {
        // variant.serialize(&mut *self)?;
        self.output += variant;
        self.output += "(";
        self.level = std::cmp::min(self.max_depth - 1, self.level + 1);
        self.num_elements[self.level] = 0;
        Ok(self)
    }

    // Maps are represented in JSON as `{ K: V, K: V, ... }`.
    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap> {
        self.output += "{";
        self.level = std::cmp::min(self.max_depth - 1, self.level + 1);
        self.num_elements[self.level] = 0;
        Ok(self)
    }

    // Structs look just like maps in JSON. In particular, JSON requires that we
    // serialize the field names of the struct. Other formats may be able to
    // omit the field names when serializing structs because the corresponding
    // Deserialize implementation is required to know what the keys are without
    // looking at the serialized data.
    fn serialize_struct(self, name: &'static str, _len: usize) -> Result<Self::SerializeStruct> {
        // self.serialize_map(Some(len))
        // name.serialize(&mut *self)?;
        if let Some(stripped) = name.strip_suffix("Helper") {
            self.output += stripped;
        } else {
            self.output += name
        }
        self.output += "(";
        self.level = std::cmp::min(self.max_depth - 1, self.level + 1);
        self.num_elements[self.level] = 0;
        Ok(self)
    }

    // Struct variants are represented in JSON as `{ NAME: { K: V, ... } }`.
    // This is the externally tagged representation.
    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant> {
        // variant.serialize(&mut *self)?;
        self.output += variant;
        self.output += "(";
        self.level = std::cmp::min(self.max_depth - 1, self.level + 1);
        self.num_elements[self.level] = 0;
        Ok(self)
    }
}

// The following 7 impls deal with the serialization of compound types like
// sequences and maps. Serialization of such types is begun by a Serializer
// method and followed by zero or more calls to serialize individual elements of
// the compound type and one call to end the compound type.
//
// This impl is SerializeSeq so these methods are called after `serialize_seq`
// is called on the Serializer.
impl ser::SerializeSeq for &mut Serializer {
    // Must match the `Ok` type of the serializer.
    type Ok = ();
    // Must match the `Error` type of the serializer.
    type Error = Error;

    // Serialize a single element of the sequence.
    fn serialize_element<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        self.num_elements[self.level] += 1;
        let num_elements = self.num_elements[self.level];
        if num_elements < self.max_elements {
            if !self.output.ends_with('[') {
                self.output += ", ";
            }
            value.serialize(&mut **self)
        } else {
            if num_elements == self.max_elements {
                self.output += ", ...";
            }
            Ok(())
        }
    }

    // Close the sequence.
    fn end(self) -> Result<()> {
        self.num_elements[self.level] = 0;
        self.level = self.level.saturating_sub(1);
        self.output += "]";
        Ok(())
    }
}

// Same thing but for tuples.
impl ser::SerializeTuple for &mut Serializer {
    type Ok = ();
    type Error = Error;

    fn serialize_element<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        self.num_elements[self.level] += 1;
        let num_elements = self.num_elements[self.level];
        if num_elements < self.max_elements {
            if !self.output.ends_with('(') {
                self.output += ", ";
            }
            value.serialize(&mut **self)
        } else {
            if num_elements == self.max_elements {
                self.output += ", ...";
            }
            Ok(())
        }
    }

    fn end(self) -> Result<()> {
        self.num_elements[self.level] = 0;
        self.level = self.level.saturating_sub(1);
        self.output += ")";
        Ok(())
    }
}

// Same thing but for tuple structs.
impl ser::SerializeTupleStruct for &mut Serializer {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        self.num_elements[self.level] += 1;
        let num_elements = self.num_elements[self.level];
        if num_elements < self.max_elements {
            if !self.output.ends_with('(') {
                self.output += ", ";
            }
            value.serialize(&mut **self)
        } else {
            if num_elements == self.max_elements {
                self.output += ", ...";
            }
            Ok(())
        }
    }

    fn end(self) -> Result<()> {
        self.num_elements[self.level] = 0;
        self.level = self.level.saturating_sub(1);
        self.output += ")";
        Ok(())
    }
}

// Tuple variants are a little different. Refer back to the
// `serialize_tuple_variant` method above:
//
//    self.output += "{";
//    variant.serialize(&mut *self)?;
//    self.output += ":[";
//
// So the `end` method in this impl is responsible for closing both the `]` and
// the `}`.
impl ser::SerializeTupleVariant for &mut Serializer {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        self.num_elements[self.level] += 1;
        let num_elements = self.num_elements[self.level];
        if num_elements < self.max_elements {
            if !self.output.ends_with('(') {
                self.output += ", ";
            }
            value.serialize(&mut **self)
        } else {
            if num_elements == self.max_elements {
                self.output += ", ...";
            }
            Ok(())
        }
    }

    fn end(self) -> Result<()> {
        self.num_elements[self.level] = 0;
        self.level = self.level.saturating_sub(1);
        self.output += ")";
        Ok(())
    }
}

// Some `Serialize` types are not able to hold a key and value in memory at the
// same time so `SerializeMap` implementations are required to support
// `serialize_key` and `serialize_value` individually.
//
// There is a third optional method on the `SerializeMap` trait. The
// `serialize_entry` method allows serializers to optimize for the case where
// key and value are both available simultaneously. In JSON it doesn't make a
// difference so the default behavior for `serialize_entry` is fine.
impl ser::SerializeMap for &mut Serializer {
    type Ok = ();
    type Error = Error;

    // The Serde data model allows map keys to be any serializable type. JSON
    // only allows string keys so the implementation below will produce invalid
    // JSON if the key serializes as something other than a string.
    //
    // A real JSON serializer would need to validate that map keys are strings.
    // This can be done by using a different Serializer to serialize the key
    // (instead of `&mut **self`) and having that other serializer only
    // implement `serialize_str` and return an error on any other data type.
    fn serialize_key<T>(&mut self, key: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        self.num_elements[self.level] += 1;
        let num_elements = self.num_elements[self.level];
        if num_elements < self.max_elements {
            if !self.output.ends_with('{') {
                self.output += ", ";
            }
            key.serialize(&mut **self)
        } else {
            if num_elements == self.max_elements {
                self.output += ", ...";
            }
            Ok(())
        }
    }

    // It doesn't make a difference whether the colon is printed at the end of
    // `serialize_key` or at the beginning of `serialize_value`. In this case
    // the code is a bit simpler having it here.
    fn serialize_value<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        let num_elements = self.num_elements[self.level];
        if num_elements < self.max_elements {
            self.output += ":";
            value.serialize(&mut **self)
        } else {
            Ok(())
        }
    }

    fn end(self) -> Result<()> {
        self.num_elements[self.level] = 0;
        self.level = self.level.saturating_sub(1);
        self.output += "}";
        Ok(())
    }
}

// Structs are like maps in which the keys are constrained to be compile-time
// constant strings.
impl ser::SerializeStruct for &mut Serializer {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        if !self.output.ends_with('(') {
            self.output += ", ";
        }
        // key.serialize(&mut **self)?;
        if key != "type" {
            self.output += key;
            self.output += "=";
            value.serialize(&mut **self)
        } else {
            Ok(())
        }
    }

    fn end(self) -> Result<()> {
        self.num_elements[self.level] = 0;
        self.level = self.level.saturating_sub(1);
        self.output += ")";
        Ok(())
    }
}

// Similar to `SerializeTupleVariant`, here the `end` method is responsible for
// closing both of the curly braces opened by `serialize_struct_variant`.
impl ser::SerializeStructVariant for &mut Serializer {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        if !self.output.ends_with('(') {
            self.output += ", ";
        }
        // key.serialize(&mut **self)?;
        self.output += key;
        self.output += "=";
        value.serialize(&mut **self)
    }

    fn end(self) -> Result<()> {
        self.num_elements[self.level] = 0;
        self.level = self.level.saturating_sub(1);
        self.output += ")";
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_basic() {
    assert_eq!(to_string(&true).unwrap(), "True");
    assert_eq!(to_string(&Some(1)).unwrap(), "1");
    assert_eq!(to_string(&None::<usize>).unwrap(), "None");
}

#[test]
fn test_struct() {
    #[derive(Serialize)]
    struct Test {
        int: u32,
        seq: Vec<&'static str>,
    }

    let test = Test {
        int: 1,
        seq: vec!["a", "b"],
    };
    let expected = r#"Test(int=1, seq=["a", "b"])"#;
    assert_eq!(to_string(&test).unwrap(), expected);
}

#[test]
fn test_enum() {
    #[derive(Serialize)]
    enum E {
        Unit,
        Newtype(u32),
        Tuple(u32, u32),
        Struct { a: u32 },
    }

    let u = E::Unit;
    let expected = r#"Unit"#;
    assert_eq!(to_string(&u).unwrap(), expected);

    let n = E::Newtype(1);
    let expected = r#"Newtype(1)"#;
    assert_eq!(to_string(&n).unwrap(), expected);

    let t = E::Tuple(1, 2);
    let expected = r#"Tuple(1, 2)"#;
    assert_eq!(to_string(&t).unwrap(), expected);

    let s = E::Struct { a: 1 };
    let expected = r#"Struct(a=1)"#;
    assert_eq!(to_string(&s).unwrap(), expected);
}

#[test]
fn test_enum_untagged() {
    #[derive(Serialize)]
    #[serde(untagged)]
    enum E {
        Unit,
        Newtype(u32),
        Tuple(u32, u32),
        Struct { a: u32 },
    }

    let u = E::Unit;
    let expected = r#"None"#;
    assert_eq!(to_string(&u).unwrap(), expected);

    let n = E::Newtype(1);
    let expected = r#"1"#;
    assert_eq!(to_string(&n).unwrap(), expected);

    let t = E::Tuple(1, 2);
    let expected = r#"(1, 2)"#;
    assert_eq!(to_string(&t).unwrap(), expected);

    let s = E::Struct { a: 1 };
    let expected = r#"E(a=1)"#;
    assert_eq!(to_string(&s).unwrap(), expected);
}

#[test]
fn test_struct_tagged() {
    #[derive(Serialize)]
    #[serde(untagged)]
    enum E {
        A(A),
    }

    #[derive(Serialize)]
    #[serde(tag = "type")]
    struct A {
        a: bool,
        b: usize,
    }

    let u = A { a: true, b: 1 };
    // let expected = r#"A(type="A", a=True, b=1)"#;
    // No we skip all `type` manually inserted variants.
    let expected = r#"A(a=True, b=1)"#;
    assert_eq!(to_string(&u).unwrap(), expected);

    let u = E::A(A { a: true, b: 1 });
    let expected = r#"A(a=True, b=1)"#;
    assert_eq!(to_string(&u).unwrap(), expected);
}

#[test]
fn test_flatten() {
    #[derive(Serialize)]
    struct A {
        a: bool,
        b: usize,
    }

    #[derive(Serialize)]
    struct B {
        c: A,
        d: usize,
    }

    #[derive(Serialize)]
    struct C {
        #[serde(flatten)]
        c: A,
        d: usize,
    }

    #[derive(Serialize)]
    #[serde(transparent)]
    struct D {
        e: A,
    }

    let u = B {
        c: A { a: true, b: 1 },
        d: 2,
    };
    let expected = r#"B(c=A(a=True, b=1), d=2)"#;
    assert_eq!(to_string(&u).unwrap(), expected);

    let u = C {
        c: A { a: true, b: 1 },
        d: 2,
    };
    // XXX This is unfortunate but true, flatten forces the serialization
    // to use the serialize_map  without any means for the Serializer to know about this
    // flattening attempt
    let expected = r#"{"a":True, "b":1, "d":2}"#;
    assert_eq!(to_string(&u).unwrap(), expected);

    let u = D {
        e: A { a: true, b: 1 },
    };
    let expected = r#"A(a=True, b=1)"#;
    assert_eq!(to_string(&u).unwrap(), expected);
}
