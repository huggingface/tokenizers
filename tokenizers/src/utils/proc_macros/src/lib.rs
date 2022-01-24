extern crate proc_macro;
use ::proc_macro::*;

// Original: https://crates.io/crates/macro_rules_attribute
/// Applies the given `macro_rules!` macro to the decorated item.
///
/// This, as with any `proc_macro_attribute`, **consumes** the item it
/// decorates: it is the `macro_rules!` macro job to generate it (_it is thus
/// able to modify it_!).
///
/// For a version with "read-only" access to the item it decorates, see
/// [`macro_rules_derive`][`macro@macro_rules_derive`].
///
/// # Example
///
/// Deriving getters for a (non-generic) `struct`:
///
/// ```rust
/// # macro_rules! ignore {($($tt:tt)*) => () }
/// # ignore! {
/// #[macro_use]
/// extern crate macro_rules_attribute;
/// # }
///
/// macro_rules! make_getters {(
///     $(#[$struct_meta:meta])*
///     $struct_vis:vis
///     struct $StructName:ident {
///         $(
///             $(#[$field_meta:meta])*
///             $field_vis:vis // this visibility will be applied to the getters instead
///             $field_name:ident : $field_ty:ty
///         ),* $(,)?
///     }
/// ) => (
///     // First, generate the struct definition we have been given, but with
///     // private fields instead.
///     $(#[$struct_meta])*
///     $struct_vis
///     struct $StructName {
///         $(
///             $(#[$field_meta])*
///             // notice the lack of visibility => private fields
///             $field_name: $field_ty,
///         )*
///     }
///
///     // Then, implement the getters:
///     impl $StructName {
///         $(
///             #[inline]
///             $field_vis
///             fn $field_name (self: &'_ Self)
///                 -> &'_ $field_ty
///             {
///                 &self.$field_name
///             }
///         )*
///     }
/// )}
///
/// mod example {
/// # use ::macro_rules_attribute_proc_macro::macro_rules_attribute;
///     #[macro_rules_attribute(make_getters!)]
///     /// The macro handles meta attributes such as docstrings
///     pub
///     struct Person {
///         pub
///         name: String,
///
///         pub
///         age: u8,
///     }
/// }
/// use example::Person;
///
/// fn is_new_born (person: &'_ Person)
///     -> bool
/// {
///     // person.age == 0
///     // ^ error[E0616]: field `age` of struct `example::Person` is private
///     *person.age() == 0
/// }
/// ```
#[proc_macro_attribute] pub
fn macro_rules_attribute (
    attrs: TokenStream,
    input: TokenStream,
)   -> TokenStream
{
    // check that `attrs` is indeed of the form `$macro_name:path !`
    {
        // FIXME: do this properly
        match attrs.clone().into_iter().last() {
            | Some(TokenTree::Punct(ref punct))
                if punct.as_char() == '!'
            => {},

            | _ => {
                panic!("Expected a parameter of the form `macro_name !`");
            },
        }
    }
    let mut ret = attrs;
    ret.extend(::std::iter::once(
        TokenTree::Group(Group::new(
            Delimiter::Brace,
            // FIXME: directly using `input` makes the token stream be seen
            // as a single token tree by the declarative macro !??
            input.into_iter().collect(),
        ))
    ));
    #[cfg(feature = "verbose-expansions")]
    eprintln!("{}", ret);
    ret
}

// Original: https://crates.io/crates/macro_rules_attribute
/// Applies the given `macro_rules!` macro to the decorated item.
///
/// This, as with any `#[derive(...)]`, **does not consume** the item it
/// decorates: instead, it only generates code on top of it.
///
/// # Example
///
/// Implementing `Into<Int>` for a given `#[repr(Int)]` `enum`:
///
/// ```rust
/// # macro_rules! ignore {($($tt:tt)*) => () }
/// # ignore! {
/// #[macro_use]
/// extern crate macro_rules_attribute;
/// # }
///
/// macro_rules! ToInteger {(
///     #[repr($Int:ident)]
///     $(#[$enum_meta:meta])*
///     $pub:vis
///     enum $Enum:ident {
///         $(
///             $Variant:ident $(= $value:expr)?
///         ),* $(,)?
///     }
/// ) => (
///     impl ::core::convert::From<$Enum> for $Int {
///         #[inline]
///         fn from (x: $Enum)
///             -> Self
///         {
///             x as _
///         }
///     }
/// )}
///
/// # use ::macro_rules_attribute_proc_macro::macro_rules_derive;
/// #[macro_rules_derive(ToInteger!)]
/// #[repr(u32)]
/// enum Bool {
///     False,
///     True,
/// }
///
/// fn main ()
/// {
///     assert_eq!(u32::from(Bool::False), 0);
///     assert_eq!(u32::from(Bool::True), 1);
///     // assert_eq!(u8::from(Bool::False), 0);
///     // ^ error[E0277]: the trait bound `u8: std::convert::From<main::Bool>` is not satisfied
/// }
/// ```
#[proc_macro_attribute] pub
fn macro_rules_derive (
    attrs: TokenStream,
    input: TokenStream,
)   -> TokenStream
{
    let mut ret = input.clone();
    ret.extend(macro_rules_attribute(attrs, input));
    ret
}
