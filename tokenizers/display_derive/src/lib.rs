extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{format_ident,quote};
use syn::{parse_macro_input, DeriveInput}; 
mod vendored;
mod parsing;
use vendored::FmtAttribute;

#[proc_macro_derive(Display)]
pub fn display_derive(input: TokenStream) -> TokenStream  {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    let attr_name = "display";
    let attrs = FmtAttributes::parse_attrs(&input.attrs, &attr_name)?
        .map(Spanning::into_inner)
        .unwrap_or_default();
    let trait_ident = format_ident!("display");
    let ident = &input.ident;

    let ctx = (&attrs, ident, &trait_ident, &attr_name);
    let body = match &input.data {
        syn::Data::Struct(s) => expand_struct(s, ctx),
        syn::Data::Enum(e) => expand_enum(e, ctx),
        syn::Data::Union(u) => return Err(syn::Error::new(u, format!("Union is not supported"))), 
    }?;

    Ok(quote! {
        impl std::fmt::Display for #ident{
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                #body
            }
        }
    })
}

/// Type alias for an expansion context:
/// - [`FmtAttribute`].
/// - Struct/enum/union [`syn::Ident`].
/// - Derived trait [`syn::Ident`].
/// - Attribute name [`syn::Ident`].
///
/// [`syn::Ident`]: struct@syn::Ident
type ExpansionCtx<'a> = (
    &'a FmtAttribute,
    &'a syn::Ident,
    &'a syn::Ident,
    &'a syn::Ident,
);

/// Expands a [`fmt::Display`]-like derive macro for the provided struct.
fn expand_struct(
    s: &syn::DataStruct,
    (attrs, ident, trait_ident, _): ExpansionCtx<'_>,
) -> syn::Result<(Vec<syn::WherePredicate>, TokenStream)> {
    let s = Expansion {
        attrs,
        fields: &s.fields,
        trait_ident,
        ident,
    };
    let body = s.generate_body()?;

    let vars = s.fields.iter().enumerate().map(|(i, f)| {
        let var = f.ident.clone().unwrap_or_else(|| format_ident!("_{i}"));
        let member = f
            .ident
            .clone()
            .map_or_else(|| syn::Member::Unnamed(i.into()), syn::Member::Named);
        quote! {
            let #var = &self.#member;
        }
    });

    let body = quote! {
        #( #vars )*
        #body
    };

    Ok(body)
}

/// Expands a [`fmt`]-like derive macro for the provided enum.
fn expand_enum(
    e: &syn::DataEnum,
    (attrs, _, trait_ident, attr_name): ExpansionCtx<'_>,
) -> syn::Result<(Vec<syn::WherePredicate>, TokenStream)> {
    if attrs.fmt.is_some() {
        todo!("https://github.com/JelteF/derive_more/issues/142");
    }

    let match_arms = e.variants.iter().try_fold(
        (Vec::new(), TokenStream::new()),
        |mut arms, variant| {
            let attrs = ContainerAttributes::parse_attrs(&variant.attrs, attr_name)?
                .map(Spanning::into_inner)
                .unwrap_or_default();
            let ident = &variant.ident;

            if attrs.fmt.is_none()
                && variant.fields.is_empty()
                && attr_name != "display"
            {
                return Err(syn::Error::new(
                    e.variants.span(),
                    format!(
                        "implicit formatting of unit enum variant is supported only for `Display` \
                         macro, use `#[{attr_name}(\"...\")]` to explicitly specify the formatting",
                    ),
                ));
            }

            let v = Expansion {
                attrs: &attrs,
                fields: &variant.fields,
                trait_ident,
                ident,
            };
            let arm_body = v.generate_body()?;

            let fields_idents =
                variant.fields.iter().enumerate().map(|(i, f)| {
                    f.ident.clone().unwrap_or_else(|| format_ident!("_{i}"))
                });
            let matcher = match variant.fields {
                syn::Fields::Named(_) => {
                    quote! { Self::#ident { #( #fields_idents ),* } }
                }
                syn::Fields::Unnamed(_) => {
                    quote! { Self::#ident ( #( #fields_idents ),* ) }
                }
                syn::Fields::Unit => quote! { Self::#ident },
            };

            arms.extend([quote! { #matcher => { #arm_body }, }]);

            Ok::<_, syn::Error>(arms)
        },
    )?;    

    let body = match_arms
        .is_empty()
        .then(|| quote! { match *self {} })
        .unwrap_or_else(|| quote! { match self { #match_arms } });

    Ok(body)
}


/// Helper struct to generate [`Display::fmt()`] implementation body and trait
/// bounds for a struct or an enum variant.
///
/// [`Display::fmt()`]: fmt::Display::fmt()
#[derive(Debug)]
struct Expansion<'a> {
    /// Derive macro [`FmtAttribute`].
    attrs: &'a FmtAttribute,

    /// Struct or enum [`syn::Ident`].
    ///
    /// [`syn::Ident`]: struct@syn::Ident
    ident: &'a syn::Ident,

    /// Struct or enum [`syn::Fields`].
    fields: &'a syn::Fields,

    /// [`fmt`] trait [`syn::Ident`].
    ///
    /// [`syn::Ident`]: struct@syn::Ident
    trait_ident: &'a syn::Ident,
}

impl<'a> Expansion<'a> {
    /// Generates [`Display::fmt()`] implementation for a struct or an enum variant.
    ///
    /// # Errors
    ///
    /// In case [`FmtAttribute`] is [`None`] and [`syn::Fields`] length is
    /// greater than 1.
    ///
    /// [`Display::fmt()`]: fmt::Display::fmt()
    /// [`FmtAttribute`]: super::FmtAttribute
    fn generate_body(&self) -> syn::Result<TokenStream> {
        match &self.attrs.fmt {
            Some(fmt) => {
                Ok(if let Some((expr, trait_ident)) = fmt.transparent_call() {
                    quote! { core::fmt::#trait_ident::fmt(&(#expr), __derive_more_f) }
                } else {
                    quote! { core::write!(__derive_more_f, #fmt) }
                })
            }
            None if self.fields.is_empty() => {
                let ident_str = self.ident.to_string();

                Ok(quote! {
                    core::write!(__derive_more_f, #ident_str)
                })
            }
            None if self.fields.len() == 1 => {
                let field = self
                    .fields
                    .iter()
                    .next()
                    .unwrap_or_else(|| unreachable!("count() == 1"));
                let ident = field.ident.clone().unwrap_or_else(|| format_ident!("_0"));
                let trait_ident = self.trait_ident;

                Ok(quote! {
                    core::fmt::#trait_ident::fmt(#ident, __derive_more_f)
                })
            }
            _ => Err(syn::Error::new(
                self.fields.span(),
                format!(
                    "TODO ARTHUR! struct or enum variant with more than 1 field must have \
                     `#[{}(\"...\", ...)]` attribute",
                    trait_name_to_attribute_name(self.trait_ident),
                ),
            )),
        }
    }
}

