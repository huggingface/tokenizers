extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, stringify_punct, DeriveInput};

mod fmt_parsing;
use fmt_parsing::{find_display_attribute, FmtAttribute};

#[proc_macro_derive(Display, attributes(display))]
pub fn display_derive(input: TokenStream) -> TokenStream {
    // Parse the parsed_input tokens into a syntax tree
    let parsed_input = parse_macro_input!(input as DeriveInput);
    // Find the `display` attribute
    let attr = find_display_attribute(&parsed_input.attrs);
    // 1. If the attrs are not None, then we defer to this.
    // Meaning we juste return quote!{ format!(#fmt, #attr)}
    let ident = &parsed_input.ident;

    let body = {
        // 2. We automatically parse
        match &parsed_input.data {
            syn::Data::Struct(s) => generate_fmt_impl_for_struct(s, ident, &attr),
            syn::Data::Enum(e) => generate_fmt_impl_for_enum(e, ident),
            syn::Data::Union(u) => {
                let error = syn::Error::new_spanned(u.union_token, "Unions are not supported");
                return proc_macro::TokenStream::from(error.into_compile_error());
            }
        }
    };

    let expanded = quote! {
        impl std::fmt::Display for #ident {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                #body
            }
        }
    };

    println!("Generated body: \n{}\n", expanded);
    TokenStream::from(expanded)
}

fn generate_fmt_impl_for_struct(
    data_struct: &syn::DataStruct,
    ident: &syn::Ident,
    attrs: &Option<FmtAttribute>,
) -> proc_macro2::TokenStream {
    let fields = &data_struct.fields;
    // TODO I am stuck here for now hehe.
    // Basically we need to produce the body that will be used.
    // Generate field formatting expressions
    let field_formats: Vec<_> = fields
        .iter()
        .map(|f| {
            let field_name = &f.ident;
            let fmts = find_display_attribute(&f.attrs);

            if let Some(attr) = attrs {
                if attr.args.is_empty() {
                    // If there is a prefix and no args, use fmts if it exists
                    if let Some(fmt) = fmts {
                        // Combine prefix and fmts
                        quote! {
                            write!(f, "{}{}", #fmt.lit.value(), #fmt.args.to_string())?;
                        }
                    } else {
                        // If no fmts, write just the field value
                        quote! {
                            write!(f, "{}", self.#field_name)?;
                        }
                    }
                } else {
                    // If there are args to the attribute, use attr.lit and attr.args exclusively
                    quote! {
                        write!(f, "{}{}", #attr.lit.value(), #attr.args.to_string())?;
                    }
                }
            } else {
                // If there is no attribute, print everything directly
                quote! {
                    write!(f, "{}", self.#field_name)?;
                }
            }
        })
        .collect();

    // Generate the final implementation of Display trait for the struct
    quote! {
        write!(f, "{}(", stringify!(#ident))?;
        #field_formats
        write!(f, ")")
    }
}
fn generate_fmt_impl_for_enum(
    data_enum: &syn::DataEnum,
    ident: &syn::Ident,
) -> proc_macro2::TokenStream {
    let arms = data_enum.variants.iter().map(|variant| {
        let variant_name = &variant.ident;
        let formatted_output = match &variant.fields {
            syn::Fields::Unit => {
                // Unit variant: just stringify the variant name
                quote! { #ident::#variant_name => {write!(f, "{}", stringify!(#variant_name))?; }}
            },
            syn::Fields::Unnamed(fields) if fields.unnamed.len() == 1 => {
                // Tuple variant with one field
                quote! { #ident::#variant_name(ref single) => {write!(f, "{}", single)?;} }
            },
            syn::Fields::Named(fields) if fields.named.len() == 1 => {
                // Tuple variant with one named field
                let field_name = fields.named[0].ident.as_ref().unwrap(); // Assuming it's named
                quote! { #ident::#variant_name{..}=>{ write!(f, "{}({})", stringify!(self.#field_name)?);} }
            },
            _ => {
                // Default case: stringify the variant name
                quote! { write!(f, "{}", stringify!(#variant_name))?; }
            }
        };
       formatted_output
    });

    println!("printing ident: {}", ident.to_string());
    quote! {
        match *self {
            #(#arms)*
        }
        Ok(())
    }
}
