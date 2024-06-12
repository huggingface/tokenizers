extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{format_ident,quote};
use syn::{parse_macro_input, DeriveInput}; 
mod vendored;
mod parsing;
use vendored::FmtAttribute;

#[proc_macro_derive(Display)]
pub fn display_derive(input: TokenStream) -> TokenStream  {
    // Parse the parsed_input tokens into a syntax tree
    let parsed_input = parse_macro_input!(input as DeriveInput);
    let attrs =  syn::parse::<FmtAttribute>(input).unwrap();
    // 1. If the attrs are not None, then we defer to this. 
    // Meaning we juste return quote!{ format!(#fmt, #attr)} 
    let trait_ident = format_ident!("display");
    let ident = &parsed_input.ident;

    // 2. We automatically parse
    let body = match &parsed_input.data {
        syn::Data::Struct(s) => generate_fmt_impl_for_struct(s, ident),
        syn::Data::Enum(e) => generate_fmt_impl_for_enum(e, ident),
        syn::Data::Union(u) => {
            let error = syn::Error::new_spanned(u.union_token, "Unions are not supported");
            return proc_macro::TokenStream::from(error.into_compile_error());
        }
    };

    let expanded = quote! {
        impl std::fmt::Display for #ident {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                #body
            }
        }
    };

    TokenStream::from(expanded)
}

fn generate_fmt_impl_for_struct(data_struct: &syn::DataStruct, ident: &syn::Ident) -> TokenStream {
    let fields = &data_struct.fields;
    let field_fmts = fields.iter().enumerate().map(|(i, field)| {
        let field_name = match &field.ident {
            Some(ident) => ident,
            None => {
                // If the field doesn't have a name, we generate a name based on its index
                let index = syn::Index::from(i);
                quote! { #index }
            }
        };
        quote! {
            write!(f, "{}: {}", stringify!(#field_name), self.#field_name)?;
        }
    });
    // Collect the mapped tokens into a TokenStream
    field_fmts
}

fn generate_fmt_impl_for_enum(data_enum: &syn::DataEnum, ident: &syn::Ident) -> TokenStream {
    let arms = data_enum.variants.iter().map(|variant| {
        let variant_name = &variant.ident;
        let variant_fmt = match &variant.fields {
            syn::Fields::Unit => {
                // If the variant has no fields, we just print its name
                quote! { write!(f, "{}", stringify!(#variant_name))?; }
            }
            syn::Fields::Named(fields) => {
                // If the variant has named fields, we print each field's name and value
                let field_fmts = fields.named.iter().map(|field| {
                    let field_name = field.ident.as_ref().unwrap();
                    quote! {
                        write!(f, "{}: {:?}", stringify!(#field_name), self.#field_name)?;
                    }
                });
                quote! {
                    write!(f, "{} {{ ", stringify!(#variant_name))?;
                    #( #field_fmts )*
                    write!(f, " }}")?;
                }
            }
            syn::Fields::Unnamed(fields) => {
                // If the variant has unnamed fields, we print each field's value without names
                let field_fmts = fields.unnamed.iter().map(|field| {
                    quote! {
                        write!(f, "{:?}, ", self.#field)?;
                    }
                });
                quote! {
                    write!(f, "{}(", stringify!(#variant_name))?;
                    #( #field_fmts )*
                    write!(f, ")")?;
                }
            }
        };
        quote! { #ident::#variant_name => { #variant_fmt } }
    });
    arms
}
