extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};
use utils::truncate_with_ellipsis;
#[proc_macro_derive(StructDisplay)]
pub fn display_derive(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    // Get the name of the struct
    let name = input.ident;

    // Generate code to match the struct's fields
    let expanded = match input.data {
        Data::Struct(data) => {
            match data.fields {
                Fields::Named(fields) => {
                    // If the struct has named fields
                    let field_names = fields.named.iter().map(|f| &f.ident);
                    let field_names2 = field_names.clone();
                    quote! {
                        impl std::fmt::Display for #name {
                            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                                write!(f, "{}(", stringify!(#name))?;
                                #(
                                    let value_str = self.#field_names2.to_string();
                                    let truncated_value_str =  truncate_with_ellipsis(value_str, 10);
                                    write!(f, "{}={}", stringify!(#field_names), truncated_value_str)?;
                                    if stringify!(#field_names) != stringify!(#field_names2.clone().last().unwrap()) {
                                        write!(f, ", ")?;
                                    }
                                )*
                                write!(f, ")")
                            }
                        }
                    }
                },
                Fields::Unit => {
                    // If the struct has no fields
                    quote! {
                        impl std::fmt::Display for #name {
                            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                                write!(f, "{}", stringify!(#name))
                            }
                        }
                    }
                },
                _ => unimplemented!(),
            }
        },
        _ => unimplemented!(),
    };

    // Convert into a token stream and return it
    TokenStream::from(expanded)
}