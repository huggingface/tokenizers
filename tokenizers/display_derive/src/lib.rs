extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

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
                    let field_types = fields.named.iter().map(|f| &f.ty);
                    quote! {
                        impl std::fmt::Display for #name {
                            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                                write!(f, "{}(", stringify!(#name))?;
                                let mut first = true;
                                #(
                                    if !first {
                                        write!(f, ", ")?;
                                    }
                                    first = false;

                                    let field_value = &self.#field_names2;
                                    write!(f, "{}=", stringify!(#field_names))?;
                                    if std::any::TypeId::of::<#field_types>() == std::any::TypeId::of::<String>(){
                                        write!(f, "\"{}\"", field_value)?;
                                    } else {
                                        let s = format!("{}", field_value);
                                        let mut chars = s.chars();
                                        let mut prefix = (&mut chars).take(100 - 1).collect::<String>();
                                        if chars.next().is_some() {
                                            prefix.push('…');
                                        }
                                        write!(f, "{}", prefix)?;
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
                                write!(f, "{}()", stringify!(#name))
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
