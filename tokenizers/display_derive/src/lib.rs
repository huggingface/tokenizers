extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

#[proc_macro_derive(Display)]
pub fn display_derive(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    // Get the name of the struct
    let name = &input.ident;

    // Generate code to match the struct's fields
    let expanded = match input.data {
        Data::Struct(data) => {
            match data.fields {
                Fields::Named(fields) => {
                    // If the struct has named fields
                    let field_names = fields.named.iter().map(|f| &f.ident);
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

                                    let field_value = &self.#field_names;
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
                Fields::Unnamed(_) => {
                    quote! {
                        compile_error!("Unnamed fields for struct are not supported.");
                    }
                },
            }
        },
        Data::Enum(ref data_enum) => {
            let variants = &data_enum.variants;
            let display_impls = variants.iter().map(|variant| {
                let ident = &variant.ident;
                if let Some((_, meta)) = variant.attrs.iter().find(|(path, _)| path.is_ident("display")) {
                    if let Ok(Meta::List(MetaList { nested, .. })) = meta.parse_meta() {
                        let format_args = nested.iter().map(|nested_meta| {
                            if let NestedMeta::Lit(Lit::Str(s)) = nested_meta {
                                quote! { #s }
                            } else {
                                quote! { compile_error!("Invalid format argument"); }
                            }
                        });
                        quote! {
                            impl std::fmt::Display for #name {
                                fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                                    match self {
                                        Self::#ident(#format_args) => write!(f, "{}", stringify!(#ident)),
                                        _ => unreachable!(),
                                    }
                                }
                            }
                        }
                    } else {
                        quote! {
                            compile_error!("Invalid display attribute format");
                        }
                    }
                } else {
                    quote! {
                        impl std::fmt::Display for #name {
                            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                                write!(f, "{}", stringify!(#ident))
                            }
                        }
                    }
                }
            });
            quote! {
                #(#display_impls)*
            }
        },
        Data::Union(_) => {
            quote! {
                compile_error!("Unions are not supported for Display derive");
        
            }
        };
    }
    TokenStream::from(expanded)
}
