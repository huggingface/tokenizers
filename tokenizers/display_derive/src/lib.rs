extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{format_ident, quote, ToTokens};
use syn::{parse_macro_input, DeriveInput, Lit, Meta, MetaNameValue};
mod parsing;
mod vendored;
use vendored::FmtAttribute;

#[proc_macro_derive(Display, attributes(display))]
pub fn display_derive(input: TokenStream) -> TokenStream {
    // Parse the parsed_input tokens into a syntax tree
    // let attrs = syn::parse::<FmtAttribute>(input.clone());
    // // Handle the Result from the parsing step
    // let attrs = match attrs {
    //     Ok(attrs) => attrs,
    //     Err(_) => return TokenStream::new(), // Handle error case appropriately
    // };
    let parsed_input = parse_macro_input!(input as DeriveInput);
    let mut fmt = quote! {};
    for attr in parsed_input.attrs {
        if attr.path.is_ident("display") {
            fmt = quote! { write!(f, "display(fmt = '', ...)")};
        }
    }

    // 1. If the attrs are not None, then we defer to this.
    // Meaning we juste return quote!{ format!(#fmt, #attr)}
    let trait_ident: syn::Ident = format_ident!("display");
    let ident = &parsed_input.ident;

    let body = if fmt.is_empty() {
        // 2. We automatically parse
        match &parsed_input.data {
            syn::Data::Struct(s) => generate_fmt_impl_for_struct(s, ident),
            syn::Data::Enum(e) => generate_fmt_impl_for_enum(e, ident),
            syn::Data::Union(u) => {
                let error = syn::Error::new_spanned(u.union_token, "Unions are not supported");
                return proc_macro::TokenStream::from(error.into_compile_error());
            }
        }
    } else {
        fmt
    };

    println!("body: {:?}", body.to_string());
    let expanded = quote! {
        impl std::fmt::Display for #ident {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                #body
            }
        }
    };

    TokenStream::from(expanded)
}

fn generate_fmt_impl_for_struct(
    data_struct: &syn::DataStruct,
    ident: &syn::Ident,
) -> proc_macro2::TokenStream {
    let fields = &data_struct.fields;

    // Extract field names and types
    let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
    let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();

    quote! {
        write!(f, "{}(", stringify!(#ident))?;
        let mut first = true;
        #(
            if !first {
                write!(f, ", ")?;
            }
            first = false;

            let field_value = &self.#field_names;
            write!(f, "{}=", stringify!(#field_names))?;
            if std::any::TypeId::of::<#field_types>() == std::any::TypeId::of::<String>() {
                println!("We have a string!");
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
