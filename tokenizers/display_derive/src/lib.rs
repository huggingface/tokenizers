extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{format_ident,quote};
use syn::{parse_macro_input, DeriveInput, Meta, MetaNameValue, Lit}; 
mod vendored;
mod parsing;
use vendored::FmtAttribute;

#[proc_macro_derive(Display, attributes(display))]
pub fn display_derive(input: TokenStream) -> TokenStream  {
    // Parse the parsed_input tokens into a syntax tree
    let parsed_input = parse_macro_input!(input as DeriveInput);
    // let attrs =  syn::parse::<FmtAttribute>(input).unwrap();
    let mut fmt = quote!{};
    for attr in parsed_input.attrs{
        if attr.path.is_ident("display"){
            println!("attrs: {:?}", attr.path.get_ident()); 
            fmt = quote!{ write!(f, "display(fmt = '', ...) is not supported yet!")};
        }
    }
    
    // 1. If the attrs are not None, then we defer to this. 
    // Meaning we juste return quote!{ format!(#fmt, #attr)} 
    let trait_ident = format_ident!("display");
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
    let expanded = quote! {
        impl std::fmt::Display for #ident {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                #body
            }
        }
    };

    TokenStream::from(expanded)
}

fn generate_fmt_impl_for_struct(data_struct: &syn::DataStruct, ident: &syn::Ident) -> proc_macro2::TokenStream {

    // return quote!{
    //    write!(f, "automatic print")
    // };
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

fn generate_fmt_impl_for_enum(data_enum: &syn::DataEnum, ident: &syn::Ident) -> proc_macro2::TokenStream {
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
                        write!(f, "{}: {}", stringify!(#field_name), self.#field_name)?;
                    }
                });
                quote! {
                    write!(f, "{} {{ ", stringify!(#variant_name))?;
                    #( #field_fmts )*
                    write!(f, " }}")?;
                }
            }
            syn::Fields::Unnamed(_) => quote! { write!(f, "__UNAMED__")} 
        };
        quote! { #variant_name => {#variant_fmt} }
    });
    quote! {
                match *self {
                    #(#arms),*
                }
            }
}
