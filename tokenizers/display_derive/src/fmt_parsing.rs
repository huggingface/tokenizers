use proc_macro2::TokenStream;
use quote::quote;
use quote::ToTokens;
use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token, Attribute, Expr,
};

/// Representation of a [`fmt`]-like attribute.
///
/// ```rust,ignore
/// #[<attribute>("<fmt-literal>", <fmt-args>)]
/// ```
///
/// [`fmt`]: std::fmt
pub struct FmtAttribute {
    /// Interpolation [`syn::LitStr`].
    ///
    /// [`syn::LitStr`]: struct@syn::LitStr
    pub lit: syn::LitStr,

    /// Optional [`token::Comma`].
    ///
    /// [`token::Comma`]: struct@token::Comma
    comma: Option<token::Comma>,

    /// Interpolation arguments.
    pub args: Punctuated<FmtArgument, token::Comma>,
}

impl Parse for FmtAttribute {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        let _ident: syn::Ident = input
            .parse()
            .map_err(|_| syn::Error::new(input.span(), "Expected 'fmt' argument"))?;
        input
            .parse::<syn::Token![=]>()
            .map_err(|_| syn::Error::new(input.span(), "Expected '=' after 'fmt'"))?;

        let attribute = Self {
            lit: input.parse()?,
            comma: input
                .peek(token::Comma)
                .then(|| input.parse())
                .transpose()?,
            args: input.parse_terminated::<FmtArgument, token::Comma>(FmtArgument::parse)?,
        };
        println!(
            "Parsed successfully!, {:?}\n parsed arguments: {}",
            attribute.lit.token().to_string(),
            attribute.args.to_token_stream(),
        );
        Ok(attribute)
    }
}

impl ToTokens for FmtAttribute {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.lit.to_tokens(tokens);
        self.comma.to_tokens(tokens);
        self.args.to_tokens(tokens);
    }
}

/// Representation of a [named parameter][1] (`identifier '=' expression`) in
/// in a [`FmtAttribute`].
/// This should be used in `[display(fmt="", alias=alias, expr)]`.
/// [1]: https://doc.rust-lang.org/stable/std/fmt/index.html#named-parameters
pub struct FmtArgument {
    /// `identifier =` [`Ident`].
    ///
    /// [`Ident`]: struct@syn::Ident
    pub alias: Option<(syn::Ident, token::Eq)>,

    /// `expression` [`Expr`].
    expr: Expr,
}

impl FmtArgument {
    /// Returns an `identifier` of the [named parameter][1].
    ///
    /// [1]: https://doc.rust-lang.org/stable/std/fmt/index.html#named-parameters
    fn alias(&self) -> Option<&syn::Ident> {
        self.alias.as_ref().map(|(ident, _)| ident)
    }
}

impl Parse for FmtArgument {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(Self {
            alias: (input.peek(syn::Ident) && input.peek2(token::Eq))
                .then(|| Ok::<_, syn::Error>((input.parse()?, input.parse()?)))
                .transpose()?,
            expr: input.parse()?,
        })
    }
}

impl ToTokens for FmtArgument {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if let Some((ident, eq)) = &self.alias {
            quote!(self . #ident).to_tokens(tokens);
            eq.to_tokens(tokens);
        }
        self.expr.to_tokens(tokens)
    }
}

pub fn find_display_attribute(attrs: &[Attribute]) -> Option<FmtAttribute> {
    let display_attr = attrs.iter().find(|attr| attr.path.is_ident("display"));

    let attr: Option<FmtAttribute> = if let Some(attr) = display_attr {
        match attr.parse_args::<FmtAttribute>() {
            Ok(display_macro) => Some(display_macro),
            Err(e) => {
                e.to_compile_error();
                None
            }
        }
    } else {
        None
    };
    attr
}
