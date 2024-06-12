use crate::parsing;
use proc_macro2::TokenStream;
use quote::{format_ident, ToTokens};
use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token, 
    Expr,
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
    lit: syn::LitStr,

    /// Optional [`token::Comma`].
    ///
    /// [`token::Comma`]: struct@token::Comma
    comma: Option<token::Comma>,

    /// Interpolation arguments.
    args: Punctuated<FmtArgument, token::Comma>,
}

impl Parse for FmtAttribute {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {

        Ok(Self {
            lit: input.parse()?,
            comma: input
                .peek(token::Comma)
                .then(|| input.parse())
                .transpose()?,
            args: input.parse_terminated(FmtArgument::parse)?,
        })
    }
}


impl ToTokens for FmtAttribute {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.lit.to_tokens(tokens);
        self.comma.to_tokens(tokens);
        self.args.to_tokens(tokens);
    }
}

impl FmtAttribute {
    /// Checks whether this [`FmtAttribute`] can be replaced with a transparent delegation (calling
    /// a formatting trait directly instead of interpolation syntax).
    ///
    /// If such transparent call is possible, the returns an [`Ident`] of the delegated trait and
    /// the [`Expr`] to pass into the call, otherwise [`None`].
    ///
    /// [`Ident`]: struct@syn::Ident
    fn transparent_call(&self) -> Option<(Expr, syn::Ident)> {
        // `FmtAttribute` is transparent when:

        // (1) There is exactly one formatting parameter.
        let lit = self.lit.value();
        let param = parsing::format(&lit).and_then(|(more, p)| more.is_empty().then_some(p))?;

        // (2) And the formatting parameter doesn't contain any modifiers.
        if param
            .spec
            .map(|s| {
                s.align.is_some()
                    || s.sign.is_some()
                    || s.alternate.is_some()
                    || s.zero_padding.is_some()
                    || s.width.is_some()
                    || s.precision.is_some()
                    || !s.ty.is_trivial()
            })
            .unwrap_or_default()
        {
            return None;
        }

        let expr = match param.arg {
            // (3) And either exactly one positional argument is specified.
            Some(parsing::Argument::Integer(_)) | None => (self.args.len() == 1)
                .then(|| self.args.first())
                .flatten()
                .map(|a| a.expr.clone()),

            // (4) Or the formatting parameter's name refers to some outer binding.
            Some(parsing::Argument::Identifier(name)) if self.args.is_empty() => {
                Some(format_ident!("{name}").into())
            }

            // (5) Or exactly one named argument is specified for the formatting parameter's name.
            Some(parsing::Argument::Identifier(name)) => (self.args.len() == 1)
                .then(|| self.args.first())
                .flatten()
                .filter(|a| a.alias.as_ref().map(|a| a.0 == name).unwrap_or_default())
                .map(|a| a.expr.clone()),
        }?;

        let trait_name = param
            .spec
            .map(|s| s.ty)
            .unwrap_or(parsing::Type::Display)
            .trait_name();

        Some((expr, format_ident!("{trait_name}")))
    }
}

/// Representation of a [named parameter][1] (`identifier '=' expression`) in
/// in a [`FmtAttribute`].
///
/// [1]: https://doc.rust-lang.org/stable/std/fmt/index.html#named-parameters
struct FmtArgument {
    /// `identifier =` [`Ident`].
    ///
    /// [`Ident`]: struct@syn::Ident
    alias: Option<(syn::Ident, token::Eq)>,

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
            ident.to_tokens(tokens);
            eq.to_tokens(tokens);
        }
        self.expr.to_tokens(tokens);
    }
}
