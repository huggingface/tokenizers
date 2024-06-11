
/// Representation of a [`fmt`]-like attribute.
///
/// ```rust,ignore
/// #[<attribute>("<fmt-literal>", <fmt-args>)]
/// ```
///
/// [`fmt`]: std::fmt
#[derive(Debug)]
struct FmtAttribute {
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
        Self::check_legacy_fmt(input)?;

        Ok(Self {
            lit: input.parse()?,
            comma: input
                .peek(token::Comma)
                .then(|| input.parse())
                .transpose()?,
            args: input.parse_terminated(FmtArgument::parse, token::Comma)?,
        })
    }
}

impl attr::ParseMultiple for FmtAttribute {}

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
        let param =
            parsing::format(&lit).and_then(|(more, p)| more.is_empty().then_some(p))?;

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

    /// Returns an [`Iterator`] over bounded [`syn::Type`]s (and correspondent trait names) by this
    /// [`FmtAttribute`].
    fn bounded_types<'a>(
        &'a self,
        fields: &'a syn::Fields,
    ) -> impl Iterator<Item = (&'a syn::Type, &'static str)> {
        let placeholders = Placeholder::parse_fmt_string(&self.lit.value());

        // We ignore unknown fields, as compiler will produce better error messages.
        placeholders.into_iter().filter_map(move |placeholder| {
            let name = match placeholder.arg {
                Parameter::Named(name) => self
                    .args
                    .iter()
                    .find_map(|a| (a.alias()? == &name).then_some(&a.expr))
                    .map_or(Some(name), |expr| expr.ident().map(ToString::to_string))?,
                Parameter::Positional(i) => self
                    .args
                    .iter()
                    .nth(i)
                    .and_then(|a| a.expr.ident().filter(|_| a.alias.is_none()))?
                    .to_string(),
            };

            let unnamed = name.strip_prefix('_').and_then(|s| s.parse().ok());
            let ty = match (&fields, unnamed) {
                (syn::Fields::Unnamed(f), Some(i)) => {
                    f.unnamed.iter().nth(i).map(|f| &f.ty)
                }
                (syn::Fields::Named(f), None) => f.named.iter().find_map(|f| {
                    f.ident.as_ref().filter(|s| **s == name).map(|_| &f.ty)
                }),
                _ => None,
            }?;

            Some((ty, placeholder.trait_name))
        })
    }

    /// Errors in case legacy syntax is encountered: `fmt = "...", (arg),*`.
    fn check_legacy_fmt(input: ParseStream<'_>) -> syn::Result<()> {
        let fork = input.fork();

        let path = fork
            .parse::<syn::Path>()
            .and_then(|path| fork.parse::<token::Eq>().map(|_| path));
        match path {
            Ok(path) if path.is_ident("fmt") => (|| {
                let args = fork
                    .parse_terminated(
                        <Either<syn::Lit, syn::Ident>>::parse,
                        token::Comma,
                    )
                    .ok()?
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, arg)| match arg {
                        Either::Left(syn::Lit::Str(str)) => Some(if i == 0 {
                            format!("\"{}\"", str.value())
                        } else {
                            str.value()
                        }),
                        Either::Right(ident) => Some(ident.to_string()),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                (!args.is_empty()).then_some(args)
            })()
            .map_or(Ok(()), |fmt| {
                Err(syn::Error::new(
                    input.span(),
                    format!(
                        "legacy syntax, remove `fmt =` and use `{}` instead",
                        fmt.join(", "),
                    ),
                ))
            }),
            Ok(_) | Err(_) => Ok(()),
        }
    }
}

/// Representation of a [named parameter][1] (`identifier '=' expression`) in
/// in a [`FmtAttribute`].
///
/// [1]: https://doc.rust-lang.org/stable/std/fmt/index.html#named-parameters
#[derive(Debug)]
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

/// Representation of a [parameter][1] used in a [`Placeholder`].
///
/// [1]: https://doc.rust-lang.org/stable/std/fmt/index.html#formatting-parameters
#[derive(Debug, Eq, PartialEq)]
enum Parameter {
    /// [Positional parameter][1].
    ///
    /// [1]: https://doc.rust-lang.org/stable/std/fmt/index.html#positional-parameters
    Positional(usize),

    /// [Named parameter][1].
    ///
    /// [1]: https://doc.rust-lang.org/stable/std/fmt/index.html#named-parameters
    Named(String),
}

impl<'a> From<parsing::Argument<'a>> for Parameter {
    fn from(arg: parsing::Argument<'a>) -> Self {
        match arg {
            parsing::Argument::Integer(i) => Self::Positional(i),
            parsing::Argument::Identifier(i) => Self::Named(i.to_owned()),
        }
    }
}

/// Representation of a formatting placeholder.
#[derive(Debug, Eq, PartialEq)]
struct Placeholder {
    /// Formatting argument (either named or positional) to be used by this placeholder.
    arg: Parameter,

    /// [Width parameter][1], if present.
    ///
    /// [1]: https://doc.rust-lang.org/stable/std/fmt/index.html#width
    width: Option<Parameter>,

    /// [Precision parameter][1], if present.
    ///
    /// [1]: https://doc.rust-lang.org/stable/std/fmt/index.html#precision
    precision: Option<Parameter>,

    /// Name of [`std::fmt`] trait to be used for rendering this placeholder.
    trait_name: &'static str,
}

impl Placeholder {
    /// Parses [`Placeholder`]s from the provided formatting string.
    fn parse_fmt_string(s: &str) -> Vec<Self> {
        let mut n = 0;
        parsing::format_string(s)
            .into_iter()
            .flat_map(|f| f.formats)
            .map(|format| {
                let (maybe_arg, ty) = (
                    format.arg,
                    format.spec.map(|s| s.ty).unwrap_or(parsing::Type::Display),
                );
                let position = maybe_arg.map(Into::into).unwrap_or_else(|| {
                    // Assign "the next argument".
                    // https://doc.rust-lang.org/stable/std/fmt/index.html#positional-parameters
                    n += 1;
                    Parameter::Positional(n - 1)
                });

                Self {
                    arg: position,
                    width: format.spec.and_then(|s| match s.width {
                        Some(parsing::Count::Parameter(arg)) => Some(arg.into()),
                        _ => None,
                    }),
                    precision: format.spec.and_then(|s| match s.precision {
                        Some(parsing::Precision::Count(parsing::Count::Parameter(
                            arg,
                        ))) => Some(arg.into()),
                        _ => None,
                    }),
                    trait_name: ty.trait_name(),
                }
            })
            .collect()
    }
}

/// Representation of a [`fmt::Display`]-like derive macro attributes placed on a container (struct
/// or enum variant).
///
/// ```rust,ignore
/// #[<attribute>("<fmt-literal>", <fmt-args>)]
/// #[<attribute>(bound(<where-predicates>))]
/// ```
///
/// `#[<attribute>(...)]` can be specified only once, while multiple `#[<attribute>(bound(...))]`
/// are allowed.
///
/// [`fmt::Display`]: std::fmt::Display
#[derive(Debug, Default)]
struct ContainerAttributes {
    /// Interpolation [`FmtAttribute`].
    fmt: Option<FmtAttribute>,

    /// Addition trait bounds.
    bounds: BoundsAttribute,
}

impl Parse for ContainerAttributes {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        // We do check `FmtAttribute::check_legacy_fmt` eagerly here, because `Either` will swallow
        // any error of the `Either::Left` if the `Either::Right` succeeds.
        FmtAttribute::check_legacy_fmt(input)?;
        <Either<FmtAttribute, BoundsAttribute>>::parse(input).map(|v| match v {
            Either::Left(fmt) => Self {
                bounds: BoundsAttribute::default(),
                fmt: Some(fmt),
            },
            Either::Right(bounds) => Self { bounds, fmt: None },
        })
    }
}

impl attr::ParseMultiple for ContainerAttributes {
    fn merge_attrs(
        prev: Spanning<Self>,
        new: Spanning<Self>,
        name: &syn::Ident,
    ) -> syn::Result<Spanning<Self>> {
        let Spanning {
            span: prev_span,
            item: mut prev,
        } = prev;
        let Spanning {
            span: new_span,
            item: new,
        } = new;

        if new.fmt.and_then(|n| prev.fmt.replace(n)).is_some() {
            return Err(syn::Error::new(
                new_span,
                format!("multiple `#[{name}(\"...\", ...)]` attributes aren't allowed"),
            ));
        }
        prev.bounds.0.extend(new.bounds.0);

        Ok(Spanning::new(
            prev,
            prev_span.join(new_span).unwrap_or(prev_span),
        ))
    }
}

/// Matches the provided `trait_name` to appropriate [`FmtAttribute`]'s argument name.
fn trait_name_to_attribute_name<T>(trait_name: T) -> &'static str
where
    T: for<'a> PartialEq<&'a str>,
{
    match () {
        _ if trait_name == "Binary" => "binary",
        _ if trait_name == "Debug" => "debug",
        _ if trait_name == "Display" => "display",
        _ if trait_name == "LowerExp" => "lower_exp",
        _ if trait_name == "LowerHex" => "lower_hex",
        _ if trait_name == "Octal" => "octal",
        _ if trait_name == "Pointer" => "pointer",
        _ if trait_name == "UpperExp" => "upper_exp",
        _ if trait_name == "UpperHex" => "upper_hex",
        _ => unimplemented!(),
    }
}

