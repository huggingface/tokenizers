use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyList, PyString};
use pyo3_introspection::model::{Argument, Arguments, Class, Constant, Expr, Function, Module};

// Some constructors take `**kwargs` and parse them by hand, so introspection only sees
// `**kwargs`. Their `text_signature` lists the accepted names: expose those as keyword-only
// parameters, typed like the property setter of the same name.
pub fn apply(module: &mut Module, runtime: &Bound<'_, PyAny>) -> PyResult<()> {
    for class in &mut module.classes {
        if !has_kwargs_only_constructor(class) {
            continue;
        }
        let params = text_signature_params(&runtime.getattr(class.name.as_str())?)?;
        if let Some(arguments) = params.and_then(|params| keyword_only_constructor(class, params)) {
            constructor_mut(class).unwrap().arguments = arguments;
        }
    }
    for submodule in &mut module.modules {
        apply(submodule, &runtime.getattr(submodule.name.as_str())?)?;
    }
    Ok(())
}

fn has_kwargs_only_constructor(class: &Class) -> bool {
    class.methods.iter().find(|m| m.name == "__new__").is_some_and(|new| {
        let args = &new.arguments;
        args.kwarg.is_some()
            && args.positional_only_arguments.iter().all(|a| a.name == "cls")
            && args.arguments.is_empty()
            && args.vararg.is_none()
            && args.keyword_only_arguments.is_empty()
    })
}

pub fn keyword_only_constructor(class: &Class, params: Vec<(String, Expr)>) -> Option<Arguments> {
    if !has_kwargs_only_constructor(class) {
        return None;
    }
    let receiver = class.methods.iter().find(|m| m.name == "__new__")?.arguments.positional_only_arguments.clone();
    let keyword_only_arguments = params
        .into_iter()
        .map(|(name, default)| Argument {
            annotation: setter_annotation(class, &name),
            name,
            default_value: Some(default),
        })
        .collect();
    Some(Arguments {
        positional_only_arguments: receiver,
        arguments: vec![],
        vararg: None,
        keyword_only_arguments,
        kwarg: None,
    })
}

fn constructor_mut(class: &mut Class) -> Option<&mut Function> {
    class.methods.iter_mut().find(|m| m.name == "__new__")
}

fn setter_annotation(class: &Class, property: &str) -> Option<Expr> {
    let is_setter = |decorator: &Expr| {
        matches!(decorator, Expr::Attribute { value, attr }
            if attr == "setter" && matches!(value.as_ref(), Expr::Attribute { attr, .. } if attr == property))
    };
    class
        .methods
        .iter()
        .find(|m| m.name == property && m.decorators.iter().any(is_setter))
        .and_then(|setter| {
            let args = &setter.arguments;
            args.positional_only_arguments.iter().chain(&args.arguments).last()?.annotation.clone()
        })
}

fn text_signature_params(class: &Bound<'_, PyAny>) -> PyResult<Option<Vec<(String, Expr)>>> {
    if class.getattr("__text_signature__")?.is_none() {
        return Ok(None);
    }
    let inspect = class.py().import("inspect")?;
    let empty = inspect.getattr("Parameter")?.getattr("empty")?;
    let signature = inspect.call_method1("signature", (class,))?;
    let mut params = vec![];
    for param in signature.getattr("parameters")?.call_method0("values")?.try_iter()? {
        let param = param?;
        let name: String = param.getattr("name")?.extract()?;
        let default = param.getattr("default")?;
        if name == "self" || default.is(&empty) {
            continue;
        }
        params.push((name, default_expr(&default)?));
    }
    Ok(Some(params))
}

fn default_expr(value: &Bound<'_, PyAny>) -> PyResult<Expr> {
    let constant = if value.is_none() {
        Constant::None
    } else if let Ok(b) = value.cast::<PyBool>() {
        Constant::Bool(b.is_true())
    } else if value.is_instance_of::<PyInt>() {
        Constant::Int(value.str()?.to_string())
    } else if value.is_instance_of::<PyFloat>() {
        Constant::Float(value.repr()?.to_string())
    } else if let Ok(s) = value.cast::<PyString>() {
        Constant::Str(s.to_string())
    } else if value.cast::<PyList>().is_ok_and(|l| l.is_empty()) {
        return Ok(Expr::List { elts: vec![] });
    } else {
        Constant::Ellipsis
    };
    Ok(Expr::Constant { value: constant })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3_introspection::model::{Constant, VariableLengthArgument};

    fn name(id: &str) -> Expr {
        Expr::Name { id: id.into() }
    }

    fn int(value: &str) -> Expr {
        Expr::Constant { value: Constant::Int(value.into()) }
    }

    fn receiver(name: &str) -> Argument {
        Argument { name: name.into(), default_value: None, annotation: None }
    }

    fn arguments(receiver_name: &str, positional: Vec<Argument>, kwarg: Option<&str>) -> Arguments {
        Arguments {
            positional_only_arguments: vec![receiver(receiver_name)],
            arguments: positional,
            vararg: None,
            keyword_only_arguments: vec![],
            kwarg: kwarg.map(|name| VariableLengthArgument { name: name.into(), annotation: None }),
        }
    }

    fn method(name: &str, decorators: Vec<Expr>, arguments: Arguments) -> Function {
        Function { name: name.into(), decorators, arguments, returns: None, is_async: false, docstring: None }
    }

    fn setter(property: &str, annotation: Expr) -> Function {
        method(
            property,
            vec![Expr::Attribute {
                value: Box::new(Expr::Attribute {
                    value: Box::new(Expr::Attribute { value: Box::new(name("tokenizers.trainers")), attr: "Trainer".into() }),
                    attr: property.into(),
                }),
                attr: "setter".into(),
            }],
            arguments("self", vec![Argument { name: "value".into(), default_value: None, annotation: Some(annotation) }], None),
        )
    }

    fn class(methods: Vec<Function>) -> Class {
        Class {
            name: "Trainer".into(),
            bases: vec![],
            methods,
            attributes: vec![],
            decorators: vec![],
            inner_classes: vec![],
            docstring: None,
        }
    }

    #[test]
    fn kwargs_constructor_takes_the_text_signature_as_keyword_only_typed_by_setters() {
        let class = class(vec![
            method("__new__", vec![], arguments("cls", vec![], Some("kwargs"))),
            setter("vocab_size", name("int")),
        ]);

        let new = keyword_only_constructor(&class, vec![("vocab_size".into(), int("30000")), ("words".into(), int("0"))]).unwrap();

        assert_eq!(
            new,
            Arguments {
                positional_only_arguments: vec![receiver("cls")],
                arguments: vec![],
                vararg: None,
                keyword_only_arguments: vec![
                    Argument { name: "vocab_size".into(), default_value: Some(int("30000")), annotation: Some(name("int")) },
                    Argument { name: "words".into(), default_value: Some(int("0")), annotation: None },
                ],
                kwarg: None,
            }
        );
    }

    #[test]
    fn constructor_with_declared_parameters_is_left_alone() {
        let declared = Argument { name: "num_merges".into(), default_value: Some(int("32000")), annotation: Some(name("int")) };
        let class = class(vec![method("__new__", vec![], arguments("cls", vec![declared], None))]);

        assert_eq!(keyword_only_constructor(&class, vec![("num_merges".into(), int("1"))]), None);
    }

    #[test]
    fn class_without_constructor_is_left_alone() {
        assert_eq!(keyword_only_constructor(&class(vec![]), vec![("vocab_size".into(), int("1"))]), None);
    }
}
