extern crate tokenizers as tk;

use crate::models::*;
use neon::prelude::*;

/// Tokenizer
pub struct Tokenizer {
    tokenizer: tk::tokenizer::Tokenizer,
}

declare_types! {
    pub class JsTokenizer for Tokenizer {
        // Create
        init(mut cx) {
            let mut model = cx.argument::<JsModel>(0)?;
            if let Some(instance) = {
                let guard = cx.lock();
                let mut model = model.borrow_mut(&guard);
                model.model.to_pointer()
            } {
                Ok(Tokenizer {
                    tokenizer: tk::tokenizer::Tokenizer::new(instance)
                })
            } else {
                cx.throw_error("The Model is already being used in another Tokenizer")
            }
        }

        method with_model(mut cx) {
            let mut model = cx.argument::<JsModel>(0)?;
            if let Some(instance) = {
                let guard = cx.lock();
                let mut model = model.borrow_mut(&guard);
                model.model.to_pointer()
            } {
                let mut this = cx.this();
                {
                    let guard = cx.lock();
                    let mut tokenizer = this.borrow_mut(&guard);
                    tokenizer.tokenizer.with_model(instance);
                }

                Ok(cx.undefined().upcast())
            } else {
                cx.throw_error("The Model is already being used in another Tokenizer")
            }
        }
    }
}

pub fn register(m: &mut ModuleContext, prefix: &str) -> Result<(), neon::result::Throw> {
    m.export_class::<JsTokenizer>(&format!("{}_Tokenizer", prefix))?;
    Ok(())
}
