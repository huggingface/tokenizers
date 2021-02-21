extern crate tokenizers as tk;

use crate::models::*;
use neon::prelude::*;
use std::sync::{Arc, RwLock};
use tk::models::bpe::{BpeBuilder, BPE};
use tk::models::wordlevel::{WordLevel, WordLevelBuilder};
use tk::models::wordpiece::{WordPiece, WordPieceBuilder};

pub struct WordPieceFromFilesTask(Option<WordPieceBuilder>);
impl WordPieceFromFilesTask {
    pub fn new(builder: WordPieceBuilder) -> Self {
        Self(Some(builder))
    }
}

impl Task for WordPieceFromFilesTask {
    type Output = WordPiece;
    type Error = String;
    type JsEvent = JsValue;

    fn perform(&self) -> Result<Self::Output, Self::Error> {
        let builder: Option<WordPieceBuilder> =
            unsafe { std::ptr::replace(&self.0 as *const _ as *mut _, None) };
        builder.unwrap().build().map_err(|e| format!("{}", e))
    }

    fn complete(
        self,
        mut cx: TaskContext,
        result: Result<Self::Output, Self::Error>,
    ) -> JsResult<Self::JsEvent> {
        let wordpiece = result.map_err(|e| cx.throw_error::<_, ()>(e).unwrap_err())?;

        let mut js_model = JsModel::new::<_, JsModel, _>(&mut cx, vec![])?;
        let guard = cx.lock();
        js_model.borrow_mut(&guard).model = Some(Arc::new(RwLock::new(wordpiece.into())));

        Ok(js_model.upcast())
    }
}

pub struct WordLevelFromFilesTask(Option<WordLevelBuilder>);
impl WordLevelFromFilesTask {
    pub fn new(builder: WordLevelBuilder) -> Self {
        Self(Some(builder))
    }
}

impl Task for WordLevelFromFilesTask {
    type Output = WordLevel;
    type Error = String;
    type JsEvent = JsValue;

    fn perform(&self) -> Result<Self::Output, Self::Error> {
        let builder: Option<WordLevelBuilder> =
            unsafe { std::ptr::replace(&self.0 as *const _ as *mut _, None) };
        builder.unwrap().build().map_err(|e| format!("{}", e))
    }

    fn complete(
        self,
        mut cx: TaskContext,
        result: Result<Self::Output, Self::Error>,
    ) -> JsResult<Self::JsEvent> {
        let wordlevel = result.map_err(|e| cx.throw_error::<_, ()>(e).unwrap_err())?;

        let mut js_model = JsModel::new::<_, JsModel, _>(&mut cx, vec![])?;
        let guard = cx.lock();
        js_model.borrow_mut(&guard).model = Some(Arc::new(RwLock::new(wordlevel.into())));

        Ok(js_model.upcast())
    }
}

pub struct BPEFromFilesTask(Option<BpeBuilder>);
impl BPEFromFilesTask {
    pub fn new(builder: BpeBuilder) -> Self {
        Self(Some(builder))
    }
}

impl Task for BPEFromFilesTask {
    type Output = BPE;
    type Error = String;
    type JsEvent = JsValue;

    fn perform(&self) -> Result<Self::Output, Self::Error> {
        let builder: Option<BpeBuilder> =
            unsafe { std::ptr::replace(&self.0 as *const _ as *mut _, None) };
        builder.unwrap().build().map_err(|e| format!("{}", e))
    }

    fn complete(
        self,
        mut cx: TaskContext,
        result: Result<Self::Output, Self::Error>,
    ) -> JsResult<Self::JsEvent> {
        let bpe = result.map_err(|e| cx.throw_error::<_, ()>(e).unwrap_err())?;

        let mut js_model = JsModel::new::<_, JsModel, _>(&mut cx, vec![])?;
        let guard = cx.lock();
        js_model.borrow_mut(&guard).model = Some(Arc::new(RwLock::new(bpe.into())));

        Ok(js_model.upcast())
    }
}
