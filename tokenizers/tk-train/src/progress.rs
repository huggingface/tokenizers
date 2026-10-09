//! Coarse training counters and progress rendering outside token processing.
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};
use std::thread::JoinHandle;
use std::time::Duration;
use tk_encode::{Result, utils::progress::ProgressFormat};

pub(crate) struct TrainingProgress {
    renderer: Option<(mpsc::Sender<Message>, JoinHandle<()>)>,
}
enum Message {
    Stage(Arc<Stage>),
    Stop,
}
struct Stage {
    name: &'static str,
    total: u64,
    completed: AtomicUsize,
}
#[derive(Clone, Default)]
pub(crate) struct WorkProgress(Option<Arc<Stage>>);
impl TrainingProgress {
    pub(crate) fn new(enabled: bool, format: ProgressFormat) -> Result<Self> {
        // Upstream's show_progress flag controls only the interactive bar.
        if format == ProgressFormat::Silent
            || (format == ProgressFormat::Indicatif && (!enabled || !cfg!(feature = "progressbar")))
        {
            return Ok(Self { renderer: None });
        }
        let (sender, receiver) = mpsc::channel();
        let renderer = std::thread::Builder::new()
            .name("tokenizer-progress".into())
            .spawn(move || render(receiver, format))?;
        Ok(Self {
            renderer: Some((sender, renderer)),
        })
    }
    pub(crate) fn stage(&self, name: &'static str, total: usize) -> WorkProgress {
        let Some((sender, _)) = &self.renderer else {
            return WorkProgress::default();
        };
        let stage = Arc::new(Stage {
            name,
            total: total as u64,
            completed: AtomicUsize::new(0),
        });
        // A closed observer does not invalidate training.
        if sender.send(Message::Stage(Arc::clone(&stage))).is_err() {
            return WorkProgress::default();
        }
        WorkProgress(Some(stage))
    }
}
impl WorkProgress {
    pub(crate) fn complete(&self, amount: usize) {
        if let Some(stage) = &self.0 {
            stage.completed.fetch_add(amount, Ordering::Relaxed);
        }
    }
    pub(crate) fn learned(&self, rules: usize) {
        if let Some(stage) = &self.0 {
            stage.completed.store(rules, Ordering::Relaxed);
        }
    }
}
impl Drop for TrainingProgress {
    fn drop(&mut self) {
        if let Some((sender, renderer)) = self.renderer.take() {
            let _ = sender.send(Message::Stop);
            let _ = renderer.join();
        }
    }
}
fn render(receiver: mpsc::Receiver<Message>, format: ProgressFormat) {
    let mut current: Option<Arc<Stage>> = None;
    #[cfg(feature = "progressbar")]
    let bar = if format == ProgressFormat::Indicatif {
        let bar = indicatif::ProgressBar::new(0);
        bar.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {msg:<30!} {wide_bar} {pos:<9!}/{len:>9!}")
                .expect("the training progress template is a constant"),
        );
        Some(bar)
    } else {
        None
    };
    let show = |stage: &Stage, completed: u64, finished: bool| {
        let total = if finished { completed } else { stage.total };
        if format == ProgressFormat::JsonLines {
            eprintln!(
                "{}",
                serde_json::json!({"stage":stage.name,"current":completed,"total":total})
            );
        }
        #[cfg(feature = "progressbar")]
        if let Some(bar) = &bar {
            bar.set_length(total);
            bar.set_position(completed);
            bar.set_message(stage.name);
            bar.tick();
        }
    };
    loop {
        match receiver.recv_timeout(Duration::from_millis(250)) {
            Ok(Message::Stage(stage)) => {
                if let Some(previous) = current.take() {
                    show(
                        &previous,
                        previous.completed.load(Ordering::Relaxed) as u64,
                        true,
                    );
                }
                #[cfg(feature = "progressbar")]
                if let Some(bar) = &bar {
                    bar.reset_elapsed();
                }
                show(&stage, 0, false);
                current = Some(stage);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Some(stage) = &current {
                    show(stage, stage.completed.load(Ordering::Relaxed) as u64, false);
                }
            }
            Ok(Message::Stop) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    if let Some(stage) = current {
        show(&stage, stage.completed.load(Ordering::Relaxed) as u64, true);
    }
    #[cfg(feature = "progressbar")]
    if let Some(bar) = bar {
        bar.finish_and_clear();
    }
}
