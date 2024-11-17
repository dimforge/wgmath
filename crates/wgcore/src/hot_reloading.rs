use async_channel::{Receiver, Sender};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct HotReloadState {
    watcher: RecommendedWatcher,
    rcv: Receiver<notify::Result<Event>>,
    file_changed: HashMap<PathBuf, bool>,
}

impl HotReloadState {
    pub fn new() -> notify::Result<Self> {
        let (snd, rcv) = async_channel::unbounded();
        Ok(Self {
            watcher: notify::recommended_watcher(move |msg| {
                let _ = snd.send_blocking(msg);
            })?,
            rcv,
            file_changed: Default::default(),
        })
    }

    pub fn update_changes(&mut self) {
        for changed in self.file_changed.values_mut() {
            *changed = false;
        }

        while let Ok(event) = self.rcv.try_recv() {
            if let Ok(event) = event {
                if event.need_rescan() || matches!(event.kind, EventKind::Modify(_)) {
                    for path in event.paths {
                        self.file_changed.insert(path, true);
                    }
                }
            }
        }
    }

    pub fn watch_file(&mut self, path: &Path) -> notify::Result<()> {
        if !self.file_changed.contains_key(path) {
            self.watcher.watch(path, RecursiveMode::NonRecursive)?;
            // NOTE: this won’t insert if the watch failed.
            self.file_changed.insert(path.to_path_buf(), false);
        }

        Ok(())
    }

    pub fn file_changed(&self, path: &Path) -> bool {
        self.file_changed.get(path).copied().unwrap_or_default()
    }
}
