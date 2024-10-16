//! Extensions over naga-oil’s Composer.

use naga_oil::compose::preprocess::Preprocessor;
use naga_oil::compose::{
    ComposableModuleDefinition, ComposableModuleDescriptor, Composer, ComposerError, ErrSource,
};

/// An extension trait for the naga-oil `Composer` to work around some of its limitations.
pub trait ComposerExt {
    /// Adds a composable module to `self` only if it hasn’t been added yet.
    ///
    /// Currently, `naga-oil` behaves strangely (some symbols stop resolving) if the same module is
    /// added twice. This function checks if the module has already been added. If it was already
    /// added, then `self` is left unchanged and `Ok(None)` is returned.
    fn add_composable_module_once(
        &mut self,
        desc: ComposableModuleDescriptor<'_>,
    ) -> Result<Option<&ComposableModuleDefinition>, ComposerError>;
}

impl ComposerExt for Composer {
    fn add_composable_module_once(
        &mut self,
        desc: ComposableModuleDescriptor<'_>,
    ) -> Result<Option<&ComposableModuleDefinition>, ComposerError> {
        let prep = Preprocessor::default();
        // TODO: not sure if allow_defines should be `true` or `false`.
        let meta = prep
            .get_preprocessor_metadata(desc.source, false)
            .map_err(|inner| ComposerError {
                inner,
                source: ErrSource::Constructing {
                    path: desc.file_path.to_string(),
                    source: desc.source.to_string(),
                    offset: 0,
                },
            })?;

        if let Some(name) = &meta.name {
            if self.contains_module(name) {
                // Module already exists, don’t insert it.
                return Ok(None);
            }
        }

        self.add_composable_module(desc).map(Some)
    }
}
