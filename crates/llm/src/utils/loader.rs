use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use candle_core::Device;
use serde::Deserialize;

use crate::models::{PackedShard, Qwen3ForCausalLM};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightLoadSummary {
    pub loaded_count: usize,
    pub skipped: Vec<String>,
}

impl WeightLoadSummary {
    pub fn skipped_count(&self) -> usize {
        self.skipped.len()
    }
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: std::collections::BTreeMap<String, String>,
}

fn resolve_weight_files(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let raw = fs::read_to_string(&index_path).with_context(|| {
            format!(
                "failed reading safetensors index at {}",
                index_path.display()
            )
        })?;
        let index: SafetensorsIndex = serde_json::from_str(&raw).with_context(|| {
            format!(
                "failed parsing safetensors index at {}",
                index_path.display()
            )
        })?;
        let mut unique_files = BTreeSet::new();
        for filename in index.weight_map.values() {
            unique_files.insert(filename.clone());
        }
        let files: Vec<PathBuf> = unique_files
            .into_iter()
            .map(|filename| model_dir.join(filename))
            .collect();
        ensure!(
            !files.is_empty(),
            "safetensors index has no weight_map entries at {}",
            index_path.display()
        );
        for file in &files {
            ensure!(
                file.exists(),
                "indexed safetensor file missing: {}",
                file.display()
            );
        }
        Ok(files)
    } else {
        let mut files = Vec::new();
        for entry in fs::read_dir(model_dir)
            .with_context(|| format!("failed listing model dir {}", model_dir.display()))?
        {
            let entry = entry.with_context(|| "failed reading model dir entry")?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                files.push(path);
            }
        }
        files.sort();
        ensure!(
            !files.is_empty(),
            "no .safetensors files found under {}",
            model_dir.display()
        );
        Ok(files)
    }
}

fn map_packed_weight_name(name: &str) -> Option<(String, PackedShard)> {
    Qwen3ForCausalLM::packed_module_for(name).map(|mapping| {
        (
            name.replacen(mapping.source_fragment, mapping.packed_module, 1),
            mapping.shard,
        )
    })
}

pub fn load_qwen3_weights_from_dir(
    model: &mut Qwen3ForCausalLM,
    model_dir: &Path,
    device: &Device,
) -> Result<WeightLoadSummary> {
    let files = resolve_weight_files(model_dir)?;
    let mut loaded_count = 0usize;
    let mut skipped = Vec::new();

    for file in files {
        let tensors = candle_core::safetensors::load(&file, device)
            .with_context(|| format!("failed loading safetensors file {}", file.display()))?;

        for (name, tensor) in tensors {
            if let Some((packed_name, shard)) = map_packed_weight_name(&name)
                && model.load_packed_shard(&packed_name, shard, &tensor)?
            {
                loaded_count += 1;
                continue;
            }

            if model.load_weight(&name, &tensor)? {
                loaded_count += 1;
            } else {
                skipped.push(name);
            }
        }
    }

    Ok(WeightLoadSummary {
        loaded_count,
        skipped,
    })
}

pub fn load_qwen3_weights_from_model_path(
    model: &mut Qwen3ForCausalLM,
    model_path: &str,
    device: &Device,
) -> Result<WeightLoadSummary> {
    let model_dir = PathBuf::from(model_path);
    ensure!(
        model_dir.is_dir(),
        "model path must be a directory: {}",
        model_dir.display()
    );
    load_qwen3_weights_from_dir(model, &model_dir, device)
}
