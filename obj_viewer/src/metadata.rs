use std::collections::HashMap;

#[cfg(target_arch = "wasm32")]
use serde::{Deserialize, Serialize};

/// Metadata entry for an IFC object with its mesh file
#[derive(Debug, Clone)]
#[cfg_attr(target_arch = "wasm32", derive(Serialize, Deserialize))]
pub struct ObjectMetadata {
    #[cfg_attr(target_arch = "wasm32", serde(rename = "relative_source_path"))]
    pub relative_source_path: String,

    #[cfg_attr(target_arch = "wasm32", serde(rename = "GlobalId"))]
    pub global_id: String,

    #[cfg_attr(target_arch = "wasm32", serde(rename = "IfcType"))]
    pub ifc_type: Option<String>,

    #[cfg_attr(target_arch = "wasm32", serde(rename = "mesh_filename"))]
    pub mesh_filename: String,
}

/// Manager for IFC metadata with efficient lookup
#[derive(Debug, Default)]
pub struct MetadataManager {
    /// All metadata entries
    objects: Vec<ObjectMetadata>,

    /// Quick lookup: mesh_filename -> index in objects
    filename_index: HashMap<String, usize>,

    /// Index by IFC type: IfcType -> list of indices
    type_index: HashMap<String, Vec<usize>>,
}

impl MetadataManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse and load metadata from JSON string
    #[cfg(target_arch = "wasm32")]
    pub fn from_json(json: &str) -> Result<Self, String> {
        let objects: Vec<ObjectMetadata> = serde_json::from_str(json)
            .map_err(|e| format!("Failed to parse metadata JSON: {}", e))?;

        Ok(Self::from_objects(objects))
    }

    /// Build manager from a list of objects
    pub fn from_objects(objects: Vec<ObjectMetadata>) -> Self {
        let mut filename_index = HashMap::new();
        let mut type_index: HashMap<String, Vec<usize>> = HashMap::new();

        for (idx, obj) in objects.iter().enumerate() {
            // Index by filename
            filename_index.insert(obj.mesh_filename.clone(), idx);

            // Index by IFC type
            if let Some(ref ifc_type) = obj.ifc_type {
                type_index
                    .entry(ifc_type.clone())
                    .or_insert_with(Vec::new)
                    .push(idx);
            }
        }

        Self {
            objects,
            filename_index,
            type_index,
        }
    }

    /// Get metadata by mesh filename
    pub fn get_by_filename(&self, filename: &str) -> Option<&ObjectMetadata> {
        self.filename_index
            .get(filename)
            .and_then(|&idx| self.objects.get(idx))
    }

    /// Get all unique IFC types, sorted alphabetically
    pub fn get_ifc_types(&self) -> Vec<String> {
        let mut types: Vec<String> = self.type_index.keys().cloned().collect();
        types.sort();
        types
    }

    /// Get all objects of a specific IFC type
    pub fn get_by_type(&self, ifc_type: &str) -> Vec<&ObjectMetadata> {
        self.type_index
            .get(ifc_type)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|&idx| self.objects.get(idx))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get count of objects for a specific IFC type
    pub fn count_by_type(&self, ifc_type: &str) -> usize {
        self.type_index
            .get(ifc_type)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Get total number of objects
    pub fn total_count(&self) -> usize {
        self.objects.len()
    }

    /// Check if metadata is loaded
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_indexing() {
        let objects = vec![
            ObjectMetadata {
                relative_source_path: "path1.ifc".to_string(),
                global_id: "id1".to_string(),
                ifc_type: Some("IfcBeam".to_string()),
                mesh_filename: "beam1.obj".to_string(),
            },
            ObjectMetadata {
                relative_source_path: "path2.ifc".to_string(),
                global_id: "id2".to_string(),
                ifc_type: Some("IfcBeam".to_string()),
                mesh_filename: "beam2.obj".to_string(),
            },
            ObjectMetadata {
                relative_source_path: "path3.ifc".to_string(),
                global_id: "id3".to_string(),
                ifc_type: Some("IfcColumn".to_string()),
                mesh_filename: "column1.obj".to_string(),
            },
        ];

        let manager = MetadataManager::from_objects(objects);

        assert_eq!(manager.total_count(), 3);
        assert_eq!(manager.count_by_type("IfcBeam"), 2);
        assert_eq!(manager.count_by_type("IfcColumn"), 1);
        assert!(manager.get_by_filename("beam1.obj").is_some());

        let types = manager.get_ifc_types();
        assert_eq!(types, vec!["IfcBeam", "IfcColumn"]);
    }
}
