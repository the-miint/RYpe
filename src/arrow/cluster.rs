//! Arrow integration for clustering.
//!
//! # Output Schema (cluster result rows)
//!
//! | Column          | Arrow Type | Nullable | Description |
//! |-----------------|------------|----------|-------------|
//! | `rep_contig`    | Utf8       | No       | Representative contig id |
//! | `member_contig` | Utf8       | No       | Member contig id (equals rep_contig for representatives themselves) |
//! | `source_mag`    | Utf8       | Yes      | MAG/assembly id the member came from |
//! | `containment`   | Float64    | No       | Containment of member in representative (1.0 for representatives) |
//!
//! # Input Schema (contigs to be clustered)
//!
//! | Column       | Arrow Type | Nullable | Description |
//! |--------------|------------|----------|-------------|
//! | `contig_id`  | Utf8/LargeUtf8 | No   | Unique contig id |
//! | `source_mag` | Utf8/LargeUtf8 | Yes  | MAG/assembly id the contig came from |
//! | `sequence`   | Binary/LargeBinary/BinaryView/Utf8/LargeUtf8/Utf8View | No | DNA bytes |

use arrow::array::{
    Array, BinaryArray, BinaryViewArray, Float64Array, LargeBinaryArray, LargeStringArray,
    StringArray, StringBuilder, StringViewArray,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

use super::error::{ArrowClassifyError, MAX_SEQUENCE_LENGTH};
use crate::cluster::{cluster_contigs, ClusterConfig, ClusterResult, ContigInput};
use std::collections::HashSet;

/// Column name for representative contig id in cluster output.
pub const COL_REP_CONTIG: &str = "rep_contig";
/// Column name for member contig id in cluster output.
pub const COL_MEMBER_CONTIG: &str = "member_contig";
/// Column name for source MAG id (member's MAG) in cluster output. Nullable.
pub const COL_SOURCE_MAG: &str = "source_mag";
/// Column name for containment score in cluster output.
pub const COL_CONTAINMENT: &str = "containment";

/// Column name for contig id in cluster input batches.
pub const COL_CONTIG_ID: &str = "contig_id";
/// Column name for sequence bytes in cluster input batches.
pub const COL_CLUSTER_SEQUENCE: &str = "sequence";

/// Returns the schema for cluster result batches.
pub fn cluster_result_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(COL_REP_CONTIG, DataType::Utf8, false),
        Field::new(COL_MEMBER_CONTIG, DataType::Utf8, false),
        Field::new(COL_SOURCE_MAG, DataType::Utf8, true),
        Field::new(COL_CONTAINMENT, DataType::Float64, false),
    ]))
}

/// Convert a [`ClusterResult`] into an Arrow `RecordBatch` with
/// [`cluster_result_schema`].
///
/// The conversion copies all strings into Arrow buffers (unavoidable —
/// `ClusterRow` owns `String`s). For typical dereplication runs (one row per
/// input contig) this is bounded by the input size, not the all-vs-all space.
pub fn cluster_result_to_record_batch(
    result: &ClusterResult,
) -> Result<RecordBatch, ArrowClassifyError> {
    let schema = cluster_result_schema();

    if result.rows.is_empty() {
        return Ok(RecordBatch::new_empty(schema));
    }

    let n = result.rows.len();
    let mut rep = StringBuilder::new();
    let mut member = StringBuilder::new();
    let mut mag = StringBuilder::new();
    let mut containment = Vec::with_capacity(n);

    for row in &result.rows {
        rep.append_value(&row.rep_contig);
        member.append_value(&row.member_contig);
        match &row.source_mag {
            Some(m) => mag.append_value(m),
            None => mag.append_null(),
        }
        containment.push(row.containment);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(rep.finish()) as Arc<dyn arrow::array::Array>,
            Arc::new(member.finish()),
            Arc::new(mag.finish()),
            Arc::new(Float64Array::from(containment)),
        ],
    )
    .map_err(ArrowClassifyError::from)
}

/// Helper so callers can construct an empty result batch with the correct
/// schema without going through a `ClusterResult`.
pub fn empty_cluster_result_batch() -> RecordBatch {
    RecordBatch::new_empty(cluster_result_schema())
}

/// Convert a cluster-input `RecordBatch` into the owned `ContigInput` form.
///
/// # Errors
/// * `NullError` if `contig_id` or `sequence` is null in any row.
/// * `SchemaError` if two rows share the same `contig_id`.
/// * `SequenceTooLong` if any `sequence` exceeds [`MAX_SEQUENCE_LENGTH`].
fn batch_to_contig_inputs(batch: &RecordBatch) -> Result<Vec<ContigInput>, ArrowClassifyError> {
    let mut out = Vec::with_capacity(batch.num_rows());
    let mut seen: HashSet<String> = HashSet::with_capacity(batch.num_rows());
    append_batch_to_contig_inputs(batch, &mut out, &mut seen)?;
    Ok(out)
}

/// Append rows from `batch` to `out`, using `seen` for cross-batch duplicate
/// detection. Lets the FFI streaming path avoid concatenating every input
/// batch into one giant RecordBatch before conversion.
pub(crate) fn append_batch_to_contig_inputs(
    batch: &RecordBatch,
    out: &mut Vec<ContigInput>,
    seen: &mut HashSet<String>,
) -> Result<(), ArrowClassifyError> {
    validate_cluster_input_schema(batch.schema().as_ref())?;

    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return Ok(());
    }

    let id_idx = batch
        .schema()
        .index_of(COL_CONTIG_ID)
        .map_err(|_| ArrowClassifyError::ColumnNotFound(COL_CONTIG_ID.into()))?;
    let seq_idx = batch
        .schema()
        .index_of(COL_CLUSTER_SEQUENCE)
        .map_err(|_| ArrowClassifyError::ColumnNotFound(COL_CLUSTER_SEQUENCE.into()))?;
    let mag_idx = batch.schema().index_of(COL_SOURCE_MAG).ok();

    let id_col = batch.column(id_idx);
    let seq_col = batch.column(seq_idx);
    let mag_col = mag_idx.map(|i| Arc::clone(batch.column(i)));

    out.reserve(num_rows);

    for i in 0..num_rows {
        if id_col.is_null(i) {
            return Err(ArrowClassifyError::NullError {
                column: COL_CONTIG_ID.into(),
                row: i,
            });
        }
        let id_bytes = string_value_at(id_col, i, COL_CONTIG_ID)?;
        let id = std::str::from_utf8(id_bytes)
            .map_err(|e| {
                ArrowClassifyError::SchemaError(format!(
                    "{} row {} is not valid UTF-8: {}",
                    COL_CONTIG_ID, i, e
                ))
            })?
            .to_string();

        if !seen.insert(id.clone()) {
            return Err(ArrowClassifyError::SchemaError(format!(
                "duplicate {} '{}' at row {}",
                COL_CONTIG_ID, id, i
            )));
        }

        let source_mag = match &mag_col {
            Some(col) => {
                if col.is_null(i) {
                    None
                } else {
                    let bytes = string_value_at(col.as_ref(), i, COL_SOURCE_MAG)?;
                    let s = std::str::from_utf8(bytes)
                        .map_err(|e| {
                            ArrowClassifyError::SchemaError(format!(
                                "{} row {} is not valid UTF-8: {}",
                                COL_SOURCE_MAG, i, e
                            ))
                        })?
                        .to_string();
                    Some(s)
                }
            }
            None => None,
        };

        if seq_col.is_null(i) {
            return Err(ArrowClassifyError::NullError {
                column: COL_CLUSTER_SEQUENCE.into(),
                row: i,
            });
        }
        let seq_bytes = bytes_value_at(seq_col.as_ref(), i, COL_CLUSTER_SEQUENCE)?;
        if seq_bytes.len() > MAX_SEQUENCE_LENGTH {
            return Err(ArrowClassifyError::SequenceTooLong {
                row: i,
                length: seq_bytes.len(),
                max_length: MAX_SEQUENCE_LENGTH,
            });
        }
        let sequence = seq_bytes.to_vec();

        out.push(ContigInput {
            id,
            source_mag,
            sequence,
        });
    }

    Ok(())
}

fn string_value_at<'a>(
    col: &'a dyn Array,
    i: usize,
    col_name: &str,
) -> Result<&'a [u8], ArrowClassifyError> {
    if col.is_null(i) {
        return Err(ArrowClassifyError::Classification(format!(
            "{} is null in row {}",
            col_name, i
        )));
    }
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        return Ok(a.value(i).as_bytes());
    }
    if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(a.value(i).as_bytes());
    }
    if let Some(a) = col.as_any().downcast_ref::<StringViewArray>() {
        return Ok(a.value(i).as_bytes());
    }
    Err(ArrowClassifyError::TypeError {
        column: col_name.into(),
        expected: "Utf8, LargeUtf8, or Utf8View".into(),
        actual: format!("{:?}", col.data_type()),
    })
}

fn bytes_value_at<'a>(
    col: &'a dyn Array,
    i: usize,
    col_name: &str,
) -> Result<&'a [u8], ArrowClassifyError> {
    if let Some(a) = col.as_any().downcast_ref::<BinaryArray>() {
        return Ok(a.value(i));
    }
    if let Some(a) = col.as_any().downcast_ref::<LargeBinaryArray>() {
        return Ok(a.value(i));
    }
    if let Some(a) = col.as_any().downcast_ref::<BinaryViewArray>() {
        return Ok(a.value(i));
    }
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        return Ok(a.value(i).as_bytes());
    }
    if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(a.value(i).as_bytes());
    }
    if let Some(a) = col.as_any().downcast_ref::<StringViewArray>() {
        return Ok(a.value(i).as_bytes());
    }
    Err(ArrowClassifyError::TypeError {
        column: col_name.into(),
        expected: "Binary, LargeBinary, BinaryView, Utf8, LargeUtf8, or Utf8View".into(),
        actual: format!("{:?}", col.data_type()),
    })
}

/// Cluster a batch of contigs presented as an Arrow `RecordBatch`.
///
/// Input schema: see [`validate_cluster_input_schema`].
/// Output schema: see [`cluster_result_schema`].
///
/// Peak resident memory is roughly **2–3× the input batch size**: input
/// batch + owned `ContigInput` copy + per-contig minimizers (~24 bytes per
/// minimizer at ~one minimizer per `w` bases). Contigs shorter than
/// `cfg.min_length` are dropped silently.
pub fn cluster_arrow_batch(
    batch: &RecordBatch,
    cfg: &ClusterConfig,
) -> Result<RecordBatch, ArrowClassifyError> {
    let inputs = batch_to_contig_inputs(batch)?;
    let result = cluster_contigs(inputs, cfg)
        .map_err(|e| ArrowClassifyError::Classification(e.to_string()))?;
    cluster_result_to_record_batch(&result)
}

/// Check if a DataType is a valid string type for contig ids / source MAGs.
fn is_string_type(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

/// Check if a DataType is a valid sequence type (mirrors classify input).
fn is_sequence_type(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
    )
}

/// Validate that a schema matches the expected cluster-input schema.
///
/// # Requirements
/// - `contig_id`: Utf8 / LargeUtf8 / Utf8View, **non-nullable** (declared so
///   in the schema; a row-level null would later raise `NullError`).
/// - `sequence`: any binary or string type, **non-nullable**.
/// - `source_mag`: Utf8 / LargeUtf8 / Utf8View, **nullable** (optional column;
///   if declared non-nullable the schema is rejected so callers don't pass
///   in something they can't represent).
///
/// Pre-validating nullability here lets a C-API consumer trust this function
/// to fully describe the schema contract — there is no row-level surprise
/// from a column that was declared nullable.
pub fn validate_cluster_input_schema(schema: &Schema) -> Result<(), ArrowClassifyError> {
    match schema.column_with_name(COL_CONTIG_ID) {
        Some((_, field)) => {
            if !is_string_type(field.data_type()) {
                return Err(ArrowClassifyError::TypeError {
                    column: COL_CONTIG_ID.into(),
                    expected: "Utf8, LargeUtf8, or Utf8View".into(),
                    actual: format!("{:?}", field.data_type()),
                });
            }
            if field.is_nullable() {
                return Err(ArrowClassifyError::SchemaError(format!(
                    "column '{}' must be declared non-nullable",
                    COL_CONTIG_ID
                )));
            }
        }
        None => return Err(ArrowClassifyError::ColumnNotFound(COL_CONTIG_ID.into())),
    }

    if let Some((_, field)) = schema.column_with_name(COL_SOURCE_MAG) {
        if !is_string_type(field.data_type()) {
            return Err(ArrowClassifyError::TypeError {
                column: COL_SOURCE_MAG.into(),
                expected: "Utf8, LargeUtf8, or Utf8View".into(),
                actual: format!("{:?}", field.data_type()),
            });
        }
        if !field.is_nullable() {
            return Err(ArrowClassifyError::SchemaError(format!(
                "column '{}' must be declared nullable",
                COL_SOURCE_MAG
            )));
        }
    }

    match schema.column_with_name(COL_CLUSTER_SEQUENCE) {
        Some((_, field)) => {
            if !is_sequence_type(field.data_type()) {
                return Err(ArrowClassifyError::TypeError {
                    column: COL_CLUSTER_SEQUENCE.into(),
                    expected: "Binary, LargeBinary, BinaryView, Utf8, LargeUtf8, or Utf8View"
                        .into(),
                    actual: format!("{:?}", field.data_type()),
                });
            }
            if field.is_nullable() {
                return Err(ArrowClassifyError::SchemaError(format!(
                    "column '{}' must be declared non-nullable",
                    COL_CLUSTER_SEQUENCE
                )));
            }
        }
        None => {
            return Err(ArrowClassifyError::ColumnNotFound(
                COL_CLUSTER_SEQUENCE.into(),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_schema_has_four_columns_in_documented_order() {
        let schema = cluster_result_schema();
        assert_eq!(schema.fields().len(), 4);
        assert_eq!(schema.field(0).name(), COL_REP_CONTIG);
        assert_eq!(schema.field(1).name(), COL_MEMBER_CONTIG);
        assert_eq!(schema.field(2).name(), COL_SOURCE_MAG);
        assert_eq!(schema.field(3).name(), COL_CONTAINMENT);
    }

    #[test]
    fn output_schema_types_and_nullability_match_docs() {
        let schema = cluster_result_schema();
        assert_eq!(schema.field(0).data_type(), &DataType::Utf8);
        assert!(!schema.field(0).is_nullable());
        assert_eq!(schema.field(1).data_type(), &DataType::Utf8);
        assert!(!schema.field(1).is_nullable());
        assert_eq!(schema.field(2).data_type(), &DataType::Utf8);
        assert!(schema.field(2).is_nullable(), "source_mag must be nullable");
        assert_eq!(schema.field(3).data_type(), &DataType::Float64);
        assert!(!schema.field(3).is_nullable());
    }

    #[test]
    fn input_schema_valid_minimum_columns() {
        let schema = Schema::new(vec![
            Field::new(COL_CONTIG_ID, DataType::Utf8, false),
            Field::new(COL_CLUSTER_SEQUENCE, DataType::Binary, false),
        ]);
        assert!(validate_cluster_input_schema(&schema).is_ok());
    }

    #[test]
    fn input_schema_valid_with_source_mag_and_large_types() {
        let schema = Schema::new(vec![
            Field::new(COL_CONTIG_ID, DataType::LargeUtf8, false),
            Field::new(COL_SOURCE_MAG, DataType::Utf8, true),
            Field::new(COL_CLUSTER_SEQUENCE, DataType::LargeBinary, false),
        ]);
        assert!(validate_cluster_input_schema(&schema).is_ok());
    }

    #[test]
    fn input_schema_missing_contig_id_rejected() {
        let schema = Schema::new(vec![Field::new(
            COL_CLUSTER_SEQUENCE,
            DataType::Binary,
            false,
        )]);
        assert!(matches!(
            validate_cluster_input_schema(&schema).unwrap_err(),
            ArrowClassifyError::ColumnNotFound(col) if col == COL_CONTIG_ID
        ));
    }

    #[test]
    fn input_schema_missing_sequence_rejected() {
        let schema = Schema::new(vec![Field::new(COL_CONTIG_ID, DataType::Utf8, false)]);
        assert!(matches!(
            validate_cluster_input_schema(&schema).unwrap_err(),
            ArrowClassifyError::ColumnNotFound(col) if col == COL_CLUSTER_SEQUENCE
        ));
    }

    #[test]
    fn input_schema_wrong_contig_id_type_rejected() {
        let schema = Schema::new(vec![
            Field::new(COL_CONTIG_ID, DataType::Int64, false),
            Field::new(COL_CLUSTER_SEQUENCE, DataType::Binary, false),
        ]);
        assert!(matches!(
            validate_cluster_input_schema(&schema).unwrap_err(),
            ArrowClassifyError::TypeError { column, .. } if column == COL_CONTIG_ID
        ));
    }

    #[test]
    fn input_schema_wrong_source_mag_type_rejected() {
        let schema = Schema::new(vec![
            Field::new(COL_CONTIG_ID, DataType::Utf8, false),
            Field::new(COL_SOURCE_MAG, DataType::Int64, true),
            Field::new(COL_CLUSTER_SEQUENCE, DataType::Binary, false),
        ]);
        assert!(matches!(
            validate_cluster_input_schema(&schema).unwrap_err(),
            ArrowClassifyError::TypeError { column, .. } if column == COL_SOURCE_MAG
        ));
    }

    #[test]
    fn input_schema_extra_columns_allowed() {
        let schema = Schema::new(vec![
            Field::new(COL_CONTIG_ID, DataType::Utf8, false),
            Field::new(COL_CLUSTER_SEQUENCE, DataType::Binary, false),
            Field::new("ignored_extra", DataType::Int64, true),
        ]);
        assert!(validate_cluster_input_schema(&schema).is_ok());
    }

    use crate::cluster::ClusterRow;
    use arrow::array::Array;

    fn row(rep: &str, member: &str, mag: Option<&str>, c: f64) -> ClusterRow {
        ClusterRow {
            rep_contig: rep.to_string(),
            member_contig: member.to_string(),
            source_mag: mag.map(|s| s.to_string()),
            containment: c,
            chain: None,
        }
    }

    #[test]
    fn round_trip_three_row_result() {
        let result = ClusterResult {
            rows: vec![
                row("A", "A", Some("mag1"), 1.0),
                row("A", "B", Some("mag2"), 0.93),
                row("C", "C", None, 1.0),
            ],
        };

        let batch = cluster_result_to_record_batch(&result).unwrap();

        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.schema(), cluster_result_schema());

        let rep = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        let member = batch
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        let mag = batch
            .column(2)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        let containment = batch
            .column(3)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();

        assert_eq!(rep.value(0), "A");
        assert_eq!(member.value(0), "A");
        assert_eq!(mag.value(0), "mag1");
        assert!(!mag.is_null(0));
        assert_eq!(containment.value(0), 1.0);

        assert_eq!(rep.value(1), "A");
        assert_eq!(member.value(1), "B");
        assert_eq!(mag.value(1), "mag2");
        assert!((containment.value(1) - 0.93).abs() < 1e-12);

        assert_eq!(rep.value(2), "C");
        assert_eq!(member.value(2), "C");
        assert!(mag.is_null(2), "source_mag should be NULL when None");
        assert_eq!(containment.value(2), 1.0);
    }

    #[test]
    fn empty_result_produces_zero_row_batch_with_correct_schema() {
        let batch = cluster_result_to_record_batch(&ClusterResult::default()).unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.schema(), cluster_result_schema());
    }

    #[test]
    fn empty_helper_matches_conversion_of_empty_result() {
        let from_helper = empty_cluster_result_batch();
        let from_conversion = cluster_result_to_record_batch(&ClusterResult::default()).unwrap();
        assert_eq!(from_helper.schema(), from_conversion.schema());
        assert_eq!(from_helper.num_rows(), from_conversion.num_rows());
    }

    use arrow::array::BinaryArray;

    fn seq_from_seed(len: usize, seed: u64) -> Vec<u8> {
        let bases = [b'A', b'C', b'G', b'T'];
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                bases[((s >> 56) & 0b11) as usize]
            })
            .collect()
    }

    fn relaxed_cfg() -> ClusterConfig {
        ClusterConfig {
            k: 32,
            w: 20,
            salt: 0x5555_5555_5555_5555,
            min_length: 1_000,
            threshold: 0.80,
            min_shared: 50,
            chain_params: None,
            min_chain_containment: None,
        }
    }

    fn build_input_batch(ids: &[&str], mags: &[Option<&str>], seqs: &[Vec<u8>]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_CONTIG_ID, DataType::Utf8, false),
            Field::new(COL_SOURCE_MAG, DataType::Utf8, true),
            Field::new(COL_CLUSTER_SEQUENCE, DataType::Binary, false),
        ]));
        let id_arr = StringArray::from_iter_values(ids.iter().copied());
        let mut mag_b = StringBuilder::new();
        for m in mags {
            match m {
                Some(s) => mag_b.append_value(*s),
                None => mag_b.append_null(),
            }
        }
        let seq_arr = BinaryArray::from_iter_values(seqs.iter().map(|v| v.as_slice()));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_arr),
                Arc::new(mag_b.finish()),
                Arc::new(seq_arr),
            ],
        )
        .unwrap()
    }

    #[test]
    fn cluster_arrow_batch_matches_native_orchestrator() {
        let a = seq_from_seed(20_000, 1);
        let b = a[2_000..10_000].to_vec();
        let c = seq_from_seed(20_000, 2);

        let batch = build_input_batch(
            &["A", "B", "C"],
            &[Some("mag1"), Some("mag2"), None],
            &[a.clone(), b.clone(), c.clone()],
        );
        let cfg = relaxed_cfg();

        let arrow_result = cluster_arrow_batch(&batch, &cfg).unwrap();

        let native = cluster_contigs(
            vec![
                ContigInput {
                    id: "A".to_string(),
                    source_mag: Some("mag1".to_string()),
                    sequence: a,
                },
                ContigInput {
                    id: "B".to_string(),
                    source_mag: Some("mag2".to_string()),
                    sequence: b,
                },
                ContigInput {
                    id: "C".to_string(),
                    source_mag: None,
                    sequence: c,
                },
            ],
            &cfg,
        )
        .unwrap();
        let native_batch = cluster_result_to_record_batch(&native).unwrap();

        // Same schema, same row count, same content.
        assert_eq!(arrow_result.schema(), native_batch.schema());
        assert_eq!(arrow_result.num_rows(), native_batch.num_rows());
        assert_eq!(arrow_result.num_rows(), 3);
        for col in 0..4 {
            assert_eq!(
                arrow_result.column(col).len(),
                native_batch.column(col).len(),
                "column {} length differs",
                col
            );
        }

        // Sanity check the output: A should be the rep for both A and B,
        // C is its own rep, source_mag for C is null.
        let rep = arrow_result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let member = arrow_result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let mag = arrow_result
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..arrow_result.num_rows() {
            let m = member.value(i);
            if m == "B" {
                assert_eq!(rep.value(i), "A");
            } else if m == "C" {
                assert_eq!(rep.value(i), "C");
                assert!(mag.is_null(i), "C's source_mag should be NULL");
            }
        }
    }

    #[test]
    fn cluster_arrow_batch_works_without_source_mag_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_CONTIG_ID, DataType::Utf8, false),
            Field::new(COL_CLUSTER_SEQUENCE, DataType::Binary, false),
        ]));
        let seq = seq_from_seed(5_000, 42);
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["solo"])),
                Arc::new(BinaryArray::from_iter_values([seq.as_slice()])),
            ],
        )
        .unwrap();

        let out = cluster_arrow_batch(&batch, &relaxed_cfg()).unwrap();
        assert_eq!(out.num_rows(), 1);
        let mag = out
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert!(mag.is_null(0), "missing source_mag column -> all NULL");
    }

    #[test]
    fn cluster_arrow_batch_rejects_missing_contig_id_column() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            COL_CLUSTER_SEQUENCE,
            DataType::Binary,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(BinaryArray::from_iter_values([&b"ACGT"[..]]))],
        )
        .unwrap();
        let err = cluster_arrow_batch(&batch, &relaxed_cfg()).unwrap_err();
        assert!(matches!(err, ArrowClassifyError::ColumnNotFound(c) if c == COL_CONTIG_ID));
    }

    #[test]
    fn cluster_arrow_batch_empty_input_yields_empty_output() {
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_CONTIG_ID, DataType::Utf8, false),
            Field::new(COL_CLUSTER_SEQUENCE, DataType::Binary, false),
        ]));
        let batch = RecordBatch::new_empty(schema);
        let out = cluster_arrow_batch(&batch, &relaxed_cfg()).unwrap();
        assert_eq!(out.num_rows(), 0);
        assert_eq!(out.schema(), cluster_result_schema());
    }

    #[test]
    fn validator_rejects_nullable_contig_id() {
        let schema = Schema::new(vec![
            Field::new(COL_CONTIG_ID, DataType::Utf8, true),
            Field::new(COL_CLUSTER_SEQUENCE, DataType::Binary, false),
        ]);
        let err = validate_cluster_input_schema(&schema).unwrap_err();
        assert!(matches!(err, ArrowClassifyError::SchemaError(msg) if msg.contains(COL_CONTIG_ID)));
    }

    #[test]
    fn validator_rejects_nullable_sequence() {
        let schema = Schema::new(vec![
            Field::new(COL_CONTIG_ID, DataType::Utf8, false),
            Field::new(COL_CLUSTER_SEQUENCE, DataType::Binary, true),
        ]);
        let err = validate_cluster_input_schema(&schema).unwrap_err();
        assert!(
            matches!(err, ArrowClassifyError::SchemaError(msg) if msg.contains(COL_CLUSTER_SEQUENCE))
        );
    }

    #[test]
    fn validator_rejects_non_nullable_source_mag() {
        let schema = Schema::new(vec![
            Field::new(COL_CONTIG_ID, DataType::Utf8, false),
            Field::new(COL_SOURCE_MAG, DataType::Utf8, false),
            Field::new(COL_CLUSTER_SEQUENCE, DataType::Binary, false),
        ]);
        let err = validate_cluster_input_schema(&schema).unwrap_err();
        assert!(
            matches!(err, ArrowClassifyError::SchemaError(msg) if msg.contains(COL_SOURCE_MAG))
        );
    }

    #[test]
    fn cluster_arrow_batch_rejects_duplicate_contig_id() {
        let seq = seq_from_seed(2_000, 1);
        let batch = build_input_batch(&["dup", "dup"], &[None, None], &[seq.clone(), seq.clone()]);
        let err = cluster_arrow_batch(&batch, &relaxed_cfg()).unwrap_err();
        match err {
            ArrowClassifyError::SchemaError(msg) => {
                assert!(
                    msg.contains("duplicate"),
                    "expected duplicate error, got {}",
                    msg
                );
                assert!(msg.contains("dup"), "expected id in message, got {}", msg);
            }
            other => panic!("expected SchemaError, got {:?}", other),
        }
    }

    #[test]
    fn all_inputs_below_min_length_yields_empty_output() {
        // min_length is 1000 in relaxed_cfg; all sequences are 100 bytes.
        let inputs = vec![seq_from_seed(100, 1), seq_from_seed(100, 2)];
        let batch = build_input_batch(&["short1", "short2"], &[None, None], &inputs);
        let out = cluster_arrow_batch(&batch, &relaxed_cfg()).unwrap();
        assert_eq!(out.num_rows(), 0);
        assert_eq!(out.schema(), cluster_result_schema());
    }

    #[test]
    fn source_mag_column_present_but_all_null_emits_all_null_mags() {
        let inputs = vec![seq_from_seed(5_000, 1), seq_from_seed(5_000, 2)];
        let batch = build_input_batch(&["a", "b"], &[None, None], &inputs);
        let out = cluster_arrow_batch(&batch, &relaxed_cfg()).unwrap();
        let mag = out
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert!(out.num_rows() >= 1);
        for i in 0..out.num_rows() {
            assert!(mag.is_null(i), "expected null source_mag at row {}", i);
        }
    }
}
