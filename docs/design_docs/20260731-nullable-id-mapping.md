# Nullable ID Mapping Design

- **Created:** 2026-07-31
- **Component:** `DataSet`, `IdMap`, `BitsetView`, `IndexNode`, EmbList strategies, backend adapters
- **Status:** Draft

## 1. Summary

Nullable vector fields can contain public rows or lists that do not have searchable vector payload. Knowhere handles this by allowing backends to build compact storage over only searchable payload while keeping public APIs, filters, and returned labels in the caller-visible id domain.

`IdMap` is the index-owned boundary between those domains. `DataSet` carries nullable input as non-owning views, `IdMap` derives and owns the forward and reverse maps, `BitsetView` projects public filters to the backend id domain, and `IndexNode` maps results back once before returning.

The design keeps three properties together:

- the backend may store a compact dense layout;
- callers continue to use public ids everywhere outside Knowhere backend internals;
- query paths borrow stable index-owned mapping buffers instead of rebuilding or copying them per query.

## 2. Goals

| Goal | Requirement |
|------|-------------|
| Preserve public API labels | Search results, range results, selected-id APIs, and raw-vector APIs use public row/list ids at the API boundary. |
| Allow compact backend storage | Backends may remove null rows or empty lists from their searchable payload and use dense backend ids internally. |
| Keep filter semantics stable | Input bitsets are always addressed by public ids, even when the backend searches compact ids. |
| Centralize nullable mapping | Mapping ownership and conversion rules live in `IdMap` and `IndexNode`, not in each caller. |
| Avoid query-local map copies | Search and iterator paths borrow index-owned map arrays. |
| Support sealed and growing indexes | Sealed maps publish immutable arrays; growing maps append new ranges while preserving previously published ids. |
| Support reload and mmap | Derived map arrays can be reconstructed after load, and sealed dense maps can use mmap-backed allocations. |
| Keep backend-local ids private | Relayout, partition, and sub-index ids are converted inside the backend wrapper before crossing shared boundaries. |

## 3. Concepts And Domains

### 3.1 ID Domains

Knowhere code that handles nullable data must keep these domains separate.

| Domain | Meaning | Examples |
|--------|---------|----------|
| Public id | Caller-visible row id for vector indexes, or caller-visible list/document id for EmbList indexes. | Input bitsets, selected ids, returned result labels. |
| Backend id | Dense id used by the backend after non-searchable public ids are compacted away. | ANN storage ids, raw-vector storage ids, rerank inputs. |
| Backend-local id | Private id introduced by backend relayout or partitioning. | HNSW sub-index local ids, chunk-local brute-force ids. |

For a vector index:

```text
public rows:        0  1  2  3  4  5
valid rows:         0     2        5
backend ids:        0     1        2
in_to_out_ids_:     [0, 2, 5]
out_to_in_ids_:     0->0, 2->1, 5->2, others->-1
```

The caller still filters and receives labels `0`, `2`, and `5`. The backend searches ids `0`, `1`, and `2`.

If the caller filters public id `5`, `BitsetView::test(2)` must return true because backend id `2` maps to public id `5`. If the backend returns id `1`, the result label returned by Knowhere is public id `2`.

### 3.2 EmbList Domains

EmbList adds a second public domain. A public EmbList id identifies a list/document; a backend base-vector id identifies one vector inside a compacted list range.

`emb_list_offset_` stores compact list offsets in backend base-vector space:

```text
compact list id:       0      1      2
base-vector range:   [0,2)  [2,2)  [2,5)
```

Nullable list compaction can make compact list ids different from public list ids. Strategies that search base vectors and evaluate list-level filters need a base-vector id -> public list id map. Strategies that search one encoded vector per list need only compact list id -> public list id at the normal result boundary.

## 4. Mapping State

### 4.1 Input View

`IdMapData` is a non-owning input view attached to `DataSet`.

| Format | Carries | Consumed as |
|--------|---------|-------------|
| `PACKED_BITMAP` | packed public-id validity bitmap | public id -> valid bit |
| `BOOL_ARRAY` | bool validity array | public id -> valid bit |
| `IDS` | compact backend id -> batch-local public id array plus public-domain count | direct forward map input |

`DataSet` does not own these buffers. The buffers must remain valid until the build/add wrapper consumes them into the index-owned `IdMap`.

### 4.2 Owned State

`IdMap` owns the derived nullable mapping state.

| Member | Domain | Purpose |
|--------|--------|---------|
| `valid_bitmap_` | public id -> valid bit | Records which public ids have searchable payload. Empty means identity/non-nullable. |
| `in_to_out_ids_` | backend id -> public row/list id | Maps backend result ids to public labels. |
| `out_to_in_ids_` | public row/list id -> backend id | Maps selected-id API inputs to backend ids. |
| `in_to_out_ebl_ids_` | backend base-vector id -> public EmbList id | Lets base-vector EmbList backends evaluate public list bitsets. |
| `out_count_` | public id domain size | Bounds public bitsets and selected ids. |
| `valid_count_` | compact backend id count | Counts searchable payload represented by the map. |

The forward and reverse maps must describe the same compacted layout. Updating one without the others changes the meaning of filters, results, or selected-id calls.

### 4.3 Storage Forms

`IdMap` selects its storage model before nullable data is consumed.

| Type | Used by | Storage rule |
|------|---------|--------------|
| `SEALED` | immutable build/load indexes | `ArrayStore` publishes complete contiguous arrays, optionally backed by mmap. |
| `GROWING` | appendable indexes | `ArrayStore` uses appendable storage so existing ids remain stable when a new public-id range is added. |

`AdaptiveStore` stores public id -> backend id lookup. It can use sparse storage for sealed heap maps when the public domain is much larger than the valid id count. Mmap-backed sealed maps and growing maps use dense storage so lookup memory has a predictable addressable range.

## 5. Lifecycle

### 5.1 Build

Build consumes nullable input before the backend build:

1. The caller attaches `IdMapData` to the build `DataSet`.
2. `Index<T>::Build()` consumes it into `IndexNode::GetIdMap()`.
3. The backend builds compact searchable payload.
4. `IndexNode::FinalizeIdMap()` derives missing maps after backend layout is available.

Validity input does not fully define every map. For example, a validity bitmap says which public ids are valid, but the backend or EmbList strategy may still need to finish offsets, relayout, or strategy state before all backend id -> public id maps can be derived.

An all-null build has no backend vector payload, but it still has public-domain nullable metadata. The build path finalizes the id map so search, selected-id validation, and load/reconstruct paths see the same public id domain as non-empty nullable indexes.

### 5.2 Add

Growing add consumes nullable input for the appended public range before the backend add:

1. The caller sets `IdMap::Type::GROWING` before append data is consumed.
2. `IdMapData` describes only the appended range.
3. `IdMap` rebases batch-local public ids by the current `out_count_`.
4. The backend receives compact payload for the valid rows in that appended range.

Tail append is the important concurrency property. Existing backend ids and their public labels stay stable while new ids are added after the current range.

Search visibility is still controlled by the caller's published public-id prefix. A search enters Knowhere with a public bitset for that prefix, and `PrepareBitset()` captures the mapped id count used by the backend for that request. Appends beyond that captured count are not part of the request.

`Index::Add()` returns backend add failures to the caller. The id map is consumed before backend add because backend layout and EmbList relayout may need the map while ingesting the payload, but Knowhere does not treat a failed backend add as recovered by rolling back only the id map. A caller that owns a growing or interim index must publish the new searchable prefix only after `Add()` succeeds.

### 5.3 Deserialize And File Load

`IdMap` is not serialized as a standalone index payload. The owner of the index restores nullable validity or compact id data around load, and `FinalizeIdMap()` reconstructs the derived maps after backend data and EmbList offsets are available.

This keeps the persisted backend index format focused on backend data. Nullable mapping is derived from the same public validity metadata used for build/add, so binary-set load and file load share the same mapping model.

### 5.4 Finalization

`FinalizeIdMap()` is the point where input metadata becomes backend-ready mapping state.

For vector indexes, finalization derives:

- `in_to_out_ids_` from public validity when no direct id array was supplied;
- `out_to_in_ids_` from the final forward map.

For EmbList indexes, finalization may also derive `in_to_out_ebl_ids_` after `emb_list_offset_` is available. That map is needed only when a base-vector backend evaluates filters in list/document space.

Backend overrides may add one more derived map for backend-local ids. That map must remain inside the backend wrapper.

## 6. Filtering

Input filters never change domain. A `BitsetView` always points to a public-id bitset supplied by the caller.

Backends call `BitsetView::test()` with the id they are about to accept or reject. `PrepareBitset()` configures how that backend id reaches the public bit:

| Projection | Used when | Meaning |
|------------|-----------|---------|
| `id_offset_` | backend ids form a contiguous public-id window | `public_id = backend_id + id_offset_` |
| `out_ids_` | backend ids need an explicit map | `public_id = out_ids_[backend_id]` |

`BitsetView::size()` and `BitsetView::count()` are in the backend id domain seen by the selector. `num_bits()` remains the public bitset size.

`filtered_count_` is optional because not every caller supplies an exact count. `std::nullopt` means unknown, while `0` means the projected filter is known to filter no backend ids. These states must not be collapsed together.

Backends that use filter ratio or empty-filter shortcuts opt in through `NeedBitsetExactCount()`. Exact counting intersects public filters with nullable validity and then reports the result in the backend vector/list domain.

Backends that accept only a contiguous filter bitmap materialize a backend-domain bitmap from `BitsetView::test()`. The source bitset still remains public-id addressed.

## 7. API And Result Boundaries

### 7.1 Search And Range Search

Backend search returns backend ids after any backend-local ids have already been handled inside the backend wrapper. `IndexNode::MapSearchResultIdsToOutIds()` maps those ids to public labels once before returning the result `DataSet`.

Negative result ids remain negative. They represent empty result slots and are not mapped.

### 7.2 Selected-ID APIs

Selected-id APIs accept public ids. `CompactOutToIn()` validates them and converts them to backend ids before raw-vector retrieval, distance calculation, or backend selected-id search.

`GetVectorByStorageIds()` is the internal boundary for ids that are already backend storage ids. Code that reaches this boundary must not map the ids again.

### 7.3 Brute Force

Brute-force kernels search contiguous buffers or chunk windows. The prepared `BitsetView` carries either a contiguous offset or an index-owned id map, so brute-force filtering and final result labels use the same public-id semantics as indexed search.

Chunked brute force adjusts only the backend id window. It must not rewrite the caller's public bitset.

### 7.4 Iterators

Iterators may evaluate a prepared `BitsetView` after the facade call that created them returns. When a prepared bitset borrows an index-owned id array, the wrapper keeps the prepared state alive for the iterator. The id array itself remains owned by the index.

## 8. EmbList Rules

EmbList strategies differ by the id domain used in their first ANN stage.

| Strategy shape | First ANN stage | Filter/result rule |
|----------------|-----------------|--------------------|
| Base-vector search | searches compact base-vector ids | Keep base-vector ids inside the strategy, use `in_to_out_ebl_ids_` for list filters, map final compact list ids to public list ids. |
| Encoded-list search | searches one compact vector per list | Use the normal search result mapping to public list ids, and map back to compact list ids only when rerank state needs compact offsets. |

The final `SearchEmbList()` boundary always returns public list ids.

`in_to_out_ebl_ids_` is intentionally separate from `in_to_out_ids_` because it maps a different backend id space. `in_to_out_ids_` maps compact lists or vectors; `in_to_out_ebl_ids_` maps each compact base vector to the public list that owns it.

Empty list metadata is still meaningful. A list can be public and valid while containing zero base vectors, or a nullable list can be absent from compact payload. Search code must keep list-domain validity and vector-domain payload counts separate.

## 9. Backend-Local Boundaries

Some backends introduce ids that are neither public ids nor the shared compact backend ids.

Examples:

- a multi-index backend can split one compact id range into sub-index local ids;
- a backend can relayout storage for its own search structure;
- a chunked brute-force path can search local chunk ids.

These ids must be translated before crossing back into common `IndexNode` behavior. Backend wrappers may install backend-local id -> public id arrays into `BitsetView`, or map local results to compact backend ids before using the common result mapper.

Composite nodes must apply the selected id-map type to every child that may consume nullable input. The outer node is still the public mapping boundary; child-specific maps are implementation details of that composite node.

## 10. Mmap

Nullable id-map mmap is a backing allocation for derived sealed arrays. It is not an additional serialized index format.

Rules:

- mmap is supported only for sealed id maps;
- mmap options must be configured before nullable data is written;
- forward maps and reverse lookup arrays may use mmap-backed allocation;
- the validity bitmap remains caller-derived metadata and is not stored as a separate mmap id-map file;
- backing files are owned by the map allocation and removed with the mmap region.

On load, the owner configures mmap options before restoring nullable input. Finalization then allocates the derived dense arrays with the configured backing files.

## 11. Invariants

- Public APIs, selected ids, input bitsets, and returned labels use public ids.
- Backend storage, raw-vector retrieval, and rerank internals use backend ids.
- Backend-local ids do not escape backend wrappers.
- Empty `valid_bitmap_` means identity/non-nullable mapping.
- Non-empty `valid_bitmap_` is addressed over `[0, out_count_)`.
- `valid_count_` is the compact backend count represented by the map.
- `in_to_out_ids_` and `out_to_in_ids_` must describe the same compact layout.
- `in_to_out_ebl_ids_` maps base-vector ids to public list ids and must not be used as a vector result map.
- `BitsetView::num_bits()` is public-domain size; `size()` and `count()` are backend-domain counts.
- Unknown filtered count is `std::nullopt`; known zero filtered count is `0`.
- `id_offset_` is for contiguous windows; `out_ids_` is for explicit projection.
- Search paths borrow index-owned map arrays and do not create per-query map copies.
- Growing maps append at the tail and preserve existing backend id labels.
- Sealed maps publish complete arrays after build/load finalization.
- Mmap applies only to sealed derived id arrays and must be configured before data is written.
