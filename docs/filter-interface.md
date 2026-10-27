# Deferred filter transport foundation

`BitsetView` supports either the original bitmap-only representation or a
mandatory bitmap plus an opaque candidate callback. A callback-only view must
declare its public row count explicitly; an empty mandatory bitmap does not mean
that the user predicate is absent. Public row offsets are not primary keys or
backend reordered IDs.

```cpp
BitsetView description(mandatory_bits, public_rows, mandatory_filtered_count);
description.set_candidate_evaluator(&descriptor);
// For no mandatory bitmap instead:
// BitsetView description;
// description.set_candidate_evaluator(&descriptor, public_rows);

auto task_filter = description.bind();
uint64_t excluded = task_filter.test(ids, count, active_mask);
```

The producer accepts at most 64 lanes per callback. Bit `i` in the returned
`test()` mask is **excluded**, whereas bit `i` from the producer is **accepted**
(TRUE and known). Inactive lanes, invalid mappings and mandatory exclusions are
removed before the producer sees them. Generic expression engines may evaluate
larger batches; splitting into transport windows is an adapter responsibility.
Count, filter ratio and raw mandatory bits never represent estimated or exact
user-predicate truth. `data()` rejects deferred views. As in the original
mapping/bounds fast-access contract, `test_direct()` requires a successful
`supports_direct_test()` check before the loop; the latter always rejects
deferred representations. Debug assertions verify that precondition without
adding a per-ID representation branch in ordinary bitmap loops. Production
search ingress independently remains fail-closed for deferred inputs.
Both `count_filtered_bits()` overloads reject deferred views without changing
their size/count/public domain; backend count projection belongs to later
consumer adaptation. Prepare exact mandatory counts before attachment when
needed, never install estimated predicate truth as an exact count.

V1 transport uses int32 backend IDs and int32 public row offsets. Larger-ID
consumer adaptation is not included; callback single-ID tests outside the
int32 backend domain are excluded. The old bitmap-only int64 single-ID API
retains its existing behavior.

One independent task calls `bind()` once and reuses its workspace across
batches. Copies of that bound filter share the same task state; they must not
be evaluated concurrently. Independent tasks bind separately. The immutable
descriptor, factory context, source owners and callback-providing library must
outlive every bound filter and its destruction. A bind is not a deep copy of
query sources and does not make a shared mutable factory thread-safe.

The V1 descriptor has a frozen size/version. Callbacks are `noexcept` and return
explicit status; create/eval errors are not false predicate results. Destruction
occurs through the producer callback, including partial failed construction.
This fixed descriptor contract does **not** promise arbitrary binary mixing of
old/new C++ `BitsetView` layouts. Recompile Knowhere and every affected consumer
with matching headers and verify the loaded library closure.

## Staged execution boundary

This change introduces transport and tests, not native/Cardinal search support
or a strategy decision. Production `Index` search/range/iterator/distance and
`BruteForce` entry points currently reject deferred inputs with
the existing capability status `Status::not_implemented` (Milvus `Unsupported`)
before backend execution. Diagnostics identify the unsupported deferred input;
no new public status enum or corresponding Milvus-version dependency is needed.
Backend filter
preparation also rejects this representation instead of taking its old
zero-bitmap early return. A later consumer adaptation must remove these guards
only when all its count/raw-bitmap/worker/error paths preserve the predicate.
No Milvus capability probe, index policy whitelist or new global flag is
introduced here. Direct backend virtual methods are lower-level implementation
interfaces, not an alternate supported deferred search entry.

`test_candidate_evaluator.cc` belongs to the default unit target. It checks
transport/mapping/mandatory/active masks, worker lifetime, faults, concurrency
and the fail-closed rollout boundary using unchanged original Cohere train IDs.
The fixture records its source URL, source SHA256 and ordered IDs hash. These
protocol tests do not qualify vector-search recall or full fusing performance.
