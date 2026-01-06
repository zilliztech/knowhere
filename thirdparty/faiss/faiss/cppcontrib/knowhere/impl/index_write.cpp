/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/cppcontrib/knowhere/index_io.h>

#include <faiss/impl/io.h>

#include <cstdio>
#include <cstdlib>

#include <sys/stat.h>
#include <sys/types.h>

#include <faiss/cppcontrib/knowhere/invlists/InvertedListsIOHook.h>
#include <faiss/cppcontrib/knowhere/invlists/OnDiskInvertedLists.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io_macros.h>
#include <faiss/cppcontrib/knowhere/utils/hamming.h>

#include <faiss/cppcontrib/knowhere/IndexAdditiveQuantizer.h>
#include <faiss/cppcontrib/knowhere/IndexCosine.h>
#include <faiss/cppcontrib/knowhere/IndexFlat.h>
#include <faiss/cppcontrib/knowhere/IndexHNSW.h>
#include <faiss/cppcontrib/knowhere/IndexIVF.h>
#include <faiss/cppcontrib/knowhere/IndexIVFFlat.h>
#include <faiss/cppcontrib/knowhere/IndexIVFPQ.h>
#include <faiss/cppcontrib/knowhere/IndexIVFPQFastScan.h>
#include <faiss/cppcontrib/knowhere/IndexIVFRaBitQ.h>
#include <faiss/cppcontrib/knowhere/IndexPQ.h>
#include <faiss/cppcontrib/knowhere/IndexPreTransform.h>
#include <faiss/cppcontrib/knowhere/IndexRefine.h>
#include <faiss/cppcontrib/knowhere/IndexSQ4Uniform.h>
#include <faiss/cppcontrib/knowhere/IndexScaNN.h>
#include <faiss/cppcontrib/knowhere/IndexScalarQuantizer.h>
#include <faiss/VectorTransform.h>

#include <faiss/cppcontrib/knowhere/IndexBinaryFlat.h>
#include <faiss/cppcontrib/knowhere/IndexBinaryHNSW.h>
#include <faiss/cppcontrib/knowhere/IndexBinaryIVF.h>

/*************************************************************
 * The I/O format is the content of the class. For objects that are
 * inherited, like Index, a 4-character-code (fourcc) indicates which
 * child class this is an instance of.
 *
 * In this case, the fields of the parent class are written first,
 * then the ones for the child classes. Note that this requires
 * classes to be serialized to have a constructor without parameters,
 * so that the fields can be filled in later. The default constructor
 * should set reasonable defaults for all fields.
 *
 * The fourccs are assigned arbitrarily. When the class changed (added
 * or deprecated fields), the fourcc can be replaced. New code should
 * be able to read the old fourcc and fill in new classes.
 *
 * TODO: in this file, the read functions that encouter errors may
 * leak memory.
 **************************************************************/



namespace faiss::cppcontrib::knowhere {

/*************************************************************
 * Write
 **************************************************************/
static void write_index_header(const Index* idx, IOWriter* f) {
    WRITE1(idx->d);
    WRITE1(idx->ntotal);
    WRITE1(idx->is_cosine);

    uint8_t dummy8 = 0;
    WRITE1(dummy8);
    WRITE1(dummy8);
    WRITE1(dummy8);
    uint32_t dummy32 = 0;
    WRITE1(dummy32);
    idx_t dummy = 0;
    WRITE1(dummy);
    
    WRITE1(idx->is_trained);
    WRITE1(idx->metric_type);
    if (idx->metric_type > 1) {
        WRITE1(idx->metric_arg);
    }
}

void write_VectorTransform(const VectorTransform* vt, IOWriter* f) {
    if (const LinearTransform* lt = dynamic_cast<const LinearTransform*>(vt)) {
        if (dynamic_cast<const RandomRotationMatrix*>(lt)) {
            uint32_t h = fourcc("rrot");
            WRITE1(h);
        } else if (const PCAMatrix* pca = dynamic_cast<const PCAMatrix*>(lt)) {
            uint32_t h = fourcc("Pcam");
            WRITE1(h);
            WRITE1(pca->eigen_power);
            WRITE1(pca->epsilon);
            WRITE1(pca->random_rotation);
            WRITE1(pca->balanced_bins);
            WRITEVECTOR(pca->mean);
            WRITEVECTOR(pca->eigenvalues);
            WRITEVECTOR(pca->PCAMat);
        } else if (const ITQMatrix* itqm = dynamic_cast<const ITQMatrix*>(lt)) {
            uint32_t h = fourcc("Viqm");
            WRITE1(h);
            WRITE1(itqm->max_iter);
            WRITE1(itqm->seed);
        } else {
            // generic LinearTransform (includes OPQ)
            uint32_t h = fourcc("LTra");
            WRITE1(h);
        }
        WRITE1(lt->have_bias);
        WRITEVECTOR(lt->A);
        WRITEVECTOR(lt->b);
    } else if (
            const RemapDimensionsTransform* rdt =
                    dynamic_cast<const RemapDimensionsTransform*>(vt)) {
        uint32_t h = fourcc("RmDT");
        WRITE1(h);
        WRITEVECTOR(rdt->map);
    } else if (
            const NormalizationTransform* nt =
                    dynamic_cast<const NormalizationTransform*>(vt)) {
        uint32_t h = fourcc("VNrm");
        WRITE1(h);
        WRITE1(nt->norm);
    } else if (
            const CenteringTransform* ct =
                    dynamic_cast<const CenteringTransform*>(vt)) {
        uint32_t h = fourcc("VCnt");
        WRITE1(h);
        WRITEVECTOR(ct->mean);
    } else if (
            const ITQTransform* itqt = dynamic_cast<const ITQTransform*>(vt)) {
        uint32_t h = fourcc("Viqt");
        WRITE1(h);
        WRITEVECTOR(itqt->mean);
        WRITE1(itqt->do_pca);
        write_VectorTransform(&itqt->itq, f);
        write_VectorTransform(&itqt->pca_then_itq, f);
    } else {
        FAISS_THROW_MSG("cannot serialize this");
    }
    // common fields
    WRITE1(vt->d_in);
    WRITE1(vt->d_out);
    WRITE1(vt->is_trained);
}

void write_ProductQuantizer(const ProductQuantizer* pq, IOWriter* f) {
    WRITE1(pq->d);
    WRITE1(pq->M);
    WRITE1(pq->nbits);
    WRITEVECTOR(pq->centroids);
}

static void write_AdditiveQuantizer(const AdditiveQuantizer* aq, IOWriter* f) {
    WRITE1(aq->d);
    WRITE1(aq->M);
    WRITEVECTOR(aq->nbits);
    WRITE1(aq->is_trained);
    WRITEVECTOR(aq->codebooks);
    WRITE1(aq->search_type);
    WRITE1(aq->norm_min);
    WRITE1(aq->norm_max);
    if (aq->search_type == AdditiveQuantizer::ST_norm_cqint8 ||
        aq->search_type == AdditiveQuantizer::ST_norm_cqint4 ||
        aq->search_type == AdditiveQuantizer::ST_norm_lsq2x4 ||
        aq->search_type == AdditiveQuantizer::ST_norm_rq2x4) {
        WRITEXBVECTOR(aq->qnorm.codes);
    }

    if (aq->search_type == AdditiveQuantizer::ST_norm_lsq2x4 ||
        aq->search_type == AdditiveQuantizer::ST_norm_rq2x4) {
        WRITEVECTOR(aq->norm_tabs);
    }
}

static void write_ResidualQuantizer(const ResidualQuantizer* rq, IOWriter* f) {
    write_AdditiveQuantizer(rq, f);
    WRITE1(rq->train_type);
    WRITE1(rq->max_beam_size);
}

static void write_LocalSearchQuantizer(
        const LocalSearchQuantizer* lsq,
        IOWriter* f) {
    write_AdditiveQuantizer(lsq, f);
    WRITE1(lsq->K);
    WRITE1(lsq->train_iters);
    WRITE1(lsq->encode_ils_iters);
    WRITE1(lsq->train_ils_iters);
    WRITE1(lsq->icm_iters);
    WRITE1(lsq->p);
    WRITE1(lsq->lambd);
    WRITE1(lsq->chunk_size);
    WRITE1(lsq->random_seed);
    WRITE1(lsq->nperts);
    WRITE1(lsq->update_codebooks_with_double);
}

static void write_ProductAdditiveQuantizer(
        const ProductAdditiveQuantizer* paq,
        IOWriter* f) {
    write_AdditiveQuantizer(paq, f);
    WRITE1(paq->nsplits);
}

static void write_ProductResidualQuantizer(
        const ProductResidualQuantizer* prq,
        IOWriter* f) {
    write_ProductAdditiveQuantizer(prq, f);
    for (const auto aq : prq->quantizers) {
        auto rq = dynamic_cast<const ResidualQuantizer*>(aq);
        write_ResidualQuantizer(rq, f);
    }
}

static void write_ProductLocalSearchQuantizer(
        const ProductLocalSearchQuantizer* plsq,
        IOWriter* f) {
    write_ProductAdditiveQuantizer(plsq, f);
    for (const auto aq : plsq->quantizers) {
        auto lsq = dynamic_cast<const LocalSearchQuantizer*>(aq);
        write_LocalSearchQuantizer(lsq, f);
    }
}

static void write_ScalarQuantizer(const ScalarQuantizer* ivsc, IOWriter* f) {
    WRITE1(ivsc->qtype);
    WRITE1(ivsc->rangestat);
    WRITE1(ivsc->rangestat_arg);
    WRITE1(ivsc->d);
    WRITE1(ivsc->code_size);
    WRITEVECTOR(ivsc->trained);
}

void write_InvertedLists(const InvertedLists* ils, IOWriter* f) {
    if (ils == nullptr) {
        uint32_t h = fourcc("il00");
        WRITE1(h);
    } else if (
            const auto& ails = dynamic_cast<const ArrayInvertedLists*>(ils)) {
        uint32_t h = fourcc("ilar");
        WRITE1(h);
        WRITE1(ails->nlist);
        WRITE1(ails->code_size);
        // don't serialize 'with_norm'
        // WRITE1(ails->with_norm);
        // here we store either as a full or a sparse data buffer
        size_t n_non0 = 0;
        for (size_t i = 0; i < ails->nlist; i++) {
            if (ails->ids[i].size() > 0)
                n_non0++;
        }
        if (n_non0 > ails->nlist / 2) {
            uint32_t list_type = fourcc("full");
            WRITE1(list_type);
            std::vector<size_t> sizes;
            for (size_t i = 0; i < ails->nlist; i++) {
                sizes.push_back(ails->ids[i].size());
            }
            WRITEVECTOR(sizes);
        } else {
            int list_type = fourcc("sprs"); // sparse
            WRITE1(list_type);
            std::vector<size_t> sizes;
            for (size_t i = 0; i < ails->nlist; i++) {
                size_t n = ails->ids[i].size();
                if (n > 0) {
                    sizes.push_back(i);
                    sizes.push_back(n);
                }
            }
            WRITEVECTOR(sizes);
        }
        // make a single contiguous data buffer (useful for mmapping)
        for (size_t i = 0; i < ails->nlist; i++) {
            size_t n = ails->ids[i].size();
            if (n > 0) {
                WRITEANDCHECK(ails->codes[i].data(), n * ails->code_size);
                WRITEANDCHECK(ails->ids[i].data(), n);
                if (ails->with_norm) {
                    WRITEANDCHECK(ails->code_norms[i].data(), n);
                }
            }
        }
    } else if (const auto & lca =
                       dynamic_cast<const ConcurrentArrayInvertedLists *>(ils)) {
        uint32_t h = fourcc("ilca");
        WRITE1(h);
        WRITE1(lca->nlist);
        WRITE1(lca->code_size);
        WRITE1(lca->segment_size);
        // don't serialize 'save_norm'
        // WRITE1(lca->save_norm);

        // here we store either as a full or a sparse data buffer
        size_t n_non0 = 0;
        for (size_t i = 0; i < lca->nlist; i++) {
            if (lca->list_size(i) > 0) {
                n_non0++;
            }
        }
        if (n_non0 > lca->nlist / 2) {
            uint32_t list_type = fourcc("full");
            WRITE1(list_type);
            std::vector<size_t> sizes;
            for (size_t i = 0; i < lca->nlist; i++) {
                sizes.push_back(lca->list_size(i));
            }
            WRITEVECTOR(sizes);
        } else {
            int list_type = fourcc("sprs"); // sparse
            WRITE1(list_type);
            std::vector<size_t> sizes;
            for (size_t i = 0; i < lca->nlist; i++) {
                size_t n = lca->list_size(i);
                if (n > 0) {
                    sizes.push_back(i);
                    sizes.push_back(n);
                }
            }
            WRITEVECTOR(sizes);
        }
        // make a single contiguous data buffer (useful for mmapping)
        for (size_t i = 0; i < lca->nlist; i++) {
            size_t n = lca->list_size(i);
            if (n > 0) {
                size_t seg_num = lca->get_segment_num(i);
                for (size_t j = 0; j < seg_num; j++) {
                    size_t seg_size = lca->get_segment_size(i, j);
                    WRITEANDCHECK(lca->codes[i][j].data_.data(), seg_size * lca->code_size);
                    WRITEANDCHECK(lca->ids[i][j].data_.data(), seg_size);
                    if (lca->save_norm) {
                        WRITEANDCHECK(lca->code_norms[i][j].data_.data(), seg_size);
                    }
                }
            }
        }
    } else if (const auto & oa =
            dynamic_cast<const ReadOnlyArrayInvertedLists *>(ils)) {
        uint32_t h = fourcc("iloa");
        WRITE1(h);
        WRITE1(oa->nlist);
        WRITE1(oa->code_size);
        WRITEVECTOR(oa->readonly_length);
#ifdef USE_GPU
        size_t n = oa->pin_readonly_ids->size() / sizeof(InvertedLists::idx_t);
        WRITE1(n);
        WRITEANDCHECK((InvertedLists::idx_t*)oa->pin_readonly_ids->data, n);
        WRITEANDCHECK((uint8_t*)oa->pin_readonly_codes->data, n * oa->code_size);
#else
        size_t n = oa->readonly_ids.size();
        WRITE1(n);
        WRITEANDCHECK(oa->readonly_ids.data(), n);
        WRITEANDCHECK(oa->readonly_codes.data(), n * oa->code_size);
#endif
    } else if (const auto & od =
               dynamic_cast<const OnDiskInvertedLists *>(ils)) {
        uint32_t h = fourcc ("ilod");
        WRITE1(h);
        WRITE1(ils->nlist);
        WRITE1(ils->code_size);
        // this is a POD object
        WRITEVECTOR(od->lists);

        {
            std::vector<OnDiskInvertedLists::Slot> v(
                      od->slots.begin(), od->slots.end());
            WRITEVECTOR(v);
        }
        {
            std::vector<char> x(od->filename.begin(), od->filename.end());
            WRITEVECTOR(x);
        }
        WRITE1(od->totsize);
    } else {
        InvertedListsIOHook::lookup_classname(typeid(*ils).name())
                ->write(ils, f);
    }
}

void write_ProductQuantizer(const ProductQuantizer* pq, const char* fname) {
    FileIOWriter writer(fname);
    write_ProductQuantizer(pq, &writer);
}

static void write_HNSW(const HNSW* hnsw, IOWriter* f) {
    WRITEVECTOR(hnsw->assign_probas);
    WRITEVECTOR(hnsw->cum_nneighbor_per_level);
    WRITEVECTOR(hnsw->levels);
    WRITEVECTOR(hnsw->offsets);
    WRITEVECTOR(hnsw->neighbors);

    WRITE1(hnsw->entry_point);
    WRITE1(hnsw->max_level);
    WRITE1(hnsw->efConstruction);
    WRITE1(hnsw->efSearch);
    WRITE1(hnsw->upper_beam);
}

static void write_RaBitQuantizer(const RaBitQuantizer* rabitq, IOWriter* f) {
    // don't care about rabitq->centroid
    WRITE1(rabitq->d);
    WRITE1(rabitq->code_size);
    WRITE1(rabitq->metric_type);
}

static void write_direct_map(const DirectMap* dm, IOWriter* f) {
    char maintain_direct_map =
            (char)dm->type; // for backwards compatibility with bool
    WRITE1(maintain_direct_map);
    WRITEVECTOR(dm->array);
    if (dm->type == DirectMap::Hashtable) {
        std::vector<std::pair<idx_t, idx_t>> v;
        const std::unordered_map<idx_t, idx_t>& map = dm->hashtable;
        v.resize(map.size());
        std::copy(map.begin(), map.end(), v.begin());
        WRITEVECTOR(v);
    }
    if (dm->type == DirectMap::ConcurrentArray) {
        std::vector<idx_t> offs;
        offs.resize(dm->concurrentArray.size());
        for (size_t i = 0; i < dm->concurrentArray.size(); i++) {
            offs[i] = dm->concurrentArray[i];
        }
        WRITEVECTOR(offs);
    }
}

static void write_ivf_header(const IndexIVF* ivf, IOWriter* f) {
    write_index_header(ivf, f);
    WRITE1(ivf->nlist);
    WRITE1(ivf->nprobe);
    // subclasses write by_residual (some of them support only one setting of
    // by_residual).
    write_index(ivf->quantizer, f);
    write_direct_map(&ivf->direct_map, f);
}

void write_index(const Index* idx, IOWriter* f, int io_flags) {
    if (idx == nullptr) {
        // eg. for a storage component of HNSW that is set to nullptr
        uint32_t h = fourcc("null");
        WRITE1(h);
    } else if (const IndexFlatCosine* idxf = dynamic_cast<const IndexFlatCosine*>(idx)) {
        uint32_t h = fourcc("IxF9");
        WRITE1(h);
        write_index_header(idx, f);
        WRITEXBVECTOR(idxf->codes);
        // we're storing real l2 norms, because of
        //   backward compatibility issues. 
        WRITEVECTOR(idxf->inverse_norms_storage.as_l2_norms());
    } else if (const IndexFlat* idxf = dynamic_cast<const IndexFlat*>(idx)) {
        uint32_t h =
                fourcc(idxf->metric_type == METRIC_INNER_PRODUCT ? "IxFI"
                               : idxf->metric_type == METRIC_L2  ? "IxF2"
                                                                 : "IxFl");
        WRITE1(h);
        write_index_header(idx, f);
        WRITEXBVECTOR(idxf->codes);
        if (idx->is_cosine) {
            WRITEVECTOR(idxf->code_norms);
        }
    } else if (const IndexPQCosine* idxp = dynamic_cast<const IndexPQCosine*>(idx)) {
        uint32_t h = fourcc("IxP7");
        WRITE1(h);
        write_index_header(idx, f);
        write_ProductQuantizer(&idxp->pq, f);
        WRITEVECTOR(idxp->codes);
        // search params -- maybe not useful to store?
        WRITE1(idxp->search_type);
        WRITE1(idxp->encode_signs);
        WRITE1(idxp->polysemous_ht);
        // inverse norms
        WRITEVECTOR(idxp->inverse_norms_storage.inverse_l2_norms);
    } else if (const IndexPQ* idxp = dynamic_cast<const IndexPQ*>(idx)) {
        uint32_t h = fourcc("IxPq");
        WRITE1(h);
        write_index_header(idx, f);
        write_ProductQuantizer(&idxp->pq, f);
        WRITEVECTOR(idxp->codes);
        // search params -- maybe not useful to store?
        WRITE1(idxp->search_type);
        WRITE1(idxp->encode_signs);
        WRITE1(idxp->polysemous_ht);
    } else if (
            const IndexResidualQuantizer* idxr =
                    dynamic_cast<const IndexResidualQuantizer*>(idx)) {
        uint32_t h = fourcc("IxRq");
        WRITE1(h);
        write_index_header(idx, f);
        write_ResidualQuantizer(&idxr->rq, f);
        WRITE1(idxr->code_size);
        WRITEVECTOR(idxr->codes);
    } else if (
            auto* idxr_2 =
                    dynamic_cast<const IndexLocalSearchQuantizer*>(idx)) {
        uint32_t h = fourcc("IxLS");
        WRITE1(h);
        write_index_header(idx, f);
        write_LocalSearchQuantizer(&idxr_2->lsq, f);
        WRITE1(idxr_2->code_size);
        WRITEVECTOR(idxr_2->codes);
    } else if (
            const IndexProductResidualQuantizerCosine* idxpr =
                    dynamic_cast<const IndexProductResidualQuantizerCosine*>(idx)) {
        uint32_t h = fourcc("IxP5");
        WRITE1(h);
        write_index_header(idx, f);
        write_ProductResidualQuantizer(&idxpr->prq, f);
        WRITE1(idxpr->code_size);
        WRITEVECTOR(idxpr->codes);
        // inverse norms
        WRITEVECTOR(idxpr->inverse_norms_storage.inverse_l2_norms);
    } else if (
            const IndexProductResidualQuantizer* idxpr =
                    dynamic_cast<const IndexProductResidualQuantizer*>(idx)) {
        uint32_t h = fourcc("IxPR");
        WRITE1(h);
        write_index_header(idx, f);
        write_ProductResidualQuantizer(&idxpr->prq, f);
        WRITE1(idxpr->code_size);
        WRITEVECTOR(idxpr->codes);
    } else if (
            const IndexProductLocalSearchQuantizer* idxpl =
                    dynamic_cast<const IndexProductLocalSearchQuantizer*>(
                            idx)) {
        uint32_t h = fourcc("IxPL");
        WRITE1(h);
        write_index_header(idx, f);
        write_ProductLocalSearchQuantizer(&idxpl->plsq, f);
        WRITE1(idxpl->code_size);
        WRITEVECTOR(idxpl->codes);
    } else if (
            const ResidualCoarseQuantizer* idxr_2 =
                    dynamic_cast<const ResidualCoarseQuantizer*>(idx)) {
        uint32_t h = fourcc("ImRQ");
        WRITE1(h);
        write_index_header(idx, f);
        write_ResidualQuantizer(&idxr_2->rq, f);
        WRITE1(idxr_2->beam_factor);
    } else if (
            const IndexScalarQuantizer4bitUniformIP* idxs =
                    dynamic_cast<const IndexScalarQuantizer4bitUniformIP*>(
                            idx)) {
        // IndexScalarQuantizer4bitUniformIP: SQ4Uniform + IP
        // Must be checked BEFORE IndexScalarQuantizer4bitUniformCosine and
        // IndexScalarQuantizer (parent classes)
        uint32_t h = fourcc("IxSI");
        WRITE1(h);
        write_index_header(idx, f);
        write_ScalarQuantizer(&idxs->sq, f);
        WRITEVECTOR(idxs->codes);
        // Must serialize l2_norms_sqr for IP distance computation
        WRITEVECTOR(idxs->l2_norms_sqr);
    } else if (
            const IndexScalarQuantizer4bitUniformCosine* idxs =
                    dynamic_cast<const IndexScalarQuantizer4bitUniformCosine*>(
                            idx)) {
        // IndexScalarQuantizer4bitUniformCosine: SQ4Uniform + COSINE
        // Must be checked BEFORE IndexScalarQuantizerCosine (parent class)
        uint32_t h = fourcc("IxS4");
        WRITE1(h);
        write_index_header(idx, f);
        write_ScalarQuantizer(&idxs->sq, f);
        WRITEVECTOR(idxs->codes);
        // inverse norms (needed for refine to work correctly)
        WRITEVECTOR(idxs->inverse_l2_norms);
    } else if (
            const IndexScalarQuantizerCosine* idxs =
                    dynamic_cast<const IndexScalarQuantizerCosine*>(idx)) {
        uint32_t h = fourcc("IxS8");
        WRITE1(h);
        write_index_header(idx, f);
        write_ScalarQuantizer(&idxs->sq, f);
        WRITEVECTOR(idxs->codes);
        // inverse norms
        WRITEVECTOR(idxs->inverse_norms_storage.inverse_l2_norms);
    } else if (
            const IndexScalarQuantizer* idxs =
                    dynamic_cast<const IndexScalarQuantizer*>(idx)) {
        uint32_t h = fourcc("IxSQ");
        WRITE1(h);
        write_index_header(idx, f);
        write_ScalarQuantizer(&idxs->sq, f);
        WRITEVECTOR(idxs->codes);
    } else if (
            const IndexIVFFlatDedup* ivfl =
                    dynamic_cast<const IndexIVFFlatDedup*>(idx)) {
        uint32_t h = fourcc("IwFd");
        WRITE1(h);
        write_ivf_header(ivfl, f);
        {
            std::vector<idx_t> tab(2 * ivfl->instances.size());
            long i = 0;
            for (auto it = ivfl->instances.begin(); it != ivfl->instances.end();
                 ++it) {
                tab[i++] = it->first;
                tab[i++] = it->second;
            }
            WRITEVECTOR(tab);
        }
        write_InvertedLists(ivfl->invlists, f);
    } else if (
            const IndexIVFFlat* ivfl =
                    dynamic_cast<const IndexIVFFlatCC*>(idx)) {
        uint32_t h = fourcc("IwFc");
        WRITE1(h);
        write_ivf_header(ivfl, f);
        write_InvertedLists(ivfl->invlists, f);
    } else if (
            const IndexIVFFlat* ivfl_2 =
                    dynamic_cast<const IndexIVFFlat*>(idx)) {
        uint32_t h = fourcc("IwFl");
        WRITE1(h);
        write_ivf_header(ivfl_2, f);
        write_InvertedLists(ivfl_2->invlists, f);
    } else if (
            const IndexIVFScalarQuantizer* ivsc =
                    dynamic_cast<const IndexIVFScalarQuantizer*>(idx)) {
        uint32_t h = fourcc("IwSq");
        WRITE1(h);
        write_ivf_header(ivsc, f);
        write_ScalarQuantizer(&ivsc->sq, f);
        WRITE1(ivsc->code_size);
        WRITE1(ivsc->by_residual);
        write_InvertedLists(ivsc->invlists, f);
    } else if (const IndexIVFPQ* ivpq = dynamic_cast<const IndexIVFPQ*>(idx)) {
        uint32_t h = fourcc("IwPQ");
        WRITE1(h);
        write_ivf_header(ivpq, f);
        WRITE1(ivpq->by_residual);
        WRITE1(ivpq->code_size);
        write_ProductQuantizer(&ivpq->pq, f);
        write_InvertedLists(ivpq->invlists, f);
    } else if (
            const IndexPreTransform* ixpt =
                    dynamic_cast<const IndexPreTransform*>(idx)) {
        uint32_t h = fourcc("IxPT");
        WRITE1(h);
        write_index_header(ixpt, f);
        int nt = ixpt->chain.size();
        WRITE1(nt);
        for (int i = 0; i < nt; i++)
            write_VectorTransform(ixpt->chain[i], f);
        write_index(ixpt->index, f);
    } else if (
            const MultiIndexQuantizer* imiq =
                    dynamic_cast<const MultiIndexQuantizer*>(idx)) {
        uint32_t h = fourcc("Imiq");
        WRITE1(h);
        write_index_header(imiq, f);
        write_ProductQuantizer(&imiq->pq, f);
    } else if (
            const IndexScaNN* idxscann = dynamic_cast<const IndexScaNN*>(idx)) {
        uint32_t h = fourcc("IxSC");
        WRITE1(h);
        write_index_header(idxscann, f);
        write_index(idxscann->base_index, f);
        bool with_raw_data = idxscann->with_raw_data();
        WRITE1(with_raw_data);
        if (with_raw_data)
            write_index(idxscann->refine_index, f);
        WRITE1(idxscann->k_factor);
    } else if (
            const IndexRefine* idxrf = dynamic_cast<const IndexRefine*>(idx)) {
        uint32_t h = fourcc("IxRF");
        WRITE1(h);
        write_index_header(idxrf, f);
        write_index(idxrf->base_index, f);
        write_index(idxrf->refine_index, f);
        WRITE1(idxrf->k_factor);
    } else if (const IndexHNSW* idxhnsw = dynamic_cast<const IndexHNSW*>(idx)) {
        uint32_t h = dynamic_cast<const IndexHNSWFlat*>(idx)    ? fourcc("IHNf")
                : dynamic_cast<const IndexHNSWPQ*>(idx)         ? fourcc("IHNp")
                : dynamic_cast<const IndexHNSWSQ*>(idx)         ? fourcc("IHNs")
                : dynamic_cast<const IndexHNSW2Level*>(idx)     ? fourcc("IHN2")
                : dynamic_cast<const IndexHNSWCagra*>(idx)      ? fourcc("IHNc")
                : dynamic_cast<const IndexHNSWFlatCosine*>(idx) ? fourcc("IHN9")
                : dynamic_cast<const IndexHNSWSQCosine*>(idx)   ? fourcc("IHN8")
                : dynamic_cast<const IndexHNSWSQ4UniformCosine*>(idx)
                ? fourcc("IHNa")
                : dynamic_cast<const IndexHNSWSQ4UniformIP*>(idx)
                ? fourcc("IHNb")
                : dynamic_cast<const IndexHNSWPQCosine*>(idx) ? fourcc("IHN7")
                : dynamic_cast<const IndexHNSWProductResidualQuantizer*>(idx)
                ? fourcc("IHN6")
                : dynamic_cast<const IndexHNSWProductResidualQuantizerCosine*>(
                          idx)
                ? fourcc("IHN5")
                : 0;
        FAISS_THROW_IF_NOT(h != 0);
        WRITE1(h);
        write_index_header(idxhnsw, f);
        if (h == fourcc("IHNc")) {
            WRITE1(idxhnsw->keep_max_size_level0);
            auto idx_hnsw_cagra = dynamic_cast<const IndexHNSWCagra*>(idxhnsw);
            WRITE1(idx_hnsw_cagra->base_level_only);
            WRITE1(idx_hnsw_cagra->num_base_level_search_entrypoints);
        }
        write_HNSW(&idxhnsw->hnsw, f);
        if (io_flags & IO_FLAG_SKIP_STORAGE) {
            uint32_t n4 = fourcc("null");
            WRITE1(n4);
        } else {
            write_index(idxhnsw->storage, f);
        }
    } else if (
            const IndexIVFPQFastScan* ivpq_2 =
                    dynamic_cast<const IndexIVFPQFastScan*>(idx)) {
        uint32_t h = fourcc("IwPf");
        WRITE1(h);
        write_ivf_header(ivpq_2, f);
        WRITE1(ivpq_2->by_residual);
        WRITE1(ivpq_2->code_size);
        WRITE1(ivpq_2->bbs);
        WRITE1(ivpq_2->M2);
        WRITE1(ivpq_2->implem);
        WRITE1(ivpq_2->qbs2);
        WRITE1(ivpq_2->is_cosine);
        if (ivpq_2->is_cosine) {
            WRITEVECTOR(ivpq_2->inverse_norms);
        }
        write_ProductQuantizer(&ivpq_2->pq, f);
        write_InvertedLists(ivpq_2->invlists, f);
    } else if (
            const IndexIVFRaBitQ* ivrq =
                    dynamic_cast<const IndexIVFRaBitQ*>(idx)) {
        // using 'IwrQ' instead of baseline's 'Iwrq'
        uint32_t h = fourcc("IwrQ");
        WRITE1(h);
        write_ivf_header(ivrq, f);
        write_RaBitQuantizer(&ivrq->rabitq, f);
        WRITE1(ivrq->code_size);
        WRITE1(ivrq->by_residual);
        WRITE1(ivrq->qb);
        write_InvertedLists(ivrq->invlists, f);
    } else {
        FAISS_THROW_MSG("don't know how to serialize this type of index");
    }
}


void write_index(const Index* idx, FILE* f, int io_flags) {
    FileIOWriter writer(f);
    write_index(idx, &writer, io_flags);
}

void write_index(const Index* idx, const char* fname, int io_flags) {
    FileIOWriter writer(fname);
    write_index(idx, &writer, io_flags);
}

void write_value(uint32_t v, IOWriter* f) {
    WRITE1(v);
}

void write_vector(const std::vector<uint32_t>& v, IOWriter* f) {
    WRITEVECTOR(v);
}

// "IHMV" is a special header for faiss hnsw to indicate whether mv or not
void write_mv(IOWriter* f) {
    uint32_t h = fourcc("IHMV");
    WRITE1(h);
}

void write_VectorTransform(const VectorTransform* vt, const char* fname) {
    FileIOWriter writer(fname);
    write_VectorTransform(vt, &writer);
}

/*************************************************************
 * Write binary indexes
 **************************************************************/

static void write_index_binary_header(const IndexBinary* idx, IOWriter* f) {
    WRITE1(idx->d);
    WRITE1(idx->code_size);
    WRITE1(idx->ntotal);
    WRITE1(idx->is_trained);
    WRITE1(idx->metric_type);
}

static void write_binary_ivf_header(const IndexBinaryIVF* ivf, IOWriter* f) {
    write_index_binary_header(ivf, f);
    WRITE1(ivf->nlist);
    WRITE1(ivf->nprobe);
    write_index_binary(ivf->quantizer, f);
    write_direct_map(&ivf->direct_map, f);
}

void write_index_binary(const IndexBinary* idx, IOWriter* f) {
    if (const IndexBinaryFlat* idxf =
                dynamic_cast<const IndexBinaryFlat*>(idx)) {
        uint32_t h = fourcc("IBxF");
        WRITE1(h);
        write_index_binary_header(idx, f);
        WRITEVECTOR(idxf->xb);
    } else if (
            const IndexBinaryIVF* ivf =
                    dynamic_cast<const IndexBinaryIVF*>(idx)) {
        uint32_t h = fourcc("IBwF");
        WRITE1(h);
        write_binary_ivf_header(ivf, f);
        write_InvertedLists(ivf->invlists, f);
    } else if (
            const IndexBinaryHNSW* idxhnsw =
                    dynamic_cast<const IndexBinaryHNSW*>(idx)) {
        uint32_t h = fourcc("IBHf");
        WRITE1(h);
        write_index_binary_header(idxhnsw, f);
        write_HNSW(&idxhnsw->hnsw, f);
        write_index_binary(idxhnsw->storage, f);
    } else {
        FAISS_THROW_MSG("don't know how to serialize this type of index");
    }
}

void write_index_binary(const IndexBinary* idx, FILE* f) {
    FileIOWriter writer(f);
    write_index_binary(idx, &writer);
}

void write_index_binary(const IndexBinary* idx, const char* fname) {
    FileIOWriter writer(fname);
    write_index_binary(idx, &writer);
}

}


