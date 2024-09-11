#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <random>

#include "simd/distances_ref.h"
#include "simd/hook.h"
TEST_CASE("Test Distance Compute", "[distance]") {
    std::mt19937 rng;
    std::uniform_int_distribution<> distrib(1, 100000);
    std::uniform_real_distribution<float> fill_distrib(1, 1000000);
    std::string ins;
    faiss::fvec_hook(ins);

    using std::make_tuple;
    SECTION("Test Binary Distance Compute") {
        typedef float (*FUNC)(const float*, const float*, size_t);
        auto [real_func, gold_func] = GENERATE(table<FUNC, FUNC>({
            make_tuple(faiss::fvec_L1, faiss::fvec_L1_ref),
            make_tuple(faiss::fvec_L2sqr, faiss::fvec_L2sqr_ref),
            make_tuple(faiss::fvec_Linf, faiss::fvec_Linf_ref),
            make_tuple(faiss::fvec_inner_product, faiss::fvec_inner_product_ref),
        }));

        for (int i = 0; i < 1000; ++i) {
            CAPTURE(i);
            auto len = distrib(rng);
            std::vector<float> a(len);
            std::vector<float> b(len);
            for (int i = 0; i < len; ++i) {
                a[i] = fill_distrib(rng);
                b[i] = fill_distrib(rng);
            }
            REQUIRE_THAT(real_func(a.data(), b.data(), len),
                         Catch::Matchers::WithinRel(gold_func(a.data(), b.data(), len), 0.001f));
        }
    }

    SECTION("Test Normal Compute") {
        typedef float (*FUNC)(const float*, size_t);
        auto [real_func, gold_func] = GENERATE(table<FUNC, FUNC>({
            make_tuple(faiss::fvec_norm_L2sqr, faiss::fvec_norm_L2sqr_ref),
        }));

        for (int i = 0; i < 1000; ++i) {
            CAPTURE(i);
            auto len = distrib(rng);
            std::vector<float> a(len);
            for (int i = 0; i < len; ++i) {
                a[i] = fill_distrib(rng);
            }
            REQUIRE_THAT(real_func(a.data(), len), Catch::Matchers::WithinRel(gold_func(a.data(), len), 0.001f));
        }
    }

    SECTION("Test Madd and Argmin") {
        typedef int (*FUNC)(size_t, const float*, float, const float*, float*);
        auto [real_func, gold_func] = GENERATE(table<FUNC, FUNC>({
            make_tuple(faiss::fvec_madd_and_argmin, faiss::fvec_madd_and_argmin_ref),
        }));

        for (int i = 0; i < 1000; ++i) {
            CAPTURE(i);
            auto len = distrib(rng);
            std::vector<float> a(len);
            std::vector<float> b(len);
            for (int i = 0; i < len; ++i) {
                a[i] = fill_distrib(rng);
                b[i] = fill_distrib(rng);
            }

            std::vector<float> c(len);
            std::vector<float> c_gold(len);
            float pf = fill_distrib(rng);
            REQUIRE(real_func(len, a.data(), pf, b.data(), c.data()) ==
                    gold_func(len, a.data(), pf, b.data(), c_gold.data()));

            for (int i = 0; i < len; ++i) {
                REQUIRE_THAT(c[i], Catch::Matchers::WithinRel(c_gold[i], 0.001f));
            }
        }
    }

    SECTION("Test Madd") {
        typedef void (*FUNC)(size_t, const float*, float, const float*, float*);
        auto [real_func, gold_func] = GENERATE(table<FUNC, FUNC>({
            make_tuple(faiss::fvec_madd, faiss::fvec_madd_ref),
        }));

        for (int i = 0; i < 1000; ++i) {
            CAPTURE(i);
            auto len = distrib(rng);
            std::vector<float> a(len);
            std::vector<float> b(len);
            for (int i = 0; i < len; ++i) {
                a[i] = fill_distrib(rng);
                b[i] = fill_distrib(rng);
            }

            std::vector<float> c(len);
            std::vector<float> c_gold(len);
            float pf = fill_distrib(rng);
            real_func(len, a.data(), pf, b.data(), c.data());
            gold_func(len, a.data(), pf, b.data(), c_gold.data());

            for (int i = 0; i < len; ++i) {
                REQUIRE_THAT(c[i], Catch::Matchers::WithinRel(c_gold[i], 0.001f));
            }
        }
    }
}
