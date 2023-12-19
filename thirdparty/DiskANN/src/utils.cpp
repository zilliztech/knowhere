#include "diskann/utils.h"
#include <stdio.h>
#include "knowhere/comp/task.h"
namespace diskann {
  void block_convert(std::ofstream& writr, std::ifstream& readr,
                     float* read_buf, _u64 npts, _u64 ndims) {
    readr.read((char*) read_buf, npts * ndims * sizeof(float));
    _u32                               ndims_u32 = (_u32) ndims;
    std::vector<std::function<void()>> tasks;
    for (_s64 i = 0; i < (_s64) npts; i++) {
      tasks.emplace_back([&, index = i]() {
        float norm_pt = std::numeric_limits<float>::epsilon();
        for (_u32 dim = 0; dim < ndims_u32; dim++) {
          norm_pt += *(read_buf + index * ndims + dim) *
                     *(read_buf + index * ndims + dim);
        }
        norm_pt = std::sqrt(norm_pt);
        for (_u32 dim = 0; dim < ndims_u32; dim++) {
          *(read_buf + index * ndims + dim) =
              *(read_buf + index * ndims + dim) / norm_pt;
        }
      });
    }
    knowhere::ExecOverBuildThreadPool(tasks);
    writr.write((char*) read_buf, npts * ndims * sizeof(float));
  }

  void normalize_data_file(const std::string& inFileName,
                           const std::string& outFileName) {
    std::ifstream readr(inFileName, std::ios::binary);
    std::ofstream writr(outFileName, std::ios::binary);

    int npts_s32, ndims_s32;
    readr.read((char*) &npts_s32, sizeof(_s32));
    readr.read((char*) &ndims_s32, sizeof(_s32));

    writr.write((char*) &npts_s32, sizeof(_s32));
    writr.write((char*) &ndims_s32, sizeof(_s32));

    _u64 npts = (_u64) npts_s32, ndims = (_u64) ndims_s32;
    LOG_KNOWHERE_DEBUG_ << "Normalizing FLOAT vectors in file: " << inFileName
                        << "Dataset: #pts = " << npts << ", # dims = " << ndims;

    _u64 blk_size = 131072;
    _u64 nblks = ROUND_UP(npts, blk_size) / blk_size;
    LOG_KNOWHERE_DEBUG_ << "# blks: " << nblks;

    float* read_buf = new float[npts * ndims];
    for (_u64 i = 0; i < nblks; i++) {
      _u64 cblk_size = std::min(npts - i * blk_size, blk_size);
      block_convert(writr, readr, read_buf, cblk_size, ndims);
    }
    delete[] read_buf;

    LOG_KNOWHERE_DEBUG_ << "Wrote normalized points to file: " << outFileName;
  }
}  // namespace diskann
