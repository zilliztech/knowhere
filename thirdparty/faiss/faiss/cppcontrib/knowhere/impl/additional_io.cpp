// #include <faiss/cppcontrib/knowhere/impl/additional_io.h>

// #include <faiss/impl/io_macros.h>
// #include <faiss/impl/io.h>

// namespace faiss {
// namespace cppcontrib {
// namespace knowhere {

// // "IHMV" is a special header for faiss hnsw to indicate whether mv or not
// bool read_is_mv(IOReader* f) {
//     uint32_t h;
//     READ1(h);
//     return h == fourcc("IHMV");
// }

// bool read_is_mv(const char* fname) {
//     FileIOReader f(fname);
//     return read_is_mv(&f);
// }

// void read_vector(std::vector<uint32_t>& v, IOReader* f) {
//     READVECTOR(v);
// }

// void write_vector(const std::vector<uint32_t>& v, IOWriter* f) {
//     WRITEVECTOR(v);
// }

// uint32_t read_value(IOReader* f) {
//     uint32_t h;
//     READ1(h)
//     return h;
// }

// void write_value(uint32_t v, IOWriter* f) {
//     WRITE1(v);
// }

// // "IHMV" is a special header for faiss hnsw to indicate whether mv or not
// void write_mv(IOWriter* f) {
//     uint32_t h = fourcc("IHMV");
//     WRITE1(h);
// }

// }
// }
// }
