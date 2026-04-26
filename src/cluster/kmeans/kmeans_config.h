// Copyright (c) KIOXIA Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef SRC_CLUSTER_KMEANS_KMEANS_CONFIG_H_
#define SRC_CLUSTER_KMEANS_KMEANS_CONFIG_H_

#include "knowhere/config.h"

namespace knowhere {

class KmeansConfig : public BaseConfig {
 public:
    CFG_INT num_clusters;
    CFG_INT num_iter;
    KNOWHERE_DECLARE_CONFIG(KmeansConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(num_clusters)
            .description("the number of clusters")
            .set_default(48)
            .set_range(1, 1024 * 1024)
            .for_cluster();
        KNOWHERE_CONFIG_DECLARE_FIELD(num_iter)
            .description("The number training iterations")
            .set_default(12)
            .set_range(1, 50)
            .for_cluster();
    }
};

}  // namespace knowhere

#endif /* SRC_CLUSTER_KMEANS_KMEANS_CONFIG_H_ */
