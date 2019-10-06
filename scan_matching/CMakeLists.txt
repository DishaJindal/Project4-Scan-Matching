set(SOURCE_FILES
    "common.h"
    "common.cu"
    "cpu.h"
    "cpu.cu"
	"svd.h"
    )

cuda_add_library(
    scan_matching
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )
