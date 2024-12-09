if (WIN32)
set(NVTX_TOP_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
else (WIN32)
set(NVTX_TOP_DIR "/usr/local/cuda-9.1")
endif (WIN32)


FIND_PATH(NVTX_INCLUDE_DIR nvToolsExt.h
	${NVTX_TOP_DIR}/include
   ${NVTX_TOP_DIR}/include/nvtx3
)

# These two are not found - Remove them?
FIND_LIBRARY(NVTX_DEBUG_LIBRARIES NAMES nvToolsExt64_1 nvToolsExt
   PATHS
	 ${NVTX_TOP_DIR}/lib/x64
	 ${NVTX_TOP_DIR}/lib64
)

FIND_LIBRARY(NVTX_RELEASE_LIBRARIES NAMES nvToolsExt64_1 nvToolsExt
   PATHS
	 ${NVTX_TOP_DIR}/lib/x64
	 ${NVTX_TOP_DIR}/lib64
)
# End of removal

if(NVTX_INCLUDE_DIR)
   set(NVTX_FOUND TRUE)
endif(NVTX_INCLUDE_DIR)
	 
if(NVTX_FOUND)
   if(NOT NVTX_FIND_QUIETLY)
      message(STATUS "Found NVTX: ${NVTX_INCLUDE_DIR}")
   endif(NOT NVTX_FIND_QUIETLY)
else(NVTX_FOUND)
   if(NVTX_FIND_REQUIRED)
      message(FATAL_ERROR "could NOT find NVTX")
   endif(NVTX_FIND_REQUIRED)
endif(NVTX_FOUND)

MARK_AS_ADVANCED(NVTX_INCLUDE_DIR)


