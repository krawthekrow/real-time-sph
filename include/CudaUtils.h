#include <iostream>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

using namespace std;

class CudaUtils {
public:
    template <typename T>
    static void DebugPrint(T *arr, const int num);
};

template <typename T>
void CudaUtils::DebugPrint(T *arr, const int num) {
    T *debugArr = (T *)malloc(num * sizeof(T));
    cudaMemcpy(debugArr, arr, num * sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < num; i++)
        cout << debugArr[i] << " ";
    cout << endl;
    free(debugArr);
}
