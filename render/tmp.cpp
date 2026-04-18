#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

int main() {
    // host 数据
    std::vector<int> h_keys   = {8, 6, 7, 5, 3, 0, 9};
    std::vector<int> h_values = {0, 1, 2, 3, 4, 5, 6};
    std::vector<int> h_offsets = {0, 3, 3, 7}; // 3 个 segment

    const int num_items = static_cast<int>(h_keys.size());
    const int num_segments = static_cast<int>(h_offsets.size()) - 1;

    int *d_keys_in = nullptr, *d_keys_out = nullptr;
    int *d_values_in = nullptr, *d_values_out = nullptr;
    int *d_offsets = nullptr;

    cudaMalloc(&d_keys_in,    num_items * sizeof(int));
    cudaMalloc(&d_keys_out,   num_items * sizeof(int));
    cudaMalloc(&d_values_in,  num_items * sizeof(int));
    cudaMalloc(&d_values_out, num_items * sizeof(int));
    cudaMalloc(&d_offsets,    (num_segments + 1) * sizeof(int));

    cudaMemcpy(d_keys_in,   h_keys.data(),   num_items * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_in, h_values.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets,   h_offsets.data(), (num_segments + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // 1) 查询临时空间大小
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceSegmentedRadixSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in,
        d_keys_out,
        d_values_in,
        d_values_out,
        num_items,
        num_segments,
        d_offsets,       // begin offsets
        d_offsets + 1    // end offsets
    );

    // 2) 分配临时空间
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // 3) 正式排序
    cub::DeviceSegmentedRadixSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in,
        d_keys_out,
        d_values_in,
        d_values_out,
        num_items,
        num_segments,
        d_offsets,
        d_offsets + 1
    );

    // 拷回结果
    std::vector<int> h_keys_out(num_items), h_values_out(num_items);
    cudaMemcpy(h_keys_out.data(),   d_keys_out,   num_items * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values_out.data(), d_values_out, num_items * sizeof(int), cudaMemcpyDeviceToHost);

    for (int x : h_keys_out) std::cout << x << " ";
    std::cout << "\n";
    for (int x : h_values_out) std::cout << x << " ";
    std::cout << "\n";

    cudaFree(d_temp_storage);
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_values_in);
    cudaFree(d_values_out);
    cudaFree(d_offsets);

    return 0;
}