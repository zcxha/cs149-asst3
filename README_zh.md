# 作业 3：一个简单的 CUDA 渲染器

**截止时间：10 月 30 日（周四）太平洋时间 23:59**

**总分：100 分**

![My Image](handout/teaser.jpg?raw=true)

## 概述

在这次作业中，你将使用 CUDA 编写一个并行渲染器，用来绘制彩色圆。
虽然这个渲染器本身非常简单，但要把它并行化，你需要设计并实现能够被高效并行构建和操作的数据结构。
这是一份有挑战性的作业，所以建议你尽早开始。**认真地说，请一定尽早开始。** 祝你好运！

## 环境配置

1. 你需要在 Amazon Web Services (AWS) 提供的、带 GPU 的虚拟机上收集本次作业的结果（也就是运行性能测试）。请按照 [cloud_readme.md](cloud_readme.md) 中的说明来配置运行本作业所需的机器。

2. 使用下面的命令从课程 Github 下载本次作业的起始代码：

`git clone https://github.com/stanford-cs149/asst3`

CUDA C Programmer's Guide 的 [PDF 版本](http://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) 或 [网页版本](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 是学习 CUDA 编程的极佳参考资料。网上也有大量 CUDA 教程和 SDK 示例（直接搜就行），NVIDIA 开发者网站上也有很多资源：[NVIDIA developer site](http://docs.nvidia.com/cuda/)。另外，你可能也会喜欢 Udacity 的免费课程 [Introduction to Parallel Programming in CUDA](https://www.udacity.com/blog/2014/01/update-on-udacity-cs344-intro-to.html)。

[CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities) 中的表 21 对于本次作业所使用的 NVIDIA T4 GPU 非常有参考价值，里面列出了每个 thread block 的最大线程数、thread block 大小、shared memory 大小等信息。NVIDIA T4 GPU 支持 CUDA compute capability 7.5。

如果你有 C++ 相关问题（比如 `_virtual_` 关键字是什么意思），[C++ Super-FAQ](https://isocpp.org/faq) 是个非常好的资源，解释详细而且容易理解（比很多 C++ 资料都更友好），它还是由 C++ 的创造者 Bjarne Stroustrup 共同编写的。

## 第 1 部分：CUDA 热身 1：SAXPY（5 分）

为了先熟悉一下 CUDA 程序的编写，你的热身任务是在 CUDA 中重新实现作业 1 里的 SAXPY 函数。这部分的起始代码位于作业仓库中的 `/saxpy` 目录。你可以在 `/saxpy` 目录下运行 `make` 和 `./cudaSaxpy` 来编译并执行这个 CUDA 版本的 saxpy 程序。

请补全 `saxpy.cu` 中 `saxpyCuda` 函数里的 SAXPY 实现。你需要先在设备端分配全局内存数组，并把主机端输入数组 `X`、`Y` 和 `result` 的内容拷贝到 CUDA 设备内存中，然后再执行计算。CUDA 计算完成后，还需要把结果拷回主机内存。请参阅 Programmer's Guide（网页版本）第 3.2.2 节中 `cudaMemcpy` 函数的定义，或者查看起始代码里给出的教程链接。

作为实现的一部分，请在 `saxpyCuda` 中为 CUDA kernel 调用添加计时器。完成后，你的程序应该统计两种执行时间：

- 起始代码中已经提供了计时器，用于测量**完整过程**：包括把数据拷到 GPU、运行 kernel，以及把数据拷回 CPU。

- 你还需要加入额外的计时器，只测量 _kernel 本身的执行时间_。（不应包含 CPU 到 GPU 的数据传输时间，也不应包含从 GPU 把结果传回 CPU 的时间。）

**在后一种情况中添加计时代码时，你需要格外小心：** 默认情况下，CUDA kernel 在 GPU 上的执行相对于 CPU 上运行的主应用线程来说是 _异步_ 的。比如，如果你写出如下代码：

```cpp
double startTime = CycleTimer::currentSeconds();
saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
double endTime = CycleTimer::currentSeconds();
```

你测出来的 kernel 执行时间会看起来快得惊人！（因为你实际上只测到了 API 调用本身的开销，而不是 GPU 上真正执行计算的时间。）

因此，你需要在 kernel 调用之后加上 `cudaDeviceSynchronize()`，以等待 GPU 上所有 CUDA 工作完成。这个 `cudaDeviceSynchronize()` 会在此前提交到 GPU 的所有 CUDA 工作都完成后才返回。注意，在 `cudaMemcpy()` 之后不需要再额外调用 `cudaDeviceSynchronize()` 来确保传输到 GPU 的内存拷贝已经完成，因为在我们这里的使用条件下，`cudaMemcpy()` 本身就是同步的。（如果你想深入了解，可以参考[这份文档](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync)。）

```cpp
double startTime = CycleTimer::currentSeconds();
saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
cudaDeviceSynchronize();
double endTime = CycleTimer::currentSeconds();
```

注意，对于包含 CPU 与 GPU 间数据传输时间的那组测量，在最终计时点之前（也就是把数据从 GPU 拷回 CPU 的 `cudaMemcpy()` 调用之后）**不需要** 调用 `cudaDeviceSynchronize()`，因为 `cudaMemcpy()` 会在拷贝完成后才返回给调用线程。

**问题 1：** 与基于顺序 CPU 的 SAXPY 实现相比，你观察到了什么样的性能表现？（回忆一下你在作业 1 的 Program 5 中关于 saxpy 的结果。）

**问题 2：** 比较并解释两组计时结果之间的差异（只计 kernel 执行时间 vs. 连同数据搬运到 GPU、再搬回来的完整过程时间）。你观察到的带宽值是否与机器不同组件的标称带宽 _大致一致_？（你需要自行上网查询 NVIDIA T4 GPU 的内存带宽。提示：<https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf>。AWS 上内存总线的期望带宽为 5.3 GB/s，这与 16 通道 [PCIe 3.0](https://en.wikipedia.org/wiki/PCI_Express) 的数值并不一致。很多因素会使峰值带宽达不到理论值，包括 CPU 主板芯片组性能，以及作为传输源的主机内存是否是 “pinned” 的。后者允许 GPU 直接访问内存，而无需经过虚拟内存地址转换。如果你感兴趣，可以看这里：<https://kth.instructure.com/courses/12406/pages/optimizing-host-device-data-communication-i-pinned-host-memory>）

## 第 2 部分：CUDA 热身 2：并行前缀和（10 分）

现在你已经熟悉了 CUDA 程序的基本结构和布局，第二个练习要求你为函数 `find_repeats` 设计一个并行实现：给定一个整数数组 `A`，返回所有满足 `A[i] == A[i+1]` 的下标 `i` 组成的列表。

例如，给定数组 `{1,2,2,1,1,1,3,5,3,3}`，程序应该输出数组 `{1,3,4,8}`。

#### Exclusive Prefix Sum

我们希望你先实现并行的 exclusive prefix-sum（独占前缀和）操作，再用它来实现 `find_repeats`。

exclusive prefix sum 接收一个数组 `A`，生成新数组 `output`，其中每个位置 `i` 上的值是 `A[i]` 之前所有元素之和，但 **不包括** `A[i]` 本身。例如，给定数组 `A={1,4,6,8,2}`，则 exclusive prefix sum 的输出为 `output={0,1,5,11,19}`。

下面是一段“类 C”风格的 scan 迭代版代码。伪代码里使用 `parallel_for` 表示潜在可并行的循环。这就是我们课堂上讲过的算法：<https://gfxcourses.stanford.edu/cs149/fall25/lecture/dataparallel/slide_17>

```cpp
void exclusive_scan_iterative(int* start, int* end, int* output) {

    int N = end - start;
    memmove(output, start, N*sizeof(int));

    // upsweep phase
    for (int two_d = 1; two_d <= N/2; two_d*=2) {
        int two_dplus1 = 2*two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            output[i+two_dplus1-1] += output[i+two_d-1];
        }
    }

    output[N-1] = 0;

    // downsweep phase
    for (int two_d = N/2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2*two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            int t = output[i+two_d-1];
            output[i+two_d-1] = output[i+two_dplus1-1];
            output[i+two_dplus1-1] += t;
        }
    }
}
```

我们希望你使用这个算法，在 CUDA 中实现一个并行前缀和版本。你必须在 `scan/scan.cu` 中实现 `exclusive_scan` 函数。你的实现会同时包含主机端和设备端代码，并且需要多次启动 CUDA kernel（对应伪代码中每一个 `parallel_for` 循环）。

**注意：** 在起始代码中，上面的参考 scan 实现假设输入数组长度（`N`）是 2 的幂。在 `cudaScan` 函数里，我们通过在 GPU 上分配缓冲区时，把输入数组长度向上取整到下一个 2 的幂来解决这个问题。不过，代码只会把 GPU 缓冲区中的前 `N` 个元素拷回 CPU 缓冲区。这一点应该能简化你的 CUDA 实现。

编译后会生成可执行文件 `cudaScan`。命令行用法如下：

```text
Usage: ./cudaScan [options]

Program Options:
  -m  --test <TYPE>      Run specified function on input.  Valid tests are: scan, find_repeats (default: scan)
  -i  --input <NAME>     Run test on given input type. Valid inputs are: ones, random (default: random)
  -n  --arraysize <INT>  Number of elements in arrays
  -t  --thrust           Use Thrust library implementation
  -?  --help             This message
```

#### 使用 Prefix Sum 实现 “Find Repeats”

完成 `exclusive_scan` 之后，请在 `scan/scan.cu` 中实现函数 `find_repeats`。这将需要你编写更多设备端代码，并调用一次或多次 `exclusive_scan()`。你的代码应当把重复元素对应的下标列表写入给定的输出指针（位于设备内存中），并返回输出列表的大小。

调用你实现的 `exclusive_scan` 时，请记住：`start` 数组的内容会被复制到 `output` 数组中。另外，传给 `exclusive_scan` 的数组默认都位于 `device` 内存中。

**评分方式：** 我们会在随机输入数组上测试你代码的正确性和性能。

作为参考，下面给出一个 scan 分数表，展示了一个简单 CUDA 实现在 K80 GPU 上的性能。要检查你的 `scan` 和 `find_repeats` 实现的正确性与性能得分，请分别运行 **`./checker.py scan`** 和 **`./checker.py find_repeats`**。运行后会生成类似下表的参考表格；你的得分完全基于代码性能。要拿到满分，你的代码性能必须达到给定参考解的 20% 以内。

```text
-------------------------
Scan Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Ref Time        | Student Time    | Score           |
-------------------------------------------------------------------------
| 1000000         | 0.766           | 0.143 (F)       | 0               |
| 10000000        | 8.876           | 0.165 (F)       | 0               |
| 20000000        | 17.537          | 0.157 (F)       | 0               |
| 40000000        | 34.754          | 0.139 (F)       | 0               |
-------------------------------------------------------------------------
|                                   | Total score:    | 0/5             |
-------------------------------------------------------------------------
```

这一部分主要是让你继续练习 CUDA 编程，并学习如何用数据并行的方式思考问题，而不是去做复杂的性能调优。要拿到这一部分的满性能分，通常不需要太多（甚至几乎不需要）性能优化，只需要把伪代码中的算法直接移植到 CUDA 即可。不过有一个技巧：一种非常朴素的 scan 实现，可能会在伪代码中每次并行循环迭代时都启动 `N` 个 CUDA 线程，然后在 kernel 中通过条件判断来决定哪些线程真正执行工作。这样的方案性能会很差！（想想 upsweep 阶段最外层最后一次循环时，实际上只有两个线程需要工作！）满分解法应当只为最内层并行循环中的每次迭代启动一个 CUDA 线程。

**测试框架：** 默认情况下，测试框架运行在一个伪随机生成、但每次运行都相同的数组上，以便调试。你可以传入参数 `-i random` 来在真正随机的数组上运行，我们评分时会这样做。我们也鼓励你自己构造其他输入来测试程序。你还可以使用 `-n <size>` 来改变输入数组长度。

参数 `--thrust` 会改用 [Thrust Library](http://thrust.github.io/) 中的 [exclusive scan](https://docs.nvidia.com/cuda/archive/12.2.2/thrust/index.html?highlight=group%20prefix%20sums#prefix-sums) 实现。**如果有人能做出一个与 Thrust 竞争的实现，最多可获得 2 分额外加分。**

## 第 3 部分：一个简单的圆形渲染器（85 分）

现在进入重头戏！

作业起始代码中的 `/render` 目录包含了一个绘制圆形的渲染器实现。请先编译代码，然后用下面的命令运行：

`./render -r cpuref rgb`

程序会输出一张名为 `output_0000.ppm` 的图像，其中包含三个圆。接着再运行：

`./render -r cpuref snow`

此时输出将是一幅下雪的画面。PPM 图像在 OSX 上可以直接用 Preview 查看；如果你使用 Windows，可能需要下载一个查看器。

注意：你也可以使用 `-i` 选项，把渲染输出直接显示出来，而不是写到文件里。（在 snow 场景下，你会看到一个下雪动画。）不过要使用交互模式，你需要能把 X-windows 转发到本地机器上。（[这个参考](http://atechyblog.blogspot.com/2014/12/google-cloud-compute-x11-forwarding.html) 或 [这个参考](https://stackoverflow.com/questions/25521486/x11-forwarding-from-debian-on-google-compute-engine) 可能会有帮助。）

作业起始代码中包含两个版本的渲染器：一个是顺序、单线程的 C++ 参考实现，位于 `refRenderer.cpp`；另一个是 _错误的_ 并行 CUDA 实现，位于 `cudaRenderer.cu`。

### 渲染器概述

我们建议你先通过阅读 `refRenderer.cpp` 来熟悉渲染器代码的整体结构。`setup` 方法会在渲染第一帧之前调用。在你的 CUDA 加速渲染器中，这个方法很可能会包含所有初始化代码（比如分配缓冲区等）。`render` 会在每一帧调用，负责把所有圆绘制到输出图像中。渲染器的另一个主要函数 `advanceAnimation` 也会在每一帧调用一次，它负责更新圆的位置和速度。在这次作业中，你不需要修改 `advanceAnimation`。

渲染器接收一个圆数组作为输入，每个圆包含 3D 位置、速度、半径和颜色。每一帧的基本顺序算法如下：

```text
Clear image
for each circle
    update position and velocity
for each circle
    compute screen bounding box
    for all pixels in bounding box
        compute pixel center point
        if center point is within the circle
            compute color of circle at point
            blend contribution of circle into image for this pixel
```

下图展示了基于点是否在圆内（point-in-circle）的圆与像素覆盖关系计算方法。注意，只有当像素中心落在圆内时，该圆才会对输出像素产生颜色贡献。

![Point in circle test](handout/point_in_circle.jpg?raw=true "A simple algorithm for computing the contribution of a circle to the output image: All pixels within the circle's bounding box are tested for coverage. For each pixel in the bounding box, the pixel is considered to be covered by the circle if its center point (black dots) is contained within the circle. Pixel centers that are inside the circle are colored red. The circle's contribution to the image will be computed only for covered pixels.")

这个渲染器的一个重要细节是：它渲染的是**半透明**圆。因此，任意一个像素的颜色并不是某一个圆的颜色，而是所有覆盖这个像素的半透明圆共同混合后的结果（注意上面伪代码中的 “blend contribution”）。渲染器用四元组红（R）、绿（G）、蓝（B）和不透明度（alpha）来表示圆的颜色，也就是 RGBA。Alpha = 1 表示完全不透明；Alpha = 0 表示完全透明。若要把一个颜色为 `(C_r, C_g, C_b, C_alpha)` 的半透明圆绘制到颜色为 `(P_r, P_g, P_b)` 的像素之上，渲染器使用如下公式：

<pre>
   result_r = C_alpha * C_r + (1.0 - C_alpha) * P_r
   result_g = C_alpha * C_g + (1.0 - C_alpha) * P_g
   result_b = C_alpha * C_b + (1.0 - C_alpha) * P_b
</pre>

注意，这种合成运算不是交换律成立的（对象 X 覆盖在 Y 上，与 Y 覆盖在 X 上，看起来并不相同），所以渲染器必须按照应用程序提供圆的顺序来绘制。（你可以假设应用程序给出的圆已经按深度顺序排好。）例如，下图中蓝色圆绘制在绿色圆之上，而绿色圆绘制在红色圆之上。左图按正确顺序绘制；右图使用了不同顺序，因此输出结果不正确。

![Ordering](handout/order.jpg?raw=true "The renderer must be careful to generate output that is the same as what is generated when sequentially drawing all circles in the order provided by the application.")

### CUDA 渲染器

在熟悉参考实现中的圆形渲染算法之后，请继续研究 `cudaRenderer.cu` 中提供的 CUDA 渲染器实现。你可以通过 `--renderer cuda`（或 `-r cuda`）这个命令行选项来运行 CUDA 版本。

给定的 CUDA 实现把并行性分配到所有输入圆上，也就是每个 CUDA 线程负责一个圆。虽然这个 CUDA 版本完整实现了圆形渲染器的数学逻辑，但它包含几个重大错误，这正是本次作业要你修复的内容。具体来说：当前实现既不能保证图像更新操作的原子性，也不能保证图像更新顺序满足要求（下面会介绍顺序要求）。

### 渲染器要求

你的并行 CUDA 渲染器实现必须保持顺序实现天然满足的两个不变量：

1. **原子性（Atomicity）：** 所有图像更新操作都必须是原子的。临界区包括读取像素的四个 32 位浮点 rgba 值、把当前圆的贡献与当前像素值进行混合，然后把新像素颜色写回内存。
2. **顺序（Order）：** 你的渲染器必须按照 _圆输入顺序_ 更新图像像素。也就是说，如果圆 1 和圆 2 都会对像素 `P` 产生贡献，那么由圆 1 导致的对 `P` 的更新必须先于圆 2 导致的更新。正如上文所述，保持这一顺序要求是透明圆正确渲染的关键。（它对图形系统还有其他一些好处。如果你好奇，可以去问 Kayvon。）**一个关键观察是：顺序的定义只约束对同一个像素的更新顺序。** 因此，如下图所示，如果两个圆不会作用到同一个像素，它们之间就不存在顺序要求，可以独立处理。

![Dependencies](handout/dependencies.jpg?raw=true "The contributions of circles 1, 2, and 3 must be applied to overlapped pixels in the order the circles are provided to the renderer.")

由于给定的 CUDA 实现既不满足原子性，也不满足顺序要求，所以在 rgb 和 circles 场景下运行时，就能看到未正确遵守这两个要求的结果。你会在结果图像中看到横向条纹，如下图所示。这些条纹会随着每一帧变化。

![Order_errors](handout/bug_example.jpg?raw=true "Errors in the output due to lack of atomicity in frame-buffer update (notice streaks in bottom of image).")

### 你需要做什么

**你的任务是写出一个尽可能快且正确的 CUDA 渲染器实现**。你可以采用任何你觉得合适的方法，但你的渲染器必须满足上面所说的原子性和顺序要求。如果方案不能同时满足这两个要求，那么第 3 部分最多只能拿到 12 分。我们已经给了你这样一个方案！

一个不错的起点是仔细阅读 `cudaRenderer.cu`，并说服自己它 _确实不满足_ 正确性要求。尤其请看 `CudaRenderer:render` 是如何启动 CUDA kernel `kernelRenderCircles` 的。（所有工作都在 `kernelRenderCircles` 中完成。）为了直观看到违反这两个要求的效果，请先执行 `make` 编译程序，然后运行 `./render -r cuda rand10k`，它会显示包含 1 万个圆的图像，也就是上图底部那一行所示的结果。再运行 `./render -r cpuref rand10k`，将其与顺序代码生成的正确图像进行比较。

我们建议你：

1. 先重写起始 CUDA 实现，使其在并行执行时逻辑上是正确的（我们推荐一种不需要锁或同步的做法）
2. 然后分析你的方案存在哪些性能问题
3. 到这里，这份作业真正需要动脑的部分才开始……（提示：`circleBoxTest.cu_inl` 中给你的 circle-intersects-box 测试函数会很有帮助。我们鼓励你使用这些子程序。）

下面是 `./render` 的命令行选项：

```text
Usage: ./render [options] scenename
Valid scenenames are: rgb, rgby, rand10k, rand100k, rand1M, biglittle, littlebig, pattern, micro2M,
                      bouncingballs, fireworks, hypnosis, snow, snowsingle
Program Options:
  -r  --renderer <cpuref/cuda>  Select renderer: ref or cuda (default=cuda)
  -s  --size  <INT>             Rendered image size: <INT>x<INT> pixels (default=1024)
  -b  --bench <START:END>       Run for frames [START,END) (default=[0,1))
  -c  --check                   Check correctness of CUDA output against CPU reference
  -i  --interactive             Render output to interactive display
  -f  --file  <FILENAME>        Output file name (FILENAME_xxxx.ppm) (default=output)
  -?  --help                    This message
```

**检查代码（Checker code）：** 为了检测程序正确性，`render` 提供了方便的 `--check` 选项。它会同时运行顺序 CPU 参考渲染器和你的 CUDA 渲染器，然后比较两者生成的图像是否一致，以确认正确性。同时，它也会打印你的 CUDA 渲染器实现的运行时间。

我们总共提供了 8 组圆形数据集用于评分。不过，若想获得满分，你的代码必须通过我们所有的正确性测试。要检查代码的正确性和性能得分，请在 `/render` 目录中运行 **`./checker.py`**（注意有 `.py` 扩展名）。如果你对起始代码运行它，程序会打印出类似下面的表格，以及完整测试集的结果：

```text
Score table:
------------
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.2622           | (F)             | 0               |
| rand10k         | 3.0658           | (F)             | 0               |
| rand100k        | 29.6144          | (F)             | 0               |
| pattern         | 0.4043           | (F)             | 0               |
| snowsingle      | 19.7155          | (F)             | 0               |
| biglittle       | 15.2422          | (F)             | 0               |
| rand1M          | 230.478          | (F)             | 0               |
| micro2M         | 439.9369         | (F)             | 0               |
--------------------------------------------------------------------------
|                                    | Total score:    | 0/72            |
--------------------------------------------------------------------------
```

注意：在某些运行中，你 _可能_ 会在部分场景上拿到分，因为给定渲染器的运行结果有时是非确定性的，偶尔可能碰巧正确。但这并不改变当前 CUDA 渲染器在一般情况下是错误实现这一事实。

“Ref time” 指的是我们的参考解在你当前机器上运行的性能（由提供的 `render_ref` 可执行文件给出）。“Your time” 指的是你当前 CUDA 渲染器方案的性能，其中 `(F)` 表示方案不正确。你的成绩将取决于你的实现相对于这些参考实现的性能表现（见下文的评分说明）。

除了代码之外，我们还希望你提交一份清晰的高层次说明，介绍你的实现是如何工作的，以及你是如何得到这个方案的。请特别说明你中途尝试过哪些方法，以及你是如何确定该如何优化代码的（例如：你做了哪些测量来指导优化？）。

写作报告中应提到的内容包括：

1. 在报告顶部写上两位合作者的姓名和 SUNet ID。
2. 复现你方案对应的得分表，并说明你是在什么机器上运行代码的。
3. 描述你如何拆分问题，以及如何把工作分配到 CUDA thread blocks、threads，甚至 warps。
4. 描述你的方案中同步发生在什么地方。
5. 你是否采取了措施来减少通信需求（例如同步开销或主存带宽需求）？如果有，请说明。
6. 简要说明你是如何得到最终方案的。途中还尝试过哪些其他方法？它们有什么问题？

### 评分说明

- 作业报告值 18 分。
- 你的并行前缀和实现值 10 分。
- 你的渲染器实现值 72 分。它们平均分配到 8 个场景上，即每个场景 9 分，具体如下：
  - 每个场景 2 分正确性分。我们只会使用图像尺寸是 256 倍数的测试。
  - 每个场景 7 分性能分（只有在方案正确时才可获得）。性能评分将相对于给定的参考渲染器性能 `T_ref` 来评定：
    - 如果你的运行时间 `T` 达到 `T_ref` 的 10 倍或更慢，则没有性能分。
    - 如果你的运行时间在优化后参考解的 20% 以内（`T <= 1.20 * T_ref`），则获得满性能分。
    - 对于其他情况（`1.20 T_ref < T < 10 * T_ref`），你的性能分（1 到 7 分）按如下公式计算：`7 * T_ref / T`。

- 对于性能显著超过要求的方案，最多可获得 5 分额外加分（由老师酌情决定）。你的报告必须非常清晰、充分地解释你的方法。
- 对于高质量的纯 CPU 并行渲染器实现，如果它能很好地利用所有 CPU 核以及 SIMD 向量单元，也最多可获得 5 分额外加分（由老师酌情决定）。你可以自由使用各种工具（例如 SIMD intrinsics、ISPC、pthreads）。若想获得这部分加分，你需要分析 GPU 方案和 CPU 方案的性能，并讨论为什么它们在实现选择上会有差异。

因此，本项目的总分构成为：

- 第 1 部分（5 分）
- 第 2 部分（10 分）
- 第 3 部分报告（13 分）
- 第 3 部分实现（72 分）
- 额外加分（最多 10 分）

## 作业提示与建议

下面整理了一些往年经验中的提示与建议。注意，渲染器有多种实现方式，并不是所有提示都一定适用于你的方案。

- 这份作业中有两个潜在的并行维度。一个是 _跨像素并行_，另一个是 _跨圆并行_（前提是满足重叠像素上的顺序要求）。可行方案通常需要同时利用这两类并行性，只是可能出现在计算流程的不同阶段。
- `circleBoxTest.cu_inl` 中提供的 circle-intersects-box 测试函数非常有用。我们鼓励你使用这些子程序。
- `exclusiveScan.cu_inl` 中提供的 shared-memory prefix-sum 操作也许会对你有帮助（当然并不是所有方案都必须使用它）。你可以先看一个关于 prefix-sum 的简要说明：[here](https://docs.nvidia.com/cuda/archive/12.2.1/thrust/index.html#prefix-sums)。我们提供的是一个在 shared memory 中、针对 **长度为 2 的幂** 的数组实现的 exclusive prefix-sum。**给定代码不能处理非 2 的幂长度输入，而且它还要求 thread block 中的线程数必须等于数组长度。请务必阅读代码中的注释。**
- 注意正在被调用的 `shadePixel` 方法。它在更新像素颜色时会进行很多次全局内存访问。也许更好的做法是在 `kernelRenderCircles` 中使用局部累加器。这样你可以先在寄存器里累加像素值，等最终像素值计算完成后，再只进行一次全局内存写入。
- 你可以在实现中使用 [Thrust library](http://thrust.github.io/)，如果你愿意的话。要达到优化后的 CUDA 参考实现的性能，其实并不需要 Thrust。有一种很常见的解法会使用我们给你的 shared memory prefix-sum 实现；还有另一种常见解法会使用 Thrust 库中的 prefix-sum 例程。这两种策略都合理。
- 渲染器中是否存在数据复用？你可以怎样利用这种复用？
- 由于 CUDA 语言没有一种原语能把整套图像更新逻辑原子化执行，那么你要如何保证图像更新的原子性？一种做法是用全局内存上的原子操作来构造锁，但请记住：即便你的图像更新是原子的，更新也仍然必须按要求顺序执行。**我们建议你先思考如何在并行解法中保证顺序，然后再考虑原子性问题（如果到那时它还存在的话）。**
- 对于包含大量圆的测试，如 `rand1M` 和 `micro2M`，你需要小心在全局内存中分配临时结构。如果分配过多，很可能会耗尽设备内存。如果你没有检查 `cudaMalloc` 返回的 `cudaError_t`，程序可能仍会继续执行，但你并不知道自己已经把设备内存用完了。最终你会因为临时结构没有成功建立而通过不了正确性检查。这也是为什么我们建议你使用下面这个 CUDA API 包装器来封装 `cudaMalloc` 调用，这样一旦设备内存耗尽，就能直接报错。
- 如果你最后还有空，不妨自己做一些有趣的场景！

### 捕获 CUDA 错误

默认情况下，如果你访问数组越界、分配了过多内存，或者以其他方式触发了错误，CUDA 通常不会主动明确提示你；它往往只是静默失败，并返回一个错误码。你可以使用下面这个宏（可以按需修改）来封装 CUDA 调用：

```cpp
#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif
```

注意，一旦你的代码已经正确，你可以取消定义 `DEBUG` 来关闭错误检查，以获得更好的性能。

然后你就可以像下面这样封装 CUDA API 调用，从而处理其返回的错误：

```cpp
cudaCheckError( cudaMalloc(&a, size*sizeof(int)) );
```

注意，你不能直接封装 kernel 启动语句。相反，kernel 的错误会在下一次被封装的 CUDA 调用中暴露出来：

```cpp
kernel<<<1,1>>>(a); // suppose kernel causes an error!
cudaCheckError( cudaDeviceSynchronize() ); // error is printed on this line
```

所有 CUDA API 函数，如 `cudaDeviceSynchronize`、`cudaMemcpy`、`cudaMemset` 等，都可以这样封装。

**重要：** 如果某个 CUDA 函数之前已经出错，但没有被捕获，那么这个错误会在下一次错误检查时显示出来，即使你封装的是另一个不同的函数。例如：

```cpp
...
line 742: cudaMalloc(&a, -1); // executes, then continues
line 743: cudaCheckError(cudaMemcpy(a,b)); // prints "CUDA Error: out of memory at cudaRenderer.cu:743"
...
```

因此，在调试时，建议你把 **所有** CUDA API 调用都封装起来（至少你自己写的那部分代码里要这么做）。

（致谢：改编自 [this Stack Overflow post](https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api)）

## 3.4 提交说明

请通过 Gradescope 提交你的作业。如果你和搭档合作完成，请记得在 Gradescope 上标记你的搭档。

1. **请将你的报告以 `writeup.pdf` 文件形式提交。**
2. **请运行 `sh create_submission.sh` 来生成要提交到 Gradescope 的 zip 包。** 注意，这个脚本会在代码目录中执行 `make clean`，所以之后如果你还要运行代码，需要重新执行 `make`。如果脚本报错提示 `Permission denied`，请先运行 `chmod +x create_submission.sh`，然后再重试。

我们的评分脚本会重新运行 checker，以验证你在 `writeup.pdf` 中提交的得分是否一致。我们也可能会在其他数据集上运行你的代码，以进一步检查其正确性。
