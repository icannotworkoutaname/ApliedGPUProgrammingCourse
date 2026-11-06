# Question 1：复现 CUDA 向量加法结果

本项目在 Google Colab GPU 运行时中执行，调用方式与课程教程一致。

## 1. 准备文件和环境

需要两个文件：

- [assignment1.ipynb](assignment1.ipynb)：编译、运行、绘图和 Nsight 分析。
- [vecAdd.cu](vecAdd.cu)：CPU/GPU 向量加法源码。

当前 notebook 保存的运行环境为 **Tesla T4、CUDA Toolkit 13.0.88**；源码每个 block 使用 **32 个线程**。

1. 在 Colab 打开 `assignment1.ipynb`。
2. 将运行时类型设为 GPU，选择 T4（如果可选）。
3. 手动把本目录中的 `vecAdd.cu` 上传到 Colab 的 `/content`。
4. 从头依次运行 notebook 单元格，重新生成结果。

下面带 `!`、`%` 的命令都放在 **Colab 代码单元格**中运行。

## 2. 检查环境并编译

```python
!nvidia-smi
!nvcc --version
```

```python
%cd /content
!ls -l vecAdd.cu
!nvcc -arch=sm_75 vecAdd.cu -o vecAdd
```

`sm_75` 对应 T4。修改源码后，需要重新上传并编译。

## 3. 运行两个指定长度

```python
!./vecAdd 512
!./vecAdd 263149
```

程序只接收向量长度 `N`；不传参数时默认 `N=512`。每次运行只启动一次 kernel。

输出列为：

```text
N,threads,blocks,H2D_ms,kernel_ms,D2H_ms,max_error
```

- `threads`：每个 block 的线程数。
- `blocks`：启动的 block 数；总线程数为 `threads × blocks`。
- `H2D_ms`：两个输入向量从 CPU 复制到 GPU 的时间。
- `kernel_ms`：CPU 观察到的 kernel 启动、执行和同步时间。
- `D2H_ms`：输出向量从 GPU 复制回 CPU 的时间。
- `max_error`：所有元素的 CPU/GPU 最大绝对误差；为 0 表示数值完全相等。

当前 notebook 保存的两个测试结果均为 `max_error=0`：

| N | threads | blocks |
| ---: | ---: | ---: |
| 512 | 32 | 16 |
| 263149 | 32 | 8224 |

重新运行后，检查自己的输出是否符合上述配置，并查看 `max_error`。

## 4. 复现计时表和堆叠柱状图

继续运行 notebook 中的“导入绘图库”“测量不同长度”“堆叠柱状图”和“查看 CPU/GPU 误差”单元格。

程序会测量 notebook 中配置的 16 组长度（512 至 16777216，包含 263149），并在 Colab `/content` 生成：

| 文件 | 内容 |
| --- | --- |
| `timing_results.csv` | 每个长度的线程配置、三段耗时和最大误差 |
| `timing_breakdown.png` | H2D、kernel、D2H 的堆叠柱状图 |

当前实现每个长度只测量一次，不预热、不重复取平均。CPU 参考计算、内存分配和结果校验不计入三段时间；计时边界会等待 GPU 完成。重新运行时耗时会波动，不要求与 notebook 已保存的数值完全一致。

## 5. 复现 Achieved Occupancy

先检查 Nsight Compute 的分析模块：

```python
!ncu --list-sections
```

再分别运行：

```python
!ncu --section Occupancy ./vecAdd 512
```

```python
!ncu --section Occupancy ./vecAdd 263149
```

读取输出中 **Achieved Occupancy** 一行，不要读取 **Theoretical Occupancy**。

当前 notebook 保存的一次 T4 运行结果为：

| N | Achieved Occupancy |
| ---: | ---: |
| 512 | 3.12% |
| 263149 | 33.67% |

这些数值是已保存运行的参考值，重新采集可能变化。绘图使用第 4 步的普通运行时间，不能使用 `ncu` 输出中的程序计时，因为 profiler 会增加开销。

## 6. 保存结果与常见问题

运行完成后，手动下载 `timing_results.csv`、`timing_breakdown.png`，并保存带最新输出的 notebook。Colab 运行时重置后需要重新上传源码、编译和运行。

- 找不到 `vecAdd.cu`：确认文件已上传到 `/content`，且名称一致。
- 找不到 `ncu`：当前运行时需要先配置 Nsight Compute。
- 出现 `ERR_NVGPUCTRPERM`：当前运行时不允许访问性能计数器，需要换到允许该功能的 GPU 环境采集占用率；普通计时和校验仍可单独运行。
