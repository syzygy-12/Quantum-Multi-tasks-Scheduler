使用python写一个简单调度算法，要求如下：

我们模拟一个真实的物理量子计算机，其中有一些拓扑和节点，然后有若干需要运行的程序，每个程序需要占用一定的节点，占用一定的计算时间，安排顺序和调度（程序可以并行运行，只要有足够的物理资源），使得运行总时间最短。

假设：

每个物理点为一个量子位，边表示直接耦合（只用于连通性要求）。

任务只有三个属性：k（需要的量子位数），d（运行时长），任务的拓扑结构。

所有任务都可同时提交（没有到达时间/优先级）。

同一任务所占的 k 个量子位必须在拓扑上连通，然而会有一定的swap cost。

任务运行期间所占量子位被独占（不能重叠）。

新要求：

在实际情况里，每一个量子程序其实是有一个topo的，如果这个topo不能fit到量子计算机的物理topo，则需要把他swap过去，这里会引入额外的运行时间，这部分cost应该怎么算？

答案：

设逻辑程序的拓扑为 \(G_{\mathrm{prog}}=(V_p,E_p)\)，物理芯片拓扑为 \(G_{\mathrm{chip}}=(V_c,E_c)\)，
任务 \(T\) 的运行时长为 \(d(T)\)。

若一个 embedding 将 \(V_p\) 映射到 \(V_c\)，则对每条逻辑边 \((u,v)\in E_p\)，
计算其在物理拓扑上的最短距离 \(\mathrm{dist}_{G_{\mathrm{chip}}}(\pi(u),\pi(v))\)。

则所需的 SWAP 数量近似为：
\[
N_{\mathrm{swap}} \approx \sum_{(u,v)\in E_p} \bigl(\mathrm{dist}_{G_{\mathrm{chip}}}(\pi(u),\pi(v)) - 1\bigr)
\]

于是带 SWAP 开销的任务时长估算为：
\[
d'(T) \;\approx\; d(T) \;+\; c_{\mathrm{swap}}\cdot N_{\mathrm{swap}}
\]

其中 \(c_{\mathrm{swap}}\) 为单个 SWAP 的等效时长（可取物理门时长的平均值）。

新功能：

我想支持人工可控的network topology，而非写死的，为了完成这个目标，我想先写一个create_topology.py用来将传入的topology转化成networkx并且存在本地，在main.py里调用。其中create_topology.py我想写成mcp风格的，因为我以后想让大模型根据自然语言去翻译成topology的关系。