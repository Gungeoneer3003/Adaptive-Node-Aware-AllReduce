# Adaptive Node-Aware Allreduce

## Notes about implementation
At the current moment, the code base is a bit messy because this also includes the vs community project, the preliminary research source file, and the range data source file that produces our final data. Something that should be done is a refactoring that creates a header file with the implementation details of both files.

## Project Goal

Design and evaluate an adaptive MPI `Allreduce` implementation for commodity
HPC clusters. The implementation should detect node topology, select between a
vendor-provided baseline and a node-aware hierarchical algorithm, and explain
its performance using a simple communication model.

## Motivation

An `Allreduce` combines values from every process and returns the result to
every process. Communication between nodes is generally more expensive than
communication within a node. A node-aware algorithm can reduce costly
inter-node communication and may outperform the default MPI implementation for
some message sizes and cluster configurations.

## Scope

Keep the study focused on two algorithms:

1. A vendor MPI `Allreduce` baseline
2. A hierarchical, node-aware `Allreduce`

This narrower comparison makes it possible to evaluate the trade-offs in depth
instead of implementing many insufficiently tested variants.

## Step 1: Detect Node Topology

MPI can create a communicator containing the ranks that share a physical node.
This avoids hardware-specific probing and remains portable across MPI
implementations.

```cpp
MPI_Comm nodeComm;

MPI_Comm_split_type(
    MPI_COMM_WORLD,
    MPI_COMM_TYPE_SHARED,
    0,
    MPI_INFO_NULL,
    &nodeComm);
```

### Result

Processes are automatically grouped by node. This provides an immediate
opportunity to perform local reduction before communicating across nodes.

### Advantages

- No hardware probing is required
- The approach is portable
- It works across MPI implementations
- It exposes locality that can improve collective performance

## Step 2: Implement the Algorithms

### Algorithm A: Vendor Baseline

Use the MPI implementation's existing collective as the baseline.

```cpp
MPI_Allreduce(
    sendBuffer,
    receiveBuffer,
    count,
    dataType,
    operation,
    MPI_COMM_WORLD);
```

The baseline represents the performance available without the project-specific
optimization. Do not reimplement an already optimized vendor algorithm unless
that is necessary to answer the research question.

### Algorithm B: Hierarchical Node-Aware Allreduce

The hierarchical algorithm has three phases.

#### Phase 1: Intra-Node Reduction

Each node reduces its local ranks' values to one node leader. The leader is
rank zero within `nodeComm`.

```cpp
MPI_Reduce(
    localSend,
    leaderReceive,
    count,
    dataType,
    operation,
    0,
    nodeComm);
```

#### Phase 2: Inter-Node Reduction

Create a communicator containing only node leaders. Non-leaders do not join
this communicator.

```cpp
MPI_Comm leaderComm;

MPI_Comm_split(
    MPI_COMM_WORLD,
    isLeader ? 0 : MPI_UNDEFINED,
    worldRank,
    &leaderComm);
```

The node leaders combine the partial results.

```cpp
MPI_Allreduce(
    MPI_IN_PLACE,
    leaderReceive,
    count,
    dataType,
    operation,
    leaderComm);
```

#### Phase 3: Intra-Node Broadcast

Each leader distributes the final result to every rank on its node.

```cpp
MPI_Bcast(
    leaderReceive,
    count,
    dataType,
    0,
    nodeComm);
```

## Runtime Algorithm Selector

The project's primary research contribution is a selector that chooses an
algorithm before the main computation begins.

Run small microbenchmarks to estimate:

- Latency
- Bandwidth
- The crossover message size where one algorithm becomes faster

Use those measurements to choose the implementation for each message-size
range. A simple starting heuristic is:

```cpp
constexpr std::size_t hierarchicalThreshold = 8 * 1024;

if (messageSize < hierarchicalThreshold) {
    useFlatAllreduce();
} else {
    useHierarchicalAllreduce();
}
```

An auto-tuned version can replace the fixed threshold with results from short
probing runs. Even a simple, well-evaluated selector may outperform the default
MPI behavior on real clusters.

## Performance Model

Use the alpha-beta communication model to interpret the measurements:

```text
T = latency + (messageSize / bandwidth)
```

The key hypothesis is that hierarchical collectives lower the latency cost by
reducing the number of expensive inter-node messages. The evaluation should
compare measured timings with the model's predictions.

## Evaluation Plan

Measure and compare both algorithms across:

- Multiple message sizes, especially around the crossover point
- Different process counts
- Different ranks-per-node configurations
- Repeated runs sufficient to report stable timing statistics

Report latency, bandwidth, speedup relative to the vendor baseline, and the
accuracy of the selector's choices.

## Possible Extensions

### Rank Reordering

Map frequently communicating processes to physically nearby resources rather
than accepting the default rank placement. This may provide substantial gains
on topology-sensitive clusters.

### Auto-Tuning

Use brief startup probes to choose the best algorithm and crossover threshold
for the current cluster and job layout. This strengthens the systems-research
contribution beyond a fixed heuristic.

## Proposed Title

> Adaptive Node-Aware Allreduce for Commodity HPC Clusters
