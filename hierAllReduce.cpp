// Original File
// This is what we started with and gave the preliminary data 
#include <algorithm>
#include <cstdio>
#include <mpi.h>
#include <vector>
using namespace std;

/*
    Communicators used by the hierarchical implementation
    node_comm groups ranks sharing one compute node
    leader_comm contains one rank from each compute node
*/
struct Comms {
    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Comm leader_comm = MPI_COMM_NULL;
};

/* Context shared by both benchmarked collective implementations */
struct BenchmarkContext {
    MPI_Comm world = MPI_COMM_NULL;
    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Comm leader_comm = MPI_COMM_NULL;
};

/* Describes one payload size and its number of timed repetitions */
struct BenchmarkCase {
    const char* label;
    int elementCount;
    int timed_iterations;
};

/* Stores the maximum per-rank average and the result validation status */
struct Measurement {
    double average_microseconds;
    bool is_correct;
};

using AllReduceOperation = int (*)(
    const int* sendBuffer,
    int* receiveBuffer,
    int elementCount,
    const BenchmarkContext& context
);

// Globals 
constexpr int kWarmupIterations = 5;
constexpr int kBenchmarkRuns = 100;
constexpr int kScalarElements = 1;
constexpr int kKilobyteElements = 1024;
constexpr int kMediumElements = 64 * 1024;
constexpr int kLargeElements = 1024 * 1024;
constexpr int kScalarIterations = 10000;
constexpr int kKilobyteIterations = 1000;
constexpr int kMediumIterations = 100;
constexpr int kLargeIterations = 10;
constexpr double kMicrosecondsPerSecond = 1000000.0;

constexpr BenchmarkCase kBenchmarkCases[] = {
    {"1 int", kScalarElements, kScalarIterations},
    {"1 Ki ints", kKilobyteElements, kKilobyteIterations},
    {"64 Ki ints", kMediumElements, kMediumIterations},
    {"1 Mi ints", kLargeElements, kLargeIterations},
};

/* Builds communicators for ranks on the same node and node leaders */
static int build_comms(MPI_Comm world, Comms* comms) {
    int return_code = MPI_Comm_split_type(
        world,
        MPI_COMM_TYPE_SHARED,
        0,
        MPI_INFO_NULL,
        &comms->node_comm
    );

    if (return_code != MPI_SUCCESS) {
        return return_code;
    }

    int world_rank = -1;
    int node_rank = -1;
    MPI_Comm_rank(world, &world_rank);
    MPI_Comm_rank(comms->node_comm, &node_rank);

    const int color = node_rank == 0 ? 0 : MPI_UNDEFINED;
    return MPI_Comm_split(
        world,
        color,
        world_rank,
        &comms->leader_comm
    );
}

/* Frees communicators created for the hierarchical implementation */
static void free_comms(Comms* comms) {
    if (comms == nullptr) {
        return;
    }

    if (comms->leader_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&comms->leader_comm);
    }

    if (comms->node_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&comms->node_comm);
    }

    comms->leader_comm = MPI_COMM_NULL;
    comms->node_comm = MPI_COMM_NULL;
}

/* Runs the node-aware reduce, leader all-reduce, and local broadcast */
static int hierarchical_all_reduce(
    const int* sendBuffer,
    int* receiveBuffer,
    int elementCount,
    const BenchmarkContext& context
) {
    int node_rank = -1;
    MPI_Comm_rank(context.node_comm, &node_rank);

    int return_code = MPI_Reduce(
        sendBuffer
,
        receiveBuffer,
        elementCount,
        MPI_INT,
        MPI_SUM,
        0,
        context.node_comm
    );

    if (return_code != MPI_SUCCESS) {
        return return_code;
    }

    if (node_rank == 0) {
        return_code = MPI_Allreduce(
            MPI_IN_PLACE,
            receiveBuffer,
            elementCount,
            MPI_INT,
            MPI_SUM,
            context.leader_comm
        );
    }

    if (return_code != MPI_SUCCESS) {
        return return_code;
    }

    return MPI_Bcast(
        receiveBuffer,
        elementCount,
        MPI_INT,
        0,
        context.node_comm
    );
}

/* Runs OpenMPI's built-in all-reduce for the same integer sum operation */
static int mpi_all_reduce(
    const int* sendBuffer,
    int* receiveBuffer,
    int elementCount,
    const BenchmarkContext& context
) {
    return MPI_Allreduce(
        sendBuffer
,
        receiveBuffer,
        elementCount,
        MPI_INT,
        MPI_SUM,
        context.world
    );
}

/* Checks that every output element contains the expected global sum */
static bool has_expected_result(
    const vector<int>& receiveBuffer,
    int expected_sum
) {
    return all_of(
        receiveBuffer.begin(),
        receiveBuffer.end(),
        [expected_sum](int value) { return value == expected_sum; }
    );
}

/* Measures one collective and validates its final result outside timing */
static Measurement measure_operation(
    AllReduceOperation operation,
    const BenchmarkCase& benchmark_case,
    const BenchmarkContext& context,
    int rank_value,
    int expected_sum
) {
    vector<int> sendBuffer(
        benchmark_case.elementCount,
        rank_value
    );
    vector<int> receiveBuffer(benchmark_case.elementCount, 0);

    for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
        operation(
            sendBuffer
    .data(),
            receiveBuffer.data(),
            benchmark_case.elementCount,
            context
        );
    }

    MPI_Barrier(context.world);
    const double start_time = MPI_Wtime();

    for (int iteration = 0;
         iteration < benchmark_case.timed_iterations;
         ++iteration) {
        operation(
            sendBuffer
    .data(),
            receiveBuffer.data(),
            benchmark_case.elementCount,
            context
        );
    }

    const double local_elapsed = MPI_Wtime() - start_time;
    double maximum_elapsed = 0.0;
    MPI_Reduce(
        &local_elapsed,
        &maximum_elapsed,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        context.world
    );

    const int local_is_correct = has_expected_result(
        receiveBuffer,
        expected_sum
    ) ? 1 : 0;
    int globally_correct = 0;
    MPI_Allreduce(
        &local_is_correct,
        &globally_correct,
        1,
        MPI_INT,
        MPI_LAND,
        context.world
    );

    const double average_microseconds = maximum_elapsed /
        benchmark_case.timed_iterations * kMicrosecondsPerSecond;
    return {average_microseconds, globally_correct != 0};
}

/* Prints spreadsheet column names from rank zero */
static void printCSVHeader() {
    printf(
        "run,payload,elements,payloadBytes,iterations,mpi_us_op,"
        "custom_us_op,slowdown,mpiCorrect,customCorrect,worldRanks,"
        "nodeCount\n"
    );
}

/* Prints one spreadsheet-compatible benchmark row from rank zero */
static void print_csv_row(
    int run_number,
    const BenchmarkCase& benchmarkCase,
    const Measurement& mpiMeasurement,
    const Measurement& customMeasurement,
    int worldSize,
    int nodeCount
) {
    const double slowdown = mpiMeasurement.averageMicroseconds > 0.0
        ? customMeasurement.averageMicroseconds /
            mpiMeasurement.averageMicroseconds
        : 0.0;

    const long long payloadBytes = static_cast<long long>(
        benchmarkCase.elementCount
    ) * sizeof(int);

    printf(
        "%d,%s,%d,%lld,%d,%.6f,%.6f,%.6f,%d,%d,%d,%d\n",
        run_number,
        benchmarkCase.label,
        benchmarkCase.elementCount,
        payloadBytes,
        benchmarkCase.timedIterations,
        mpiMeasurement.averageMicroseconds,
        custom_measurement.averageMicroseconds,
        slowdown,
        mpi_measurement.isCorrect ? 1 : 0,
        custom_measurement.isCorrect ? 1 : 0,
        world_size,
        node_count
    );
}

int main(int argc, char** argv) {
    // Begin MPI 
    MPI_Init(&argc, &argv);

    int world_rank = -1;
    int world_size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Set up communication 
    Comms comms;
    const int build_result = build_comms(MPI_COMM_WORLD, &comms);

    if (build_result != MPI_SUCCESS) {
        if (world_rank == 0) {
            fprintf(stderr, "Could not build node-aware communicators\n");
        }

        MPI_Abort(MPI_COMM_WORLD, build_result);
    }

    // Set up context 
    BenchmarkContext context;
    context.world = MPI_COMM_WORLD;
    context.node_comm = comms.node_comm;
    context.leader_comm = comms.leader_comm;

    int node_count = 0;

    if (world_rank == 0) {
        MPI_Comm_size(context.leader_comm, &node_count);
    }

    const int expected_sum = world_size * (world_size + 1) / 2;

    if (world_rank == 0) {
        printCSVHeader();
    }

    for (int run_number = 1;
         run_number <= kBenchmarkRuns;
         ++run_number) {
        for (const BenchmarkCase& benchmark_case : kBenchmarkCases) {
            const Measurement mpi_measurement = measure_operation(
                mpi_all_reduce,
                benchmark_case,
                context,
                world_rank + 1,
                expected_sum
            );
            const Measurement custom_measurement = measure_operation(
                hierarchical_all_reduce,
                benchmark_case,
                context,
                world_rank + 1,
                expected_sum
            );

            if (world_rank == 0) {
                print_csv_row(
                    run_number,
                    benchmark_case,
                    mpi_measurement,
                    custom_measurement,
                    world_size,
                    node_count
                );
            }
        }
    }

    free_comms(&comms);
    MPI_Finalize();
    return 0;
}
