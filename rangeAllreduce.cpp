// Later file
// Provides the data for 1K-150K
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mpi.h>
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

/* Stores one collective latency and its global correctness status */
struct Measurement {
    double microseconds;
    bool is_locally_correct;
};

using AllReduceOperation = int (*)(
    const int* send_buffer,
    int* receive_buffer,
    int element_count,
    const BenchmarkContext& context
);


// Constants
constexpr int kMinimumElements = 1000;
constexpr int kMaximumElements = 150000;
constexpr int kElementStep = 1000;
constexpr int kWarmupIterations = 5;
constexpr int kTrialsPerPayload = 100;
constexpr double kMicrosecondsPerSecond = 1000000.0;

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
    const int* send_buffer,
    int* receive_buffer,
    int element_count,
    const BenchmarkContext& context
) {
    int node_rank = -1;
    MPI_Comm_rank(context.node_comm, &node_rank);

    int return_code = MPI_Reduce(
        send_buffer,
        receive_buffer,
        element_count,
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
            receive_buffer,
            element_count,
            MPI_INT,
            MPI_SUM,
            context.leader_comm
        );
    }

    if (return_code != MPI_SUCCESS) {
        return return_code;
    }

    return MPI_Bcast(
        receive_buffer,
        element_count,
        MPI_INT,
        0,
        context.node_comm
    );
}

/* Runs OpenMPI's built-in all-reduce for the same integer sum operation */
static int mpi_all_reduce(
    const int* send_buffer,
    int* receive_buffer,
    int element_count,
    const BenchmarkContext& context
) {
    return MPI_Allreduce(
        send_buffer,
        receive_buffer,
        element_count,
        MPI_INT,
        MPI_SUM,
        context.world
    );
}

/* Aborts every rank when a required MPI call fails */
static void require_success(
    int return_code,
    MPI_Comm world,
    const char* operation_name
) {
    if (return_code == MPI_SUCCESS) {
        return;
    }

    int world_rank = -1;
    MPI_Comm_rank(world, &world_rank);
    fprintf(
        stderr,
        "Rank %d failed MPI call: %s\n",
        world_rank,
        operation_name
    );
    MPI_Abort(world, return_code);
}

/* Checks that every output element contains the expected global sum */
static bool has_expected_result(
    const vector<int>& receive_buffer,
    int expected_sum
) {
    return all_of(
        receive_buffer.begin(),
        receive_buffer.end(),
        [expected_sum](int value) { return value == expected_sum; }
    );
}

/* Warms one collective before its timed trial series */
static void warm_operation(
    AllReduceOperation operation,
    int element_count,
    const BenchmarkContext& context,
    const vector<int>& send_buffer,
    vector<int>* receive_buffer
) {
    for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
        const int return_code = operation(
            send_buffer.data(),
            receive_buffer->data(),
            element_count,
            context
        );
        require_success(return_code, context.world, "warm-up collective");
    }
}

/* Times one collective locally after a series-level synchronization */
static Measurement measure_operation(
    AllReduceOperation operation,
    int element_count,
    const BenchmarkContext& context,
    const vector<int>& send_buffer,
    vector<int>* receive_buffer,
    int expected_sum
) {
    const double start_time = MPI_Wtime();
    const int return_code = operation(
        send_buffer.data(),
        receive_buffer->data(),
        element_count,
        context
    );
    const double local_elapsed = MPI_Wtime() - start_time;
    require_success(return_code, context.world, "timed collective");

    return {
        local_elapsed * kMicrosecondsPerSecond,
        has_expected_result(*receive_buffer, expected_sum),
    };
}

/* Prints the spreadsheet header from rank zero */
static void print_csv_header() {
    printf(
        "elements,payload_bytes,trial,mpi_us_op,custom_us_op,"
        "mpi_correct,custom_correct,mpi_first,world_ranks,node_count\n"
    );
}

/* Prints one paired MPI and custom measurement from rank zero */
static void print_csv_row(
    int element_count,
    int trial_number,
    const Measurement& mpi_measurement,
    const Measurement& custom_measurement,
    bool mpi_correct,
    bool custom_correct,
    bool mpi_first,
    int world_size,
    int node_count
) {
    const long long payload_bytes = static_cast<long long>(element_count) *
        sizeof(int);

    printf(
        "%d,%lld,%d,%.6f,%.6f,%d,%d,%d,%d,%d\n",
        element_count,
        payload_bytes,
        trial_number,
        mpi_measurement.microseconds,
        custom_measurement.microseconds,
        mpi_correct ? 1 : 0,
        custom_correct ? 1 : 0,
        mpi_first ? 1 : 0,
        world_size,
        node_count
    );
}

/* Runs all 100 paired trials for one message size */
static void benchmark_payload(
    int element_count,
    const BenchmarkContext& context,
    int world_rank,
    int world_size,
    int node_count,
    int expected_sum
) {
    // Use vectors for reduction 
    vector<int> send_buffer(element_count, world_rank + 1);
    vector<int> receive_buffer(element_count, 0);

    // Have some throwaway operations to warm up
    warm_operation(
        mpi_all_reduce,
        element_count,
        context,
        send_buffer,
        &receive_buffer
    );

    // Vectors for timing 
    vector<double> local_mpi_times(kTrialsPerPayload);
    vector<double> local_custom_times(kTrialsPerPayload);
    vector<double> maximum_mpi_times(kTrialsPerPayload);
    vector<double> maximum_custom_times(kTrialsPerPayload);
    
    // Flags for correctness 
    int mpi_locally_correct = 1;
    int custom_locally_correct = 1;

    require_success(
        MPI_Barrier(context.world),
        context.world,
        "pre-series barrier"
    );

    for (int trial_number = 1;
         trial_number <= kTrialsPerPayload;
         trial_number++) 
    {
        // Have a flag to alternate between mpi and custom
        const bool mpi_first = trial_number % 2 == 1;
        Measurement mpi_measurement = {};
        Measurement custom_measurement = {};

        if (mpi_first) {
            mpi_measurement = measure_operation(
                mpi_all_reduce,
                element_count,
                context,
                send_buffer,
                &receive_buffer,
                expected_sum
            );
            custom_measurement = measure_operation(
                hierarchical_all_reduce,
                element_count,
                context,
                send_buffer,
                &receive_buffer,
                expected_sum
            );
        } else {
            custom_measurement = measure_operation(
                hierarchical_all_reduce,
                element_count,
                context,
                send_buffer,
                &receive_buffer,
                expected_sum
            );
            mpi_measurement = measure_operation(
                mpi_all_reduce,
                element_count,
                context,
                send_buffer,
                &receive_buffer,
                expected_sum
            );
        }

        const int trial_index = trial_number - 1;
        local_mpi_times[trial_index] = mpi_measurement.microseconds;
        local_custom_times[trial_index] = custom_measurement.microseconds;
        mpi_locally_correct &= mpi_measurement.is_locally_correct;
        custom_locally_correct &= custom_measurement.is_locally_correct;
    }

    require_success(
        MPI_Reduce(
            local_mpi_times.data(),
            maximum_mpi_times.data(),
            kTrialsPerPayload,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            context.world
        ),
        context.world,
        "MPI maximum timing reduction"
    );
    require_success(
        MPI_Reduce(
            local_custom_times.data(),
            maximum_custom_times.data(),
            kTrialsPerPayload,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            context.world
        ),
        context.world,
        "custom maximum timing reduction"
    );

    int mpi_globally_correct = 0;
    int custom_globally_correct = 0;
    require_success(
        MPI_Allreduce(
            &mpi_locally_correct,
            &mpi_globally_correct,
            1,
            MPI_INT,
            MPI_LAND,
            context.world
        ),
        context.world,
        "MPI correctness reduction"
    );
    require_success(
        MPI_Allreduce(
            &custom_locally_correct,
            &custom_globally_correct,
            1,
            MPI_INT,
            MPI_LAND,
            context.world
        ),
        context.world,
        "custom correctness reduction"
    );

    if (world_rank == 0) {
        for (int trial_index = 0;
             trial_index < kTrialsPerPayload;
             ++trial_index) {
            const int trial_number = trial_index + 1;
            const Measurement mpi_measurement = {
                maximum_mpi_times[trial_index],
                mpi_globally_correct != 0,
            };
            const Measurement custom_measurement = {
                maximum_custom_times[trial_index],
                custom_globally_correct != 0,
            };
            print_csv_row(
                element_count,
                trial_number,
                mpi_measurement,
                custom_measurement,
                mpi_globally_correct != 0,
                custom_globally_correct != 0,
                trial_number % 2 == 1,
                world_size,
                node_count
            );
        }

        // Preserve completed payload results if a later range stalls
        fflush(stdout);
    }
}

/* Parses one positive integer range boundary from the command line */
static bool parse_element_count(const char* text, int* element_count) {
    char* end = nullptr;
    const long value = strtol(text, &end, 10);

    if (text == end || *end != '\0' || value < 1 || value > INT_MAX) {
        return false;
    }

    *element_count = static_cast<int>(value);
    return true;
}

int main(int argc, char** argv) {
    //Set up for MPI
    MPI_Init(&argc, &argv);

    int world_rank = -1;
    int world_size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int minimum_elements = 0;
    int maximum_elements = 0;

    // Determine if the CLI is valid (readin, ranges, and steps)
    const bool has_valid_bounds = argc == 3 &&
        parse_element_count(argv[1], &minimum_elements) &&
        parse_element_count(argv[2], &maximum_elements) &&
        minimum_elements >= kMinimumElements &&
        maximum_elements <= kMaximumElements &&
        minimum_elements <= maximum_elements &&
        minimum_elements % kElementStep == 0 &&
        maximum_elements % kElementStep == 0;

    // Exit early if CLI is invalid
    if (!has_valid_bounds) {
        if (world_rank == 0) {
            fprintf(
                stderr,
                "Usage: rangeAllreduce <minimum> <maximum>\n"
            );
        }

        MPI_Finalize();
        return 1;
    }

    // Set up comms for node
    Comms comms;
    const int build_result = build_comms(MPI_COMM_WORLD, &comms);
    require_success(build_result, MPI_COMM_WORLD, "communicator setup");

    // Set up a context for each MPI operation
    BenchmarkContext context;
    context.world = MPI_COMM_WORLD;
    context.node_comm = comms.node_comm;
    context.leader_comm = comms.leader_comm;

    int node_count = 0;

    // Set up the csv
    if (world_rank == 0) {
        MPI_Comm_size(context.leader_comm, &node_count);
        print_csv_header();
    }

    // Determine the target for checking work
    const int expected_sum = world_size * (world_size + 1) / 2;

    // Do the work, start the benchmark
    for (int element_count = minimum_elements; 
            element_count <= maximum_elements; 
            element_count += kElementStep) 
    {
        benchmark_payload(
            element_count,
            context,
            world_rank,
            world_size,
            node_count,
            expected_sum
        );
    }

    // Conclude and cleanup
    free_comms(&comms);
    MPI_Finalize();
    return 0;
}
