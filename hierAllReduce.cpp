#include <mpi.h>
#include <iostream>

using namespace std;

/*
    struct Comms
        nodeComm: all ranks on curr node
        leaderComm: one rank per node
*/
struct Comms {
    MPI_Comm nodeComm   = MPI_COMM_NULL;
    MPI_Comm leaderComm = MPI_COMM_NULL;
};


/*
    int buildComms
        builds 2 communicators once when code is run
*/
static int buildComms(MPI_Comm world, Comms* c) {
    int rc = MPI_Comm_split_type(
        world,
        MPI_COMM_TYPE_SHARED,
        0,
        MPI_INFO_NULL,
        &c->nodeComm
    ); // group ranks

    int world_rank = -1;
    int node_rank = -1;

    MPI_Comm_rank(world, &world_rank);

    // decide leader
    MPI_Comm_rank(c->nodeComm, &node_rank);

    // leaders: node_rank == 0
    int color;

    if (node_rank == 0) {
        color = 0; // leader
    }
    else {
        color  = MPI_UNDEFINED; // not leader
    }

    rc = MPI_Comm_split(
        world,
        color,
        world_rank,
        &c->leaderComm
    ); // create leader
    // if color = 0 : leader
    // else MPI_COMM_NULL and no leader communication
    return rc;
}

/*
    void freeComms
        clean up communicators
        avoids memory leaks
*/
static void freeComms(Comms* c) {
    if (!c) return;
    if (c->leaderComm != MPI_COMM_NULL) MPI_Comm_free(&c->leaderComm);
    if (c->nodeComm   != MPI_COMM_NULL) MPI_Comm_free(&c->nodeComm);
    c->leaderComm = MPI_COMM_NULL;
    c->nodeComm   = MPI_COMM_NULL;
}

// hAllReduce(), algo B
static int hAllReduce(
    const void* sendbuf,
    void* recvbuf,
    int count,
    MPI_Datatype datatype,
    MPI_Op op,
    MPI_Comm world,
    MPI_Comm nodeComm,
    MPI_Comm leaderComm
) {

    int node_rank = -1;
    MPI_Comm_rank(nodeComm, &node_rank);

    // MPI_IN_PLACE handling, need this for MPI_reduce/MPI_allreduce to work
    const void* effectiveSend = sendbuf;
    if (sendbuf == MPI_IN_PLACE) {
        effectiveSend = recvbuf;
    }

    // phase 1: intra-node reduce to node leader
    int rc = MPI_Reduce(
        effectiveSend,
        recvbuf,     // meaningful at node leader
        count,
        datatype,
        op,
        0,
        nodeComm
    );

    // phase 2: inter-node allreduce for leaders
    if (node_rank == 0) {
        // leaderComm is valid only for leaders, others have MPI_COMM_NULL
        rc = MPI_Allreduce(
            MPI_IN_PLACE,
            recvbuf,
            count,
            datatype,
            op,
            leaderComm
        );
    }

    // phase 3: broadcast result within node
    rc = MPI_Bcast(
        recvbuf,
        count,
        datatype,
        0,
        nodeComm
    );
    return rc;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = -1, world_size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    Comms c;
    int rc = buildComms(MPI_COMM_WORLD, &c);

    int node_rank = -1, node_size = -1;
    MPI_Comm_rank(c.nodeComm, &node_rank);
    MPI_Comm_size(c.nodeComm, &node_size);

    // sum of (rank + 1)
    int x = world_rank + 1;
    int y = 0;

    rc = hAllReduce(
        &x,
        &y,
        1,
        MPI_INT,
        MPI_SUM,
        MPI_COMM_WORLD,
        c.nodeComm,
        c.leaderComm
    );

    cout << "WorldRank = " << world_rank << "/" << world_size
              << " NodeRank = " << node_rank << "/" << node_size
              << " Result = " << y << endl;

    freeComms(&c);
    MPI_Finalize();
    return 0;
}