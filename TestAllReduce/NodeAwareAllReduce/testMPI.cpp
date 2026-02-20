#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
using namespace std;

typedef unsigned char BYTE;
typedef unsigned short WORD;
//Structs for the thread and bmp 
struct BMPFileHeader {
    unsigned short bfType; // 'BM' = 0x4D42
    unsigned int bfSize; // file size in bytes
    unsigned short bfReserved1; // must be 0
    unsigned short bfReserved2; // must be 0
    unsigned int bfOffBits; // offset to pixel data
};

struct BMPInfoHeader {
    unsigned int biSize; // header size (40)
    int biWidth; // image width
    int biHeight; // image height
    unsigned short biPlanes; // must be 1
    unsigned short biBitCount; // 24 for RGB
    unsigned int biCompression; // 0 = BI_RGB
    unsigned int biSizeImage; // image data size (can be 0 for BI_RGB)
    int biXPelsPerMeter; // resolution
    int biYPelsPerMeter; // resolution
    unsigned int biClrUsed; // colors used (0)
    unsigned int biClrImportant; // important colors (0)
};

int main(int argc, char** argv) {
    //Terminal Inputs
    /*if (argc != 2) {
        cout << "Invalid number of arguments" << endl;
        cout << "Format: ./spawn {file} {instances}" << endl;
        return 0;
    }

    char* input = argv[1];
    int n = atoi(argv[2]);*/

    //Hardcoded inputs
    char input[] = "lion.bmp";

    //Set up MPI
    MPI_Init(&argc, &argv);
    MPI_Comm topo_comm;
    topo_comm = MPI_COMM_WORLD;

    int rank, nprocs;
    MPI_Comm_rank(topo_comm, &rank); //Rank = id
    MPI_Comm_size(topo_comm, &nprocs); //Count of processes

    printf("Process %d is starting\n", rank);

    //Set bounds
    BYTE* data = 0;

    //Figure out division
    int base, rem;
    int nloc;

    int rwb, w, h;
    BMPFileHeader fh;
    BMPInfoHeader fih;

    //ID 0 Setting up for the rest
    int* counts = 0, * displs = 0;

    if (rank == 0)
    {
        //Take BMP Input
        //Load file
        FILE* f = fopen(input, "rb");
        if (f == NULL) {
            perror("Failed to open the file.");
            exit(1);
        }

        fread(&fh.bfType, 2, 1, f);
        fread(&fh.bfSize, 4, 1, f);
        fread(&fh.bfReserved1, 2, 1, f);
        fread(&fh.bfReserved2, 2, 1, f);
        fread(&fh.bfOffBits, 4, 1, f);

        fread(&fih, sizeof(fih), 1, f);

        data = (BYTE*)malloc(fih.biSizeImage * sizeof(BYTE));
        fread(data, fih.biSizeImage, 1, f);

        int byteWidth = fih.biWidth * 3;
        int padding = 4 - byteWidth % 4;
        if (padding == 4)
            padding = 0;
        rwb = byteWidth + padding;

        w = fih.biWidth;
        h = fih.biHeight;

        counts = (int*)malloc(nprocs * sizeof(int));
        displs = (int*)malloc(nprocs * sizeof(int));

        base = h / nprocs, rem = h % nprocs;

        int off = 0;
        for (int p = 0; p < nprocs; p++)
        {
            int rows = base + (p < rem);
            counts[p] = rows * rwb;
            displs[p] = off;
            off += counts[p];
        }

    }

    //The other option that doesn't use b_cast cuts the padding out of the image
    MPI_Bcast(&rwb, 1, MPI_INT, 0, topo_comm);
    MPI_Bcast(&w, 1, MPI_INT, 0, topo_comm);

    MPI_Scatter(counts, 1, MPI_INT, &nloc, 1, MPI_INT, 0, topo_comm);
    BYTE* loc = (BYTE*)malloc((nloc + 2 * rwb) * sizeof(BYTE));
    BYTE* newLoc = (BYTE*)malloc((nloc + 2 * rwb) * sizeof(BYTE));
    MPI_Scatterv(data, counts, displs, MPI_CHAR, &loc[rwb], nloc, MPI_CHAR, 0, topo_comm);

    //Note: This data is sequential, rank 0 has the first chunk
    //send and receive the data

    //First round of sendrecv
    if (rank % 2 == 0)
        if (rank != nprocs - 1) MPI_Sendrecv(&loc[nloc], rwb, MPI_CHAR, rank + 1, 0, &loc[nloc + rwb], rwb, MPI_CHAR, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank % 2 == 1)
        if (rank != 0) MPI_Sendrecv(&loc[rwb], rwb, MPI_CHAR, rank - 1, 0, loc, rwb, MPI_CHAR, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    //Second round of sendrecv
    if (rank % 2 == 1)
        if (rank != nprocs - 1) MPI_Sendrecv(&loc[nloc], rwb, MPI_CHAR, rank + 1, 0, &loc[nloc + rwb], rwb, MPI_CHAR, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank % 2 == 0)
        if (rank != 0) MPI_Sendrecv(&loc[rwb], rwb, MPI_CHAR, rank - 1, 0, loc, rwb, MPI_CHAR, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


    //Do operation with the image
    //printf("Rank %d is starting\n", rank
    int left, right;
    int ls = 0, gs = 0;
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < nloc / rwb; y++) {
            for (int RGB = 0; RGB < 3; RGB++) {
                int c = (y + 1) * rwb + x * 3 + RGB;

                left = (x == 0) ? 0 : loc[c - 3];
                right = (x == w - 1) ? 0 : loc[c + 3];

                if (rank == 0 && y == 0)
                    newLoc[c] = loc[c];
                else if (rank == nprocs - 1 && y == nloc / rwb - 1)
                    newLoc[c] = loc[c];
                else
                    newLoc[c] = 0.25 * loc[c] + 0.1875 * (loc[c - rwb] + loc[c + rwb] + left + right);

                ls += newLoc[c];
            }
        }
    }

    MPI_Gatherv(&newLoc[rwb], nloc, MPI_CHAR, data, counts, displs, MPI_CHAR, 0, topo_comm);
    MPI_Reduce(&ls, &gs, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("The sum gotten was %d\n", gs);

        char output[] = "output.bmp";
        FILE* secondFile = fopen(output, "wb");

        fwrite(&fh.bfType, 2, 1, secondFile);
        fwrite(&fh.bfSize, 4, 1, secondFile);
        fwrite(&fh.bfReserved1, 2, 1, secondFile);
        fwrite(&fh.bfReserved2, 2, 1, secondFile);
        fwrite(&fh.bfOffBits, 4, 1, secondFile);

        fwrite(&fih, sizeof(fih), 1, secondFile);

        fwrite(data, fih.biSizeImage, 1, secondFile);

        fclose(secondFile);
    }

    //Conclude Program
    free(loc); free(newLoc);
    if (rank == 0) { free(data); free(counts); free(displs); }
    MPI_Finalize();
    return 0;
}
