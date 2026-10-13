/* Test code I wrote quickly for Gary */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
float *p0, *p1, *p2, *u0, *u1, *u2;
float *f0, *f1, *f2;  // Fluxes
float *h0, *h1, *h2;  // Fluxes
float *fp0, *fp1, *fp2;  // Split flux (positive)
float *fm0, *fm1, *fm2;  // Split flux (minus [negative])
float *hp0, *hp1, *hp2;  // Split flux (positive)
float *hm0, *hm1, *hm2;  // Split flux (minus [negative])
float *Left_f0, *Left_f1, *Left_f2;
float *Right_f0, *Right_f1, *Right_f2;
float *Bottom_f0, *Bottom_f1, *Bottom_f2;
float *Top_f0, *Top_f1, *Top_f2;
float *a;  // Wave propagation speed
const int NX = 512;
const int NY = 512;
const int N = NX*NY;
const float G = 9.81;
const float L = 200.0;
const float H = 200.0;
const float DX = L/NX;
const float DY = H/NY;
const float CFL = 0.25;
float DT;
float DT_ON_DX, DT_ON_DY;
int NO_STEPS = 0;
const float TOTAL_TIME = 7.2; // Seconds for water

void Allocate_and_Init_Memory() {
    size_t alignment = 32; int i, j, cell_index;
    posix_memalign((void**)&p0, alignment, N*sizeof(float));
    posix_memalign((void**)&p1, alignment, N*sizeof(float));
    posix_memalign((void**)&p2, alignment, N*sizeof(float));
    posix_memalign((void**)&u0, alignment, N*sizeof(float));
    posix_memalign((void**)&u1, alignment, N*sizeof(float));
    posix_memalign((void**)&u2, alignment, N*sizeof(float));
    posix_memalign((void**)&f0, alignment, N*sizeof(float));
    posix_memalign((void**)&f1, alignment, N*sizeof(float));
    posix_memalign((void**)&f2, alignment, N*sizeof(float));
    posix_memalign((void**)&h0, alignment, N*sizeof(float));
    posix_memalign((void**)&h1, alignment, N*sizeof(float));
    posix_memalign((void**)&h2, alignment, N*sizeof(float));
    posix_memalign((void**)&fp0, alignment, N*sizeof(float));
    posix_memalign((void**)&fp1, alignment, N*sizeof(float));
    posix_memalign((void**)&fp2, alignment, N*sizeof(float));
    posix_memalign((void**)&fm0, alignment, N*sizeof(float));
    posix_memalign((void**)&fm1, alignment, N*sizeof(float));
    posix_memalign((void**)&fm2, alignment, N*sizeof(float));

    posix_memalign((void**)&hp0, alignment, N*sizeof(float));
    posix_memalign((void**)&hp1, alignment, N*sizeof(float));
    posix_memalign((void**)&hp2, alignment, N*sizeof(float));
    posix_memalign((void**)&hm0, alignment, N*sizeof(float));
    posix_memalign((void**)&hm1, alignment, N*sizeof(float));
    posix_memalign((void**)&hm2, alignment, N*sizeof(float));
    posix_memalign((void**)&Left_f0, alignment, N*sizeof(float));
    posix_memalign((void**)&Left_f1, alignment, N*sizeof(float));
    posix_memalign((void**)&Left_f2, alignment, N*sizeof(float));
    posix_memalign((void**)&Right_f0, alignment, N*sizeof(float));
    posix_memalign((void**)&Right_f1, alignment, N*sizeof(float));
    posix_memalign((void**)&Right_f2, alignment, N*sizeof(float));

    posix_memalign((void**)&Bottom_f0, alignment, N*sizeof(float));
    posix_memalign((void**)&Bottom_f1, alignment, N*sizeof(float));
    posix_memalign((void**)&Bottom_f2, alignment, N*sizeof(float));
    posix_memalign((void**)&Top_f0, alignment, N*sizeof(float));
    posix_memalign((void**)&Top_f1, alignment, N*sizeof(float));
    posix_memalign((void**)&Top_f2, alignment, N*sizeof(float));

    posix_memalign((void**)&a, alignment, N*sizeof(float));

    // Set up the problem - 2D dam break
    cell_index = 0;
    for (i = 0; i < NX; i++) {
        for (j = 0; j < NY; j++) {
            if (i > 0.5*NX) {
                p0[cell_index] = 10.0; p1[cell_index] = 0.0; p2[cell_index] = 0.0;
            } else {
                p0[cell_index] = 5.0; p1[cell_index] = 0.0; p2[cell_index] = 0.0;
            }
            cell_index++;
        }
    }
}


void Free_Memory() {
    free(p0); free(p1); free(p2);
    free(u0); free(u1); free(u2);
    free(f0); free(f1); free(f2);
    free(h0); free(h1); free(h2);
    free(fp0); free(fp1); free(fp2);
    free(fm0); free(fm1); free(fm2);
    free(hp0); free(hp1); free(hp2);
    free(hm0); free(hm1); free(hm2);
    free(a);
    free(Left_f0); free(Left_f1); free(Left_f2); 
    free(Right_f0); free(Right_f1); free(Right_f2); 
    free(Bottom_f0); free(Bottom_f1); free(Bottom_f2);
    free(Top_f0); free(Top_f1); free(Top_f2); 
}

void Compute_U_from_P() {
    int cell;
    for (cell = 0; cell < N; cell++) {
        u0[cell] = p0[cell];
        u1[cell] = p0[cell]*p1[cell];
        u2[cell] = p0[cell]*p2[cell];
    }
    // Update DT based on desired CFL
    DT = (CFL/(20.0))*DX; // Estimated!
    DT_ON_DX = DT/DX;
    DT_ON_DY = DT/DY;   
}

void Update_U_from_F() {
    int index, i, j;
    // We shall break this down into two parts
    // i) Compute the cell left and right fluxes using serial computation, and then
    // ii) Compute the update to U based on these using SVE
    index = 0;
    // printf("Updating stencil (neighbour values)\n");
    for (i = 0; i < NX; i++) {
        for (j = 0; j < NY; j++) {
            // X boundary conditions
            // These need to be corrected
            if (i == 0) {
                // Left end boundary condition - reflective in x direction
                Left_f0[index] = -fm0[index];
                Left_f1[index] = fm1[index];
                Left_f2[index] = -fm2[index];
                // Compute Right contribution using fm(index+1)
                Right_f0[index] = fm0[index+NY];
                Right_f1[index] = fm1[index+NY];
                Right_f2[index] = fm2[index+NY];
            } else if (i == (NX - 1)) {
                // Right end reflective boundary condition
                Left_f0[index] = fp0[index-NY];
                Left_f1[index] = fp1[index-NY];
                Left_f2[index] = fp2[index-NY];
                // Reflective conditions for Right
                Right_f0[index] = -fp0[index];
                Right_f1[index] = fp1[index];
                Right_f2[index] = -fp2[index];
            } else {
                // Now we need to check if there is a wall before we apply reflective conditions
                // This only applies in the X direction!
                int wall_cell = (int)(0.5*NX);
                // Need to check left of the wall
                if ((i == wall_cell) && ((j < 0.48*NY) || (j > 0.85*NY))) {
                    // Right end reflective boundary condition
                    Left_f0[index] = fp0[index-NY];
                    Left_f1[index] = fp1[index-NY];
                    Left_f2[index] = fp2[index-NY];
                    // Reflective conditions for Right
                    Right_f0[index] = -fp0[index];
                    Right_f1[index] = fp1[index];
                    Right_f2[index] = -fp2[index];
                } else if ((i == (wall_cell+1)) && ((j < 0.48*NY) || (j > 0.85*NY))) {
                    Left_f0[index] = -fm0[index];
                    Left_f1[index] = fm1[index];
                    Left_f2[index] = -fm2[index];
                    // Compute Right contribution using fm(index+1)
                    Right_f0[index] = fm0[index+NY];
                    Right_f1[index] = fm1[index+NY];
                    Right_f2[index] = fm2[index+NY];               
                } else {
                    Left_f0[index] = fp0[index-NY];
                    Left_f1[index] = fp1[index-NY];
                    Left_f2[index] = fp2[index-NY];
                    Right_f0[index] = fm0[index+NY];
                    Right_f1[index] = fm1[index+NY];
                    Right_f2[index] = fm2[index+NY];
                }
            }

            // Y boundary condition
            if (j == 0) {
                // Bottom Boundary conditions are reflected in the y direction
                Bottom_f0[index] = -hm0[index];
                Bottom_f1[index] = -hm1[index];
                Bottom_f2[index] = hm2[index];
                // Grab the Right contribution
                Top_f0[index] = hm0[index+1];
                Top_f1[index] = hm1[index+1];
                Top_f2[index] = hm2[index+1];
            } else if (j == (NY - 1)) {
                // Bottom is fine
                Bottom_f0[index] = hp0[index-1];
                Bottom_f1[index] = hp1[index-1];
                Bottom_f2[index] = hp2[index-1];
                // Top is reflected in y direction
                Top_f0[index] = -hp0[index];
                Top_f1[index] = -hp1[index];
                Top_f2[index] = hp2[index];
            } else {
                // Internal cell in y direction
                Bottom_f0[index] = hp0[index-1];
                Bottom_f1[index] = hp1[index-1];
                Bottom_f2[index] = hp2[index-1];
                Top_f0[index] = hm0[index+1];
                Top_f1[index] = hm1[index+1];
                Top_f2[index] = hm2[index+1];
            }

            // Increment the index
            index++;
        }
    }

    // Update the state
    // dU = dU - PHI*(FP - FM + FR - FL)
    for (index = 0; index < N; index++) {    
        // Update X Contributions
        u0[index] = u0[index] - DT_ON_DX*(fp0[index] - fm0[index] + Right_f0[index] - Left_f0[index]);
        u1[index] = u1[index] - DT_ON_DX*(fp1[index] - fm1[index] + Right_f1[index] - Left_f1[index]);
        u2[index] = u2[index] - DT_ON_DX*(fp2[index] - fm2[index] + Right_f2[index] - Left_f2[index]);
        // Update Y Contributions
        u0[index] = u0[index] - DT_ON_DY*(hp0[index] - hm0[index] + Top_f0[index] - Bottom_f0[index]);
        u1[index] = u1[index] - DT_ON_DY*(hp1[index] - hm1[index] + Top_f1[index] - Bottom_f1[index]);
        u2[index] = u2[index] - DT_ON_DY*(hp2[index] - hm2[index] + Top_f2[index] - Bottom_f2[index]);
    }
}

void Compute_F_from_P() {
    int cell;
    float Z1, Z2, Z3, M, P;

    for (cell = 0; cell < N; cell++) {

        /* X Direction */
        // This isn't the mach number, but it is the critical number (Froud number)
        M = p1[cell]/a[cell];
        // Z invariants
        Z1 = 0.5*(M + 1.0);
        Z2 = 0.5*a[cell]*(1.0-M*M);
        Z3 = 0.5*(M - 1.0);
        // Fluxes of conserved quantities
        f0[cell] = u1[cell];
        f1[cell] = u1[cell]*p1[cell] + 0.5*G*p0[cell]*p0[cell];
        f2[cell] = u1[cell]*p2[cell];

        // Split fluxes - positive
        // FP[:,:,0] = F[:,:,0]*Z1 + U[:,:,0]*Z2
        fp0[cell] = f0[cell]*Z1 + u0[cell]*Z2;
        fp1[cell] = f1[cell]*Z1 + u1[cell]*Z2;
        fp2[cell] = f2[cell]*Z1 + u2[cell]*Z2;

        // Split fluxes - negative
        // FM[:,:,0] = -F[:,:,0]*Z3 - U[:,:,0]*Z2
        fm0[cell] = -f0[cell]*Z3 - u0[cell]*Z2;
        fm1[cell] = -f1[cell]*Z3 - u1[cell]*Z2;
        fm2[cell] = -f2[cell]*Z3 - u2[cell]*Z2;

        /* Y Direction */
        M = p2[cell]/a[cell];
        // Z invariants
        Z1 = 0.5*(M + 1.0);
        Z2 = 0.5*a[cell]*(1.0-M*M);
        Z3 = 0.5*(M - 1.0);
        // Fluxes of conserved quantities
        h0[cell] = u2[cell];
        h1[cell] = u2[cell]*p1[cell];
        h2[cell] = u2[cell]*p2[cell] + 0.5*G*p0[cell]*p0[cell];

        // Split fluxes - positive
        // FP[:,:,0] = F[:,:,0]*Z1 + U[:,:,0]*Z2
        hp0[cell] = h0[cell]*Z1 + u0[cell]*Z2;
        hp1[cell] = h1[cell]*Z1 + u1[cell]*Z2;
        hp2[cell] = h2[cell]*Z1 + u2[cell]*Z2;

        // Split fluxes - negative
        // FM[:,:,0] = -F[:,:,0]*Z3 - U[:,:,0]*Z2
        hm0[cell] = -h0[cell]*Z3 - u0[cell]*Z2;
        hm1[cell] = -h1[cell]*Z3 - u1[cell]*Z2;
        hm2[cell] = -h2[cell]*Z3 - u2[cell]*Z2;
    }
}

void Compute_P_from_U() {
    int cell;
    for (cell = 0; cell < N; cell++) {
        p0[cell] = u0[cell];
        p1[cell] = u1[cell]/u0[cell];
        p2[cell] = u2[cell]/u0[cell];
        a[cell] = sqrt(G*p0[cell]);
    }
}


void Save_Results() {
    FILE *fptr;
    int i, j, index;
    float cx, cy;
    index = 0;
    printf("Saving to file\n");
    fptr = fopen("results.dat", "w");
    for (i = 0; i < NX; i++) {
        for (j = 0; j < NY; j++) {
            cx = (i+0.5)*DX;
            cy = (j+0.5)*DY;
            fprintf(fptr, "%e\t%e\t%e\t%e\t%e\n", cx, cy, p0[index], p1[index], p2[index]);
            index++;
        }
    }
    // Close the file
    fclose(fptr);
    printf("Completed saving data\n");
}


int main() {
    int i;
    float time = 0.0;
    // Allocate
    Allocate_and_Init_Memory();
    // Compute U from P
    Compute_U_from_P();
    Compute_P_from_U();

    // Take some timesteps
    while (time < TOTAL_TIME) {
        // Compute split fluxes (Fp, Fm) from primitives P (i.e. density, temperature etc)
        Compute_F_from_P();
        // Update conserved quantities U based on fluxes of conserved quantities
        Update_U_from_F();
        // Update primitives based on conserved quantities (i.e. energy to temperature)
        Compute_P_from_U();
        // Increment time
        time += DT;
        NO_STEPS += 1;
    }

    printf("Completed in %d steps\n", NO_STEPS);
    Save_Results();

    // Free
    Free_Memory();
}

