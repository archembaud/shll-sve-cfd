/* https://developer.arm.com/architectures/instruction-sets/intrinsics
Euler solver using low level SVE intrinsics for computation on next
generation ARM architectures. In this case, we are targetting AWS's Graviton4 processor.
This is the base C code equivalent to the SVE code. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
float *p0, *p1, *p2, *p3, *u0, *u1, *u2, *u3;
float *f0, *f1, *f2, *f3;  // Fluxes
float *h0, *h1, *h2, *h3;  // Fluxes
float *fp0, *fp1, *fp2, *fp3;  // Split flux (positive)
float *fm0, *fm1, *fm2, *fm3;  // Split flux (minus [negative])
float *hp0, *hp1, *hp2, *hp3;  // Split flux (positive)
float *hm0, *hm1, *hm2, *hm3;  // Split flux (minus [negative])
float *Left_f0, *Left_f1, *Left_f2, *Left_f3;
float *Right_f0, *Right_f1, *Right_f2, *Right_f3;
float *Bottom_f0, *Bottom_f1, *Bottom_f2, *Bottom_f3;
float *Top_f0, *Top_f1, *Top_f2, *Top_f3;
float *a;
const int NX = 256;
const int NY = 256;
const int N = NX*NY;
const float R = 1.0;
const float GAMMA=1.4;
const float CV = R/(GAMMA-1.0);
const float L = 1.0;
const float H = 1.0;
const float DX = L/NX;
const float DY = H/NY;
const float CFL = 0.25;
float DT;
float DT_ON_DX, DT_ON_DY;
int NO_STEPS = 0;
const float TOTAL_TIME = 0.1;

void Allocate_and_Init_Memory() {
    size_t alignment = 32; int i, j, cell_index;
    posix_memalign((void**)&p0, alignment, N*sizeof(float));
    posix_memalign((void**)&p1, alignment, N*sizeof(float));
    posix_memalign((void**)&p2, alignment, N*sizeof(float));
    posix_memalign((void**)&p3, alignment, N*sizeof(float));
    posix_memalign((void**)&u0, alignment, N*sizeof(float));
    posix_memalign((void**)&u1, alignment, N*sizeof(float));
    posix_memalign((void**)&u2, alignment, N*sizeof(float));
    posix_memalign((void**)&u3, alignment, N*sizeof(float));
    posix_memalign((void**)&f0, alignment, N*sizeof(float));
    posix_memalign((void**)&f1, alignment, N*sizeof(float));
    posix_memalign((void**)&f2, alignment, N*sizeof(float));
    posix_memalign((void**)&f3, alignment, N*sizeof(float));
    posix_memalign((void**)&h0, alignment, N*sizeof(float));
    posix_memalign((void**)&h1, alignment, N*sizeof(float));
    posix_memalign((void**)&h2, alignment, N*sizeof(float));
    posix_memalign((void**)&h3, alignment, N*sizeof(float));
    posix_memalign((void**)&fp0, alignment, N*sizeof(float));
    posix_memalign((void**)&fp1, alignment, N*sizeof(float));
    posix_memalign((void**)&fp2, alignment, N*sizeof(float));
    posix_memalign((void**)&fp3, alignment, N*sizeof(float));
    posix_memalign((void**)&fm0, alignment, N*sizeof(float));
    posix_memalign((void**)&fm1, alignment, N*sizeof(float));
    posix_memalign((void**)&fm2, alignment, N*sizeof(float));
    posix_memalign((void**)&fm3, alignment, N*sizeof(float));

    posix_memalign((void**)&hp0, alignment, N*sizeof(float));
    posix_memalign((void**)&hp1, alignment, N*sizeof(float));
    posix_memalign((void**)&hp2, alignment, N*sizeof(float));
    posix_memalign((void**)&hp3, alignment, N*sizeof(float));
    posix_memalign((void**)&hm0, alignment, N*sizeof(float));
    posix_memalign((void**)&hm1, alignment, N*sizeof(float));
    posix_memalign((void**)&hm2, alignment, N*sizeof(float));
    posix_memalign((void**)&hm3, alignment, N*sizeof(float));
    posix_memalign((void**)&Left_f0, alignment, N*sizeof(float));
    posix_memalign((void**)&Left_f1, alignment, N*sizeof(float));
    posix_memalign((void**)&Left_f2, alignment, N*sizeof(float));
    posix_memalign((void**)&Left_f3, alignment, N*sizeof(float));
    posix_memalign((void**)&Right_f0, alignment, N*sizeof(float));
    posix_memalign((void**)&Right_f1, alignment, N*sizeof(float));
    posix_memalign((void**)&Right_f2, alignment, N*sizeof(float));
    posix_memalign((void**)&Right_f3, alignment, N*sizeof(float));

    posix_memalign((void**)&Bottom_f0, alignment, N*sizeof(float));
    posix_memalign((void**)&Bottom_f1, alignment, N*sizeof(float));
    posix_memalign((void**)&Bottom_f2, alignment, N*sizeof(float));
    posix_memalign((void**)&Bottom_f3, alignment, N*sizeof(float));
    posix_memalign((void**)&Top_f0, alignment, N*sizeof(float));
    posix_memalign((void**)&Top_f1, alignment, N*sizeof(float));
    posix_memalign((void**)&Top_f2, alignment, N*sizeof(float));
    posix_memalign((void**)&Top_f3, alignment, N*sizeof(float));

    posix_memalign((void**)&a, alignment, N*sizeof(float));

    // Set up the problem - 2D implosion
    cell_index = 0;
    for (i = 0; i < NX; i++) {
        for (j = 0; j < NY; j++) {
            if ((i > 0.2*NX) && (i < 0.8*NX) && (j > 0.2*NY) && (j < 0.8*NY)) {
                p0[cell_index] = 1.0; p1[cell_index] = 0.0; p2[cell_index] = 0.0; p3[cell_index] = 1.0;
            } else {
                p0[cell_index] = 10.0; p1[cell_index] = 0.0; p2[cell_index] = 0.0; p3[cell_index] = 1.0;
            }
            cell_index++;
        }
    }
    // printf("Allocation of Memory completed\n");
}


void Free_Memory() {
    free(p0); free(p1); free(p2); free(p3);
    free(u0); free(u1); free(u2); free(u3);
    free(f0); free(f1); free(f2); free(f3);
    free(h0); free(h1); free(h2); free(h3);
    free(fp0); free(fp1); free(fp2); free(fp3);
    free(fm0); free(fm1); free(fm2); free(fm3);
    free(hp0); free(hp1); free(hp2); free(hp3);
    free(hm0); free(hm1); free(hm2); free(hm3);
    free(a);
    free(Left_f0); free(Left_f1); free(Left_f2); free(Left_f3);
    free(Right_f0); free(Right_f1); free(Right_f2); free(Right_f3);
    free(Bottom_f0); free(Bottom_f1); free(Bottom_f2); free(Bottom_f3);
    free(Top_f0); free(Top_f1); free(Top_f2); free(Top_f3);
}

void Compute_U_from_P() {
    int cell;
    for (cell = 0; cell < N; cell++) {
        u0[cell] = p0[cell];
        u1[cell] = p0[cell]*p1[cell];
        u2[cell] = p0[cell]*p2[cell];
        u3[cell] = p0[cell]*(p3[cell]*CV + 0.5*(p1[cell]*p1[cell] + p2[cell]*p2[cell]));
    }
    // Update DT based on desired CFL
    // Estimated CFL = ((R + 1)*DT)/DX;
    DT = (CFL/(R+1))*DX;
    DT_ON_DX = (CFL/(R+1));
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
                Left_f3[index] = -fm3[index];
                // Compute Right contribution using fm(index+1)
                Right_f0[index] = fm0[index+NY];
                Right_f1[index] = fm1[index+NY];
                Right_f2[index] = fm2[index+NY];
                Right_f3[index] = fm3[index+NY];
            } else if (i == (NX - 1)) {
                // Right end reflective boundary condition
                Left_f0[index] = fp0[index-NY];
                Left_f1[index] = fp1[index-NY];
                Left_f2[index] = fp2[index-NY];
                Left_f3[index] = fp3[index-NY];
                // Reflective conditions for Right
                Right_f0[index] = -fp0[index];
                Right_f1[index] = fp1[index];
                Right_f2[index] = -fp2[index];
                Right_f3[index] = -fp3[index];
            } else {
                Left_f0[index] = fp0[index-NY];
                Left_f1[index] = fp1[index-NY];
                Left_f2[index] = fp2[index-NY];
                Left_f3[index] = fp3[index-NY];
                Right_f0[index] = fm0[index+NY];
                Right_f1[index] = fm1[index+NY];
                Right_f2[index] = fm2[index+NY];
                Right_f3[index] = fm3[index+NY];
            }

            // Y boundary condition
            if (j == 0) {
                // Bottom Boundary conditions are reflected in the y direction
                Bottom_f0[index] = -hm0[index];
                Bottom_f1[index] = -hm1[index];
                Bottom_f2[index] = hm2[index];
                Bottom_f3[index] = -hm3[index];
                // Grab the Right contribution
                Top_f0[index] = hm0[index+1];
                Top_f1[index] = hm1[index+1];
                Top_f2[index] = hm2[index+1];
                Top_f3[index] = hm3[index+1];
            } else if (j == (NY - 1)) {
                // Bottom is fine
                Bottom_f0[index] = hp0[index-1];
                Bottom_f1[index] = hp1[index-1];
                Bottom_f2[index] = hp2[index-1];
                Bottom_f3[index] = hp3[index-1];
                // Top is reflected in y direction
                Top_f0[index] = -hp0[index];
                Top_f1[index] = -hp1[index];
                Top_f2[index] = hp2[index];
                Top_f3[index] = -hp3[index];
            } else {
                // Internal cell in y direction
                Bottom_f0[index] = hp0[index-1];
                Bottom_f1[index] = hp1[index-1];
                Bottom_f2[index] = hp2[index-1];
                Bottom_f3[index] = hp3[index-1];
                Top_f0[index] = hm0[index+1];
                Top_f1[index] = hm1[index+1];
                Top_f2[index] = hm2[index+1];
                Top_f3[index] = hm3[index+1];
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
        u3[index] = u3[index] - DT_ON_DX*(fp3[index] - fm3[index] + Right_f3[index] - Left_f3[index]);
        // Update Y Contributions
        u0[index] = u0[index] - DT_ON_DY*(hp0[index] - hm0[index] + Top_f0[index] - Bottom_f0[index]);
        u1[index] = u1[index] - DT_ON_DY*(hp1[index] - hm1[index] + Top_f1[index] - Bottom_f1[index]);
        u2[index] = u2[index] - DT_ON_DY*(hp2[index] - hm2[index] + Top_f2[index] - Bottom_f2[index]);
        u3[index] = u3[index] - DT_ON_DY*(hp3[index] - hm3[index] + Top_f3[index] - Bottom_f3[index]);
    }
}

void Compute_F_from_P() {
    int cell;
    float Z1, Z2, Z3, M, P;

    for (cell = 0; cell < N; cell++) {

        // Pressure
        P = p0[cell]*R*p3[cell];

        /* X Direction */
        M = p1[cell]/a[cell];
        // Z invariants
        Z1 = 0.5*(M + 1.0);
        Z2 = 0.5*a[cell]*(1.0-M*M);
        Z3 = 0.5*(M - 1.0);
        // Fluxes of conserved quantities
        f0[cell] = u1[cell];
        f1[cell] = u1[cell]*p1[cell] + P;
        f2[cell] = u1[cell]*p2[cell];
        f3[cell] = p1[cell]*(u3[cell] + P);

        // Split fluxes - positive
        // FP[:,:,0] = F[:,:,0]*Z1 + U[:,:,0]*Z2
        fp0[cell] = f0[cell]*Z1 + u0[cell]*Z2;
        fp1[cell] = f1[cell]*Z1 + u1[cell]*Z2;
        fp2[cell] = f2[cell]*Z1 + u2[cell]*Z2;
        fp3[cell] = f3[cell]*Z1 + u3[cell]*Z2;

        // Split fluxes - negative
        // FM[:,:,0] = -F[:,:,0]*Z3 - U[:,:,0]*Z2
        fm0[cell] = -f0[cell]*Z3 - u0[cell]*Z2;
        fm1[cell] = -f1[cell]*Z3 - u1[cell]*Z2;
        fm2[cell] = -f2[cell]*Z3 - u2[cell]*Z2;
        fm3[cell] = -f3[cell]*Z3 - u3[cell]*Z2;

        /* Y Direction */
        M = p2[cell]/a[cell];
        // Z invariants
        Z1 = 0.5*(M + 1.0);
        Z2 = 0.5*a[cell]*(1.0-M*M);
        Z3 = 0.5*(M - 1.0);
        // Fluxes of conserved quantities
        h0[cell] = u2[cell];
        h1[cell] = u2[cell]*p1[cell];
        h2[cell] = u2[cell]*p2[cell] + P;
        h3[cell] = p2[cell]*(u3[cell] + P);

        // Split fluxes - positive
        // FP[:,:,0] = F[:,:,0]*Z1 + U[:,:,0]*Z2
        hp0[cell] = h0[cell]*Z1 + u0[cell]*Z2;
        hp1[cell] = h1[cell]*Z1 + u1[cell]*Z2;
        hp2[cell] = h2[cell]*Z1 + u2[cell]*Z2;
        hp3[cell] = h3[cell]*Z1 + u3[cell]*Z2;

        // Split fluxes - negative
        // FM[:,:,0] = -F[:,:,0]*Z3 - U[:,:,0]*Z2
        hm0[cell] = -h0[cell]*Z3 - u0[cell]*Z2;
        hm1[cell] = -h1[cell]*Z3 - u1[cell]*Z2;
        hm2[cell] = -h2[cell]*Z3 - u2[cell]*Z2;
        hm3[cell] = -h3[cell]*Z3 - u3[cell]*Z2;
    }
}

void Compute_P_from_U() {
    /*
    P[:,:,0] = U[:,:,0]   		# Water Height
    P[:,:,1] = U[:,:,1]/U[:,:,0]	# X vel
    P[:,:,2] = U[:,:,2]/U[:,:,0]	# Y vel
    P[:,:,3] = ((U[:,:,3]/U[:,:,0]) - 0.5*(P[:,:,1]*P[:,:,1]+P[:,:,2]*P[:,:,2]))/CV # Temp	
    CFL = (P[:,:,1] + 2.0*np.sqrt(GAMMA*R*P[:,:,3]))*DT/DX
    */
    int cell;
    // printf("Iterating over vectors of length %d, performing %d iterations\n", no_bytes, no_vectors);
    for (cell = 0; cell < N; cell++) {
        p0[cell] = u0[cell];
        p1[cell] = u1[cell]/u0[cell];
        p2[cell] = u2[cell]/u0[cell];
        p3[cell] = ((u3[cell]/u0[cell]) - 0.5*(p1[cell]*p1[cell] + p2[cell]*p2[cell]))/CV;
        a[cell] = sqrt(GAMMA*R*p3[cell]);
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
            fprintf(fptr, "%e\t%e\t%e\t%e\t%e\t%e\n", cx, cy, p0[index], p1[index], p2[index], p3[index]);
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
    //Save_Results();

    // Free
    Free_Memory();
}

