/* https://developer.arm.com/architectures/instruction-sets/intrinsics
Euler solver using low level SVE intrinsics for computation on next
generation ARM architectures. In this case, we are targetting AWS's Graviton4 processor.
This is the base C code equivalent to the SVE code. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
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
float *Left_df0, *Left_df1, *Left_df2, *Left_df3;
float *Right_df0, *Right_df1, *Right_df2, *Right_df3;
float *Bottom_df0, *Bottom_df1, *Bottom_df2, *Bottom_df3;
float *Top_df0, *Top_df1, *Top_df2, *Top_df3;
float *a;
float *dfp0, *dfp1, *dfp2, *dfp3;
float *dfm0, *dfm1, *dfm2, *dfm3;
float *dhp0, *dhp1, *dhp2, *dhp3;
float *dhm0, *dhm1, *dhm2, *dhm3;
const int NX = 1024;
const int NY = 128;
const int N = NX*NY;
const float R = 1.0;
const float GAMMA= 1.4;
const float CV = R/(GAMMA-1.0);
const float L = 1.0;
const float H = 1.0;
const float DX = L/NX;
const float DY = H/NY;
const float CFL = 0.25;
float DT;
float DT_ON_DX, DT_ON_DY;
const float TOTAL_TIME = 0.2;
float alpha = 1.25;

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

    posix_memalign((void**)&dfp0, alignment, N*sizeof(float));
    posix_memalign((void**)&dfp1, alignment, N*sizeof(float));
    posix_memalign((void**)&dfp2, alignment, N*sizeof(float));
    posix_memalign((void**)&dfp3, alignment, N*sizeof(float));
    posix_memalign((void**)&dfm0, alignment, N*sizeof(float));
    posix_memalign((void**)&dfm1, alignment, N*sizeof(float));
    posix_memalign((void**)&dfm2, alignment, N*sizeof(float));
    posix_memalign((void**)&dfm3, alignment, N*sizeof(float));
    posix_memalign((void**)&dhp0, alignment, N*sizeof(float));
    posix_memalign((void**)&dhp1, alignment, N*sizeof(float));
    posix_memalign((void**)&dhp2, alignment, N*sizeof(float));
    posix_memalign((void**)&dhp3, alignment, N*sizeof(float));
    posix_memalign((void**)&dhm0, alignment, N*sizeof(float));
    posix_memalign((void**)&dhm1, alignment, N*sizeof(float));
    posix_memalign((void**)&dhm2, alignment, N*sizeof(float));
    posix_memalign((void**)&dhm3, alignment, N*sizeof(float));
    posix_memalign((void**)&Left_df0, alignment, N*sizeof(float));
    posix_memalign((void**)&Left_df1, alignment, N*sizeof(float));
    posix_memalign((void**)&Left_df2, alignment, N*sizeof(float));
    posix_memalign((void**)&Left_df3, alignment, N*sizeof(float));
    posix_memalign((void**)&Right_df0, alignment, N*sizeof(float));
    posix_memalign((void**)&Right_df1, alignment, N*sizeof(float));
    posix_memalign((void**)&Right_df2, alignment, N*sizeof(float));
    posix_memalign((void**)&Right_df3, alignment, N*sizeof(float));
    posix_memalign((void**)&Bottom_df0, alignment, N*sizeof(float));
    posix_memalign((void**)&Bottom_df1, alignment, N*sizeof(float));
    posix_memalign((void**)&Bottom_df2, alignment, N*sizeof(float));
    posix_memalign((void**)&Bottom_df3, alignment, N*sizeof(float));
    posix_memalign((void**)&Top_df0, alignment, N*sizeof(float));
    posix_memalign((void**)&Top_df1, alignment, N*sizeof(float));
    posix_memalign((void**)&Top_df2, alignment, N*sizeof(float));
    posix_memalign((void**)&Top_df3, alignment, N*sizeof(float));

    // Set up the problem - Euler 4 shocks problem
    cell_index = 0;
    for (i = 0; i < NX; i++) {
        for (j = 0; j < NY; j++) {
            if (i < 0.5*NX) {
                p0[cell_index] = 10.0; p1[cell_index] = 0.0; p2[cell_index] = 0.0; p3[cell_index] = 1.0;
            } else {
                p0[cell_index] = 1.0; p1[cell_index] = 0.0; p2[cell_index] = 0.0; p3[cell_index] = 1.0;                
            }
            cell_index++;
        }
    }

    // Configuration 6
    /*
    for (i = 0; i < NX; i++) {
        for (j = 0; j < NY; j++) {
            if ((i < 0.5*NX) && (j < 0.5*NY)) {
                // Region 3 ok
                p0[cell_index] = 1.0; p1[cell_index] = -0.75; p2[cell_index] = 0.5; p3[cell_index] = (1.0/(p0[cell_index]*R));
            } else if ((i > 0.5*NX) && (j < 0.5*NY)) {
                // Region 4 ok
                p0[cell_index] = 3.0; p1[cell_index] = -0.75; p2[cell_index] = -0.5; p3[cell_index] = (1.0/(p0[cell_index]*R));
            } else  if ((i < 0.5*NX) && (j  > 0.5*NY)) {
                // Region 2
                p0[cell_index] = 2.0; p1[cell_index] = 0.75; p2[cell_index] = 0.5; p3[cell_index] = (1.0/(p0[cell_index]*R));
                // Region 1 ok
            } else {
                p0[cell_index] = 1.0; p1[cell_index] = 0.75; p2[cell_index] = -0.5; p3[cell_index] = (1.0/(p0[cell_index]*R));                
            }
            cell_index++;
        }
    }
    */    


    // 4 shocks
    /*
    for (i = 0; i < NX; i++) {
        for (j = 0; j < NY; j++) {
            if ((i < 0.75*NX) && (j < 0.75*NY)) {
                // Region 3 ok
                p0[cell_index] = 0.138; p1[cell_index] = 1.206; p2[cell_index] = 1.206; p3[cell_index] = (0.029/(p0[cell_index]*R));
            } else if ((i > 0.75*NX) && (j < 0.75*NY)) {
                // Region 4 ok
                p0[cell_index] = 0.5323; p1[cell_index] = 0.0; p2[cell_index] = 1.206; p3[cell_index] = (0.3/(p0[cell_index]*R));
            } else  if ((i < 0.75*NX) && (j  > 0.75*NY)) {
                // Region 2 ok
                p0[cell_index] = 0.5323; p1[cell_index] = 1.206; p2[cell_index] = 0.0; p3[cell_index] = (0.3/(p0[cell_index]*R));
                // Region 1 ok
            } else {
                p0[cell_index] = 1.5; p1[cell_index] = 0.0; p2[cell_index] = 0.0; p3[cell_index] = (1.5/(p0[cell_index]*R));                
            }
            cell_index++;
        }
    }
    */
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
    free(dfp0); free(dfp1); free(dfp2); free(dfp3);
    free(dfm0); free(dfm1); free(dfm2); free(dfm3);
    free(dhp0); free(dhp1); free(dhp2); free(dhp3);
    free(dhm0); free(dhm1); free(dhm2); free(dhm3);
    free(Left_df0); free(Left_df1); free(Left_df2); free(Left_df3);
    free(Right_df0); free(Right_df1); free(Right_df2); free(Right_df3);
    free(Bottom_df0); free(Bottom_df1); free(Bottom_df2); free(Bottom_df3);
    free(Top_df0); free(Top_df1); free(Top_df2); free(Top_df3);
}

void Compute_U_from_P() {
    #pragma omp for
    for (int cell = 0; cell < N; cell++) {
        u0[cell] = p0[cell];
        u1[cell] = p0[cell]*p1[cell];
        u2[cell] = p0[cell]*p2[cell];
        u3[cell] = p0[cell]*(p3[cell]*CV + 0.5*(p1[cell]*p1[cell] + p2[cell]*p2[cell]));
    }
    // Update DT based on desired CFL
    // Estimated CFL = ((R + 1)*DT)/DX;
    #pragma omp single
    {
        DT = (CFL/(R+1))*DX;
        DT_ON_DX = (CFL/(R+1));
        DT_ON_DY = DT/DY;
    }   
}

float minmod(float left, float right) {
    // First order
    // return 0.0;
    if (left*right < 0.0) {
        return 0.0;
    } else {
        if (fabs(left) < fabs(right)) {
            return left;
        } else {
            return right;
        }
    }

}

void Update_U_from_F() {
    // We shall break this down into two parts
    // i) Compute the cell left and right fluxes using serial computation, and then
    // ii) Compute the update to U based on these using SVE

    #pragma omp for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int index = i*NY + j;
            // X boundary conditions
            // These need to be corrected
            if (i == 0) {
                // All boundaries are outflows
                Left_f0[index] = fp0[index];
                Left_f1[index] = fp1[index];
                Left_f2[index] = fp2[index];
                Left_f3[index] = fp3[index];
                // Compute Right contribution using fm(index+1)
                Right_f0[index] = fm0[index+NY];
                Right_f1[index] = fm1[index+NY];
                Right_f2[index] = fm2[index+NY];
                Right_f3[index] = fm3[index+NY];

                // Use first order here
                dfp0[index] = 0.0;
                dfp1[index] = 0.0;
                dfp2[index] = 0.0;
                dfp3[index] = 0.0;
                dfm0[index] = 0.0;
                dfm1[index] = 0.0;
                dfm2[index] = 0.0;
                dfm3[index] = 0.0;

            } else if (i == (NX - 1)) {
                // Right end reflective boundary condition
                Left_f0[index] = fp0[index-NY];
                Left_f1[index] = fp1[index-NY];
                Left_f2[index] = fp2[index-NY];
                Left_f3[index] = fp3[index-NY];
                // Outflow
                Right_f0[index] = fm0[index];
                Right_f1[index] = fm1[index];
                Right_f2[index] = fm2[index];
                Right_f3[index] = fm3[index];

                // Use first order here
                dfp0[index] = 0.0;
                dfp1[index] = 0.0;
                dfp2[index] = 0.0;
                dfp3[index] = 0.0;
                dfm0[index] = 0.0;
                dfm1[index] = 0.0;
                dfm2[index] = 0.0;
                dfm3[index] = 0.0;

            } else {
                Left_f0[index] = fp0[index-NY];
                Left_f1[index] = fp1[index-NY];
                Left_f2[index] = fp2[index-NY];
                Left_f3[index] = fp3[index-NY];
                Right_f0[index] = fm0[index+NY];
                Right_f1[index] = fm1[index+NY];
                Right_f2[index] = fm2[index+NY];
                Right_f3[index] = fm3[index+NY];

                // Apply MC limiter
                dfp0[index] = minmod(0.5*(fp0[index+NY] - fp0[index-NY]), alpha*minmod(fp0[index] - fp0[index-NY], fp0[index+NY] - fp0[index]));
                dfp1[index] = minmod(0.5*(fp1[index+NY] - fp1[index-NY]), alpha*minmod(fp1[index] - fp1[index-NY], fp1[index+NY] - fp1[index]));
                dfp2[index] = minmod(0.5*(fp2[index+NY] - fp2[index-NY]), alpha*minmod(fp2[index] - fp2[index-NY], fp2[index+NY] - fp2[index]));
                dfp3[index] = minmod(0.5*(fp3[index+NY] - fp3[index-NY]), alpha*minmod(fp3[index] - fp3[index-NY], fp3[index+NY] - fp3[index]));

                dfm0[index] = minmod(0.5*(fm0[index+NY] - fm0[index-NY]), alpha*minmod(fm0[index] - fm0[index-NY], fm0[index+NY] - fm0[index]));
                dfm1[index] = minmod(0.5*(fm1[index+NY] - fm1[index-NY]), alpha*minmod(fm1[index] - fm1[index-NY], fm1[index+NY] - fm1[index]));
                dfm2[index] = minmod(0.5*(fm2[index+NY] - fm2[index-NY]), alpha*minmod(fm2[index] - fm2[index-NY], fm2[index+NY] - fm2[index]));
                dfm3[index] = minmod(0.5*(fm3[index+NY] - fm3[index-NY]), alpha*minmod(fm3[index] - fm3[index-NY], fm3[index+NY] - fm3[index]));
            }

            // Y boundary condition
            if (j == 0) {
                // Outflow boundary condition
                Bottom_f0[index] = hp0[index];
                Bottom_f1[index] = hp1[index];
                Bottom_f2[index] = hp2[index];
                Bottom_f3[index] = hp3[index];
                // Grab the Right contribution
                Top_f0[index] = hm0[index+1];
                Top_f1[index] = hm1[index+1];
                Top_f2[index] = hm2[index+1];
                Top_f3[index] = hm3[index+1];

                // Use first order here
                dhp0[index] = 0.0;
                dhp1[index] = 0.0;
                dhp2[index] = 0.0;
                dhp3[index] = 0.0;
                dhm0[index] = 0.0;
                dhm1[index] = 0.0;
                dhm2[index] = 0.0;
                dhm3[index] = 0.0;

            } else if (j == (NY - 1)) {
                // Bottom is fine
                Bottom_f0[index] = hp0[index-1];
                Bottom_f1[index] = hp1[index-1];
                Bottom_f2[index] = hp2[index-1];
                Bottom_f3[index] = hp3[index-1];
                // Outflow
                Top_f0[index] = hm0[index];
                Top_f1[index] = hm1[index];
                Top_f2[index] = hm2[index];
                Top_f3[index] = hm3[index];

                // Use first order here
                dhp0[index] = 0.0;
                dhp1[index] = 0.0;
                dhp2[index] = 0.0;
                dhp3[index] = 0.0;
                dhm0[index] = 0.0;
                dhm1[index] = 0.0;
                dhm2[index] = 0.0;
                dhm3[index] = 0.0;

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

                // Use first order here
                dhp0[index] = minmod(0.5*(hp0[index+1] - hp0[index-1]), alpha*minmod(hp0[index] - hp0[index-1], hp0[index+1] - hp0[index]));
                dhp1[index] = minmod(0.5*(hp1[index+1] - hp1[index-1]), alpha*minmod(hp1[index] - hp1[index-1], hp1[index+1] - hp1[index]));
                dhp2[index] = minmod(0.5*(hp2[index+1] - hp2[index-1]), alpha*minmod(hp2[index] - hp2[index-1], hp2[index+1] - hp2[index]));
                dhp3[index] = minmod(0.5*(hp3[index+1] - hp3[index-1]), alpha*minmod(hp3[index] - hp3[index-1], hp3[index+1] - hp3[index]));
                dhm0[index] = minmod(0.5*(hm0[index+1] - hm0[index-1]), alpha*minmod(hm0[index] - hm0[index-1], hm0[index+1] - hm0[index]));
                dhm1[index] = minmod(0.5*(hm1[index+1] - hm1[index-1]), alpha*minmod(hm1[index] - hm1[index-1], hm1[index+1] - hm1[index]));
                dhm2[index] = minmod(0.5*(hm2[index+1] - hm2[index-1]), alpha*minmod(hm2[index] - hm2[index-1], hm2[index+1] - hm2[index]));
                dhm3[index] = minmod(0.5*(hm3[index+1] - hm3[index-1]), alpha*minmod(hm3[index] - hm3[index-1], hm3[index+1] - hm3[index]));

            }

            // Increment the index
            index++;
        }
    }

    // This is slow, but perhaps required
    // Now we have the df's computed, we can compute Neighbour dF values

    // printf("Updating stencil (neighbour values)\n");
    #pragma omp for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int index = i*NY + j;
            // X boundary conditions
            // These need to be corrected
            if (i == 0) {
                // Left end boundary condition - reflective in x direction
                Left_df0[index] = 0.0;
                Left_df1[index] = 0.0;
                Left_df2[index] = 0.0;
                Left_df3[index] = 0.0;
                // Compute Right contribution using fm(index+1)
                Right_df0[index] = dfm0[index+NY];
                Right_df1[index] = dfm1[index+NY];
                Right_df2[index] = dfm2[index+NY];
                Right_df3[index] = dfm3[index+NY];
            } else if (i == (NX - 1)) {
                // Right end reflective boundary condition
                Left_df0[index] = dfp0[index-NY];
                Left_df1[index] = dfp1[index-NY];
                Left_df2[index] = dfp2[index-NY];
                Left_df3[index] = dfp3[index-NY];
                // Reflective conditions for Right
                Right_df0[index] = 0.0;
                Right_df1[index] = 0.0;
                Right_df2[index] = 0.0;
                Right_df3[index] = 0.0;
            } else {
                Left_df0[index] = dfp0[index-NY];
                Left_df1[index] = dfp1[index-NY];
                Left_df2[index] = dfp2[index-NY];
                Left_df3[index] = dfp3[index-NY];
                Right_df0[index] = dfm0[index+NY];
                Right_df1[index] = dfm1[index+NY];
                Right_df2[index] = dfm2[index+NY];
                Right_df3[index] = dfm3[index+NY];
            }

            // Y boundary condition
            if (j == 0) {
                // Bottom Boundary conditions are reflected in the y direction
                Bottom_df0[index] = 0.0;
                Bottom_df1[index] = 0.0;
                Bottom_df2[index] = 0.0;
                Bottom_df3[index] = 0.0;
                // Grab the Right contribution
                Top_df0[index] = dhm0[index+1];
                Top_df1[index] = dhm1[index+1];
                Top_df2[index] = dhm2[index+1];
                Top_df3[index] = dhm3[index+1];
            } else if (j == (NY - 1)) {
                // Bottom is fine
                Bottom_df0[index] = dhp0[index-1];
                Bottom_df1[index] = dhp1[index-1];
                Bottom_df2[index] = dhp2[index-1];
                Bottom_df3[index] = dhp3[index-1];
                // Top is reflected in y direction
                Top_df0[index] = 0.0;
                Top_df1[index] = 0.0;
                Top_df2[index] = 0.0;
                Top_df3[index] = 0.0;
            } else {
                // Internal cell in y direction
                Bottom_df0[index] = dhp0[index-1];
                Bottom_df1[index] = dhp1[index-1];
                Bottom_df2[index] = dhp2[index-1];
                Bottom_df3[index] = dhp3[index-1];
                Top_df0[index] = dhm0[index+1];
                Top_df1[index] = dhm1[index+1];
                Top_df2[index] = dhm2[index+1];
                Top_df3[index] = dhm3[index+1];
            }

            // Increment the index
            index++;
        }
    }    


    // Update the state
    // dU = dU - PHI*(FP - FM + FR - FL)
    #pragma omp for
    for (int index = 0; index < N; index++) {    
        // Update X Contributions        
        u0[index] = u0[index] - DT_ON_DX*(fp0[index] - fm0[index] + Right_f0[index] - Left_f0[index]);
        u1[index] = u1[index] - DT_ON_DX*(fp1[index] - fm1[index] + Right_f1[index] - Left_f1[index]);
        u2[index] = u2[index] - DT_ON_DX*(fp2[index] - fm2[index] + Right_f2[index] - Left_f2[index]);
        u3[index] = u3[index] - DT_ON_DX*(fp3[index] - fm3[index] + Right_f3[index] - Left_f3[index]);
        // Update u (2nd order)
        u0[index] = u0[index] - 0.5*DT_ON_DX*(dfp0[index] + dfm0[index] - Right_df0[index] - Left_df0[index]);
        u1[index] = u1[index] - 0.5*DT_ON_DX*(dfp1[index] + dfm1[index] - Right_df1[index] - Left_df1[index]);
        u2[index] = u2[index] - 0.5*DT_ON_DX*(dfp2[index] + dfm2[index] - Right_df2[index] - Left_df2[index]);
        u3[index] = u3[index] - 0.5*DT_ON_DX*(dfp3[index] + dfm3[index] - Right_df3[index] - Left_df3[index]);

        // Update Y Contributions
        u0[index] = u0[index] - DT_ON_DY*(hp0[index] - hm0[index] + Top_f0[index] - Bottom_f0[index]);
        u1[index] = u1[index] - DT_ON_DY*(hp1[index] - hm1[index] + Top_f1[index] - Bottom_f1[index]);
        u2[index] = u2[index] - DT_ON_DY*(hp2[index] - hm2[index] + Top_f2[index] - Bottom_f2[index]);
        u3[index] = u3[index] - DT_ON_DY*(hp3[index] - hm3[index] + Top_f3[index] - Bottom_f3[index]);
        // Update u (2nd order)
        u0[index] = u0[index] - 0.5*DT_ON_DY*(dhp0[index] + dhm0[index] - Top_df0[index] - Bottom_df0[index]);
        u1[index] = u1[index] - 0.5*DT_ON_DY*(dhp1[index] + dhm1[index] - Top_df1[index] - Bottom_df1[index]);
        u2[index] = u2[index] - 0.5*DT_ON_DY*(dhp2[index] + dhm2[index] - Top_df2[index] - Bottom_df2[index]);
        u3[index] = u3[index] - 0.5*DT_ON_DY*(dhp3[index] + dhm3[index] - Top_df3[index] - Bottom_df3[index]);
    }
}

void Compute_F_from_P() {
    #pragma omp for
    for (int cell = 0; cell < N; cell++) {
        float Z1, Z2, Z3, M, P;
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
    // printf("Iterating over vectors of length %d, performing %d iterations\n", no_bytes, no_vectors);
    #pragma omp for
    for (int cell = 0; cell < N; cell++) {
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
    // Allocate
    Allocate_and_Init_Memory();

    omp_set_num_threads(24);

    #pragma omp parallel
    {
        int NO_STEPS = 0;
        int tid = omp_get_thread_num();
        // Compute U from P
        Compute_U_from_P();
        Compute_P_from_U();

        // Take some timesteps
        float time = 0.0;
        while (time < TOTAL_TIME) {

            // printf("Thread %d in step %d\n", tid, NO_STEPS);
            // Compute split fluxes (Fp, Fm) from primitives P (i.e. density, temperature etc)
            Compute_F_from_P();
            #pragma omp barrier
            // Update conserved quantities U based on fluxes of conserved quantities
            Update_U_from_F();
            #pragma omp barrier
            // Update primitives based on conserved quantities (i.e. energy to temperature)
            Compute_P_from_U();
            #pragma omp barrier
            // Increment time

            time += DT;
            NO_STEPS += 1;

            #pragma omp barrier
        }

        printf("Completed in %d steps\n", NO_STEPS);
    }
    Save_Results();

    // Free
    Free_Memory();
}

