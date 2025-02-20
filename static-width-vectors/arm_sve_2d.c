/* https://developer.arm.com/architectures/instruction-sets/intrinsics
Euler solver using low level SVE intrinsics for computation on next
generation ARM architectures. In this case, we are targetting AWS's Graviton4 processor. */

#include <stdio.h>
#include <arm_sve.h>
#include <stdlib.h>
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
    // Compute the number of bytes
    int no_bytes = svcntb()*8; int no_vectors = (int)(N/4); // Assuming 4 elements int vector = 0;
    int vector;
    // Create SVE pointers which point to our aligned memory space
    svfloat32_t *vec_p0, *vec_p1, *vec_p2, *vec_p3, *vec_u0, *vec_u1, *vec_u2, *vec_u3;
    // Create some constants
    svfloat32_t HALF = svdup_f32(0.5); svfloat32_t ONE = svdup_f32(1.0);
    svfloat32_t CV = svdup_f32((R/(GAMMA-1.0)));
    svfloat32_t T_CV;  svfloat32_t KE; svfloat32_t KE_x; svfloat32_t KE_y; svfloat32_t T_CV_KE;
    // printf("(U-from-P) - Iterating over vectors of length %d, performing %d iterations\n", no_bytes, no_vectors);
    for (vector = 0; vector < no_vectors; vector++) {
        // Now we can point our sve float types at these temp pointers
        vec_p0 = (svfloat32_t*)((float*)p0+(vector*4)); vec_p1 = (svfloat32_t*)((float*)p1+(vector*4));
        vec_p2 = (svfloat32_t*)((float*)p2+(vector*4)); vec_p3 = (svfloat32_t*)((float*)p3+(vector*4));
        vec_u0 = (svfloat32_t*)((float*)u0+(vector*4)); vec_u1 = (svfloat32_t*)((float*)u1+(vector*4));
        vec_u2 = (svfloat32_t*)((float*)u2+(vector*4)); vec_u3 = (svfloat32_t*)((float*)u3+(vector*4));
        // Now we can start - compute U from P Euler Equations Density
        *vec_u0 = svmul_f32_m(svptrue_b32(), *vec_p0, ONE);
        // Momentum (x)
        *vec_u1 = svmul_f32_m(svptrue_b32(), *vec_p0, *vec_p1);
        // Momentum (y)
        *vec_u2 = svmul_f32_m(svptrue_b32(), *vec_p0, *vec_p2);
        // Energy (a little complex)
        KE_x = svmul_f32_m(svptrue_b32(), *vec_p1, *vec_p1);
        KE_y = svmul_f32_m(svptrue_b32(), *vec_p2, *vec_p2);
        KE = svadd_f32_m(svptrue_b32(), KE_x, KE_y);
        // Multiply by 1/2
        KE = svmul_f32_m(svptrue_b32(), KE, HALF); T_CV = svmul_f32_m(svptrue_b32(), *vec_p3, CV);
        // Add these
        T_CV_KE = svadd_f32_m(svptrue_b32(), T_CV, KE);
        *vec_u3 = svmul_f32_m(svptrue_b32(), *vec_p0, T_CV_KE);
        // 3D U[index+4] = P[index]*(P[index+4]*CV+0.5*(P[index+1]*P[index+1]+P[index+2]*P[index+2]+P[index+3]*P[index+3]));
    }

    // Update DT based on desired CFL
    // Estimated CFL = ((R + 1)*DT)/DX;
    DT = (CFL/(R+1))*DX;
    DT_ON_DX = (CFL/(R+1));
    DT_ON_DY = DT/DY;
    // printf("(U-from-P) completed\n");
}

void Update_U_from_F() {
    // Compute the number of bytes
    int no_bytes = svcntb()*8; int no_vectors = (int)(N/4); // Assuming 4 elements int vector = 0;
    int i, j;
    int vector;
    int index;
    // Create SVE pointers which point to our aligned memory space
    svfloat32_t *vec_u0, *vec_u1, *vec_u2, *vec_u3;
    svfloat32_t *vec_fp0, *vec_fp1, *vec_fp2, *vec_fp3;
    svfloat32_t *vec_fm0, *vec_fm1, *vec_fm2, *vec_fm3;
    svfloat32_t *vec_hp0, *vec_hp1, *vec_hp2, *vec_hp3;
    svfloat32_t *vec_hm0, *vec_hm1, *vec_hm2, *vec_hm3;
    svfloat32_t *vec_fR0, *vec_fR1, *vec_fR2, *vec_fR3;
    svfloat32_t *vec_fL0, *vec_fL1, *vec_fL2, *vec_fL3;
    svfloat32_t *vec_fT0, *vec_fT1, *vec_fT2, *vec_fT3;
    svfloat32_t *vec_fB0, *vec_fB1, *vec_fB2, *vec_fB3;

    // Create some constants
    // printf("Check on DT_ON_DY: %g\n", DT_ON_DY);
    svfloat32_t DT_DX = svdup_f32(DT_ON_DX);
    svfloat32_t DT_DY = svdup_f32(DT_ON_DY);
    svfloat32_t LEFT_FLUX, RIGHT_FLUX;
    svfloat32_t SUM_FLUXES;
    // printf("(U-from-F) Iterating over vectors of length %d, performing %d iterations\n", no_bytes, no_vectors);

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
    // printf("Stencil updated - now updating state\n");


    // Update the state
    // dU = dU - PHI*(FP - FM + FR - FL) - PHI*(HP - HM + TOP - BOTTOM)
    for (vector = 0; vector < no_vectors; vector++) {
        // Map SVE pointers onto their respective floating point arrays
        vec_u0 = (svfloat32_t*)((float*)u0+(vector*4)); vec_u1 = (svfloat32_t*)((float*)u1+(vector*4));
        vec_u2 = (svfloat32_t*)((float*)u2+(vector*4)); vec_u3 = (svfloat32_t*)((float*)u3+(vector*4));
        vec_fp0 = (svfloat32_t*)((float*)fp0+(vector*4)); vec_fp1 = (svfloat32_t*)((float*)fp1+(vector*4));
        vec_fp2 = (svfloat32_t*)((float*)fp2+(vector*4)); vec_fp3 = (svfloat32_t*)((float*)fp3+(vector*4));
        vec_fm0 = (svfloat32_t*)((float*)fm0+(vector*4)); vec_fm1 = (svfloat32_t*)((float*)fm1+(vector*4));
        vec_fm2 = (svfloat32_t*)((float*)fm2+(vector*4)); vec_fm3 = (svfloat32_t*)((float*)fm3+(vector*4));
        vec_hp0 = (svfloat32_t*)((float*)hp0+(vector*4)); vec_hp1 = (svfloat32_t*)((float*)hp1+(vector*4));
        vec_hp2 = (svfloat32_t*)((float*)hp2+(vector*4)); vec_hp3 = (svfloat32_t*)((float*)hp3+(vector*4));
        vec_hm0 = (svfloat32_t*)((float*)hm0+(vector*4)); vec_hm1 = (svfloat32_t*)((float*)hm1+(vector*4));
        vec_hm2 = (svfloat32_t*)((float*)hm2+(vector*4)); vec_hm3 = (svfloat32_t*)((float*)hm3+(vector*4));
        vec_fR0 = (svfloat32_t*)((float*)Right_f0+(vector*4)); vec_fR1 = (svfloat32_t*)((float*)Right_f1+(vector*4));
        vec_fR2 = (svfloat32_t*)((float*)Right_f2+(vector*4)); vec_fR3 = (svfloat32_t*)((float*)Right_f3+(vector*4));
        vec_fL0 = (svfloat32_t*)((float*)Left_f0+(vector*4)); vec_fL1 = (svfloat32_t*)((float*)Left_f1+(vector*4));
        vec_fL2 = (svfloat32_t*)((float*)Left_f2+(vector*4)); vec_fL3 = (svfloat32_t*)((float*)Left_f3+(vector*4));
        vec_fT0 = (svfloat32_t*)((float*)Top_f0+(vector*4)); vec_fT1 = (svfloat32_t*)((float*)Top_f1+(vector*4));
        vec_fT2 = (svfloat32_t*)((float*)Top_f2+(vector*4)); vec_fT3 = (svfloat32_t*)((float*)Top_f3+(vector*4));
        vec_fB0 = (svfloat32_t*)((float*)Bottom_f0+(vector*4)); vec_fB1 = (svfloat32_t*)((float*)Bottom_f1+(vector*4));
        vec_fB2 = (svfloat32_t*)((float*)Bottom_f2+(vector*4)); vec_fB3 = (svfloat32_t*)((float*)Bottom_f3+(vector*4));

        // X contribution
        // printf("Computing X contribution\n");
        SUM_FLUXES =  svsub_f32_m(svptrue_b32(), *vec_fp0, *vec_fm0);
        SUM_FLUXES =  svsub_f32_m(svptrue_b32(), SUM_FLUXES, *vec_fL0);
        SUM_FLUXES =  svadd_f32_m(svptrue_b32(), SUM_FLUXES, *vec_fR0);
        SUM_FLUXES =  svmul_f32_m(svptrue_b32(), SUM_FLUXES, DT_DX);
        *vec_u0 = svsub_f32_m(svptrue_b32(), *vec_u0, SUM_FLUXES);

        SUM_FLUXES =  svsub_f32_m(svptrue_b32(), *vec_fp1, *vec_fm1);
        SUM_FLUXES =  svsub_f32_m(svptrue_b32(), SUM_FLUXES, *vec_fL1);
        SUM_FLUXES =  svadd_f32_m(svptrue_b32(), SUM_FLUXES, *vec_fR1);
        SUM_FLUXES =  svmul_f32_m(svptrue_b32(), SUM_FLUXES, DT_DX);
        *vec_u1 = svsub_f32_m(svptrue_b32(), *vec_u1, SUM_FLUXES);

        SUM_FLUXES =  svsub_f32_m(svptrue_b32(), *vec_fp2, *vec_fm2);
        SUM_FLUXES =  svsub_f32_m(svptrue_b32(), SUM_FLUXES, *vec_fL2);
        SUM_FLUXES =  svadd_f32_m(svptrue_b32(), SUM_FLUXES, *vec_fR2);
        SUM_FLUXES =  svmul_f32_m(svptrue_b32(), SUM_FLUXES, DT_DX);
        *vec_u2 = svsub_f32_m(svptrue_b32(), *vec_u2, SUM_FLUXES);

        SUM_FLUXES =  svsub_f32_m(svptrue_b32(), *vec_fp3, *vec_fm3);
        SUM_FLUXES =  svsub_f32_m(svptrue_b32(), SUM_FLUXES, *vec_fL3);
        SUM_FLUXES =  svadd_f32_m(svptrue_b32(), SUM_FLUXES, *vec_fR3);
        SUM_FLUXES =  svmul_f32_m(svptrue_b32(), SUM_FLUXES, DT_DX);
        *vec_u3 = svsub_f32_m(svptrue_b32(), *vec_u3, SUM_FLUXES);

        // Y contribution
        // printf("Computing Y contribution\n");
        // printf("Mass\n");

        SUM_FLUXES =  svsub_f32_m(svptrue_b32(), *vec_hp0, *vec_hm0);
        SUM_FLUXES =  svsub_f32_m(svptrue_b32(), SUM_FLUXES, *vec_fB0);
        SUM_FLUXES =  svadd_f32_m(svptrue_b32(), SUM_FLUXES, *vec_fT0);
        SUM_FLUXES =  svmul_f32_m(svptrue_b32(), SUM_FLUXES, DT_DY);
        *vec_u0 = svsub_f32_m(svptrue_b32(), *vec_u0, SUM_FLUXES);

        // printf("X Mom\n");
        SUM_FLUXES =  svsub_f32_m(svptrue_b32(), *vec_hp1, *vec_hm1);
        SUM_FLUXES =  svsub_f32_m(svptrue_b32(), SUM_FLUXES, *vec_fB1);
        SUM_FLUXES =  svadd_f32_m(svptrue_b32(), SUM_FLUXES, *vec_fT1);
        SUM_FLUXES =  svmul_f32_m(svptrue_b32(), SUM_FLUXES, DT_DY);
        *vec_u1 = svsub_f32_m(svptrue_b32(), *vec_u1, SUM_FLUXES);

        // printf("Y Mom\n");
        SUM_FLUXES =  svsub_f32_m(svptrue_b32(), *vec_hp2, *vec_hm2);
        SUM_FLUXES =  svsub_f32_m(svptrue_b32(), SUM_FLUXES, *vec_fB2);
        SUM_FLUXES =  svadd_f32_m(svptrue_b32(), SUM_FLUXES, *vec_fT2);
        SUM_FLUXES =  svmul_f32_m(svptrue_b32(), SUM_FLUXES, DT_DY);
        *vec_u2 = svsub_f32_m(svptrue_b32(), *vec_u2, SUM_FLUXES);

        // printf("Eng\n");
        SUM_FLUXES =  svsub_f32_m(svptrue_b32(), *vec_hp3, *vec_hm3);
        SUM_FLUXES =  svsub_f32_m(svptrue_b32(), SUM_FLUXES, *vec_fB3);
        SUM_FLUXES =  svadd_f32_m(svptrue_b32(), SUM_FLUXES, *vec_fT3);
        SUM_FLUXES =  svmul_f32_m(svptrue_b32(), SUM_FLUXES, DT_DY);
        *vec_u3 = svsub_f32_m(svptrue_b32(), *vec_u3, SUM_FLUXES);
    }
    // printf("(U-from-F) Completed\n");
}

void Compute_F_from_P() {
    int vector;
    int no_bytes = svcntb()*8; int no_vectors = (int)(N/4); // Assuming 4 elements int vector = 0;
    // Create SVE pointers which point to our aligned memory space
    svfloat32_t *vec_a;
    svfloat32_t *vec_p0, *vec_p1, *vec_p2, *vec_p3, *vec_u0, *vec_u1, *vec_u2, *vec_u3, *vec_f0, *vec_f1, *vec_f2, *vec_f3;
    svfloat32_t *vec_fp0, *vec_fp1, *vec_fp2, *vec_fp3, *vec_fm0, *vec_fm1, *vec_fm2, *vec_fm3;
    svfloat32_t *vec_hp0, *vec_hp1, *vec_hp2, *vec_hp3, *vec_hm0, *vec_hm1, *vec_hm2, *vec_hm3;
    svfloat32_t *vec_h0, *vec_h1, *vec_h2, *vec_h3;
    // Create some constants
    svfloat32_t HALF = svdup_f32(0.5); svfloat32_t ONE = svdup_f32(1.0); svfloat32_t NEG_ONE = svdup_f32(-1.0);
    svfloat32_t CV = svdup_f32((R/(GAMMA-1.0))); svfloat32_t vec_R = svdup_f32(R);
    svfloat32_t T_CV, KE, KE_x, KE_y, T_CV_KE;
    svfloat32_t M;  // Mach number
    svfloat32_t Z1, Z2, Z3, P;
    svfloat32_t FLUX_COMPONENT, DISSIPATIVE_COMPONENT;

    // printf("(F-from-P) Iterating over vectors of length %d, performing %d iterations\n", no_bytes, no_vectors);
    for (vector = 0; vector < no_vectors; vector++) {
        // Now we can point our sve float types at these temp pointers
        // printf("Looking at vector %d\n", vector);
        vec_u0 = (svfloat32_t*)((float*)u0+(vector*4)); vec_u1 = (svfloat32_t*)((float*)u1+(vector*4));
        vec_u2 = (svfloat32_t*)((float*)u2+(vector*4)); vec_u3 = (svfloat32_t*)((float*)u3+(vector*4));
        vec_p0 = (svfloat32_t*)((float*)p0+(vector*4)); vec_p1 = (svfloat32_t*)((float*)p1+(vector*4));
        vec_p2 = (svfloat32_t*)((float*)p2+(vector*4)); vec_p3 = (svfloat32_t*)((float*)p3+(vector*4));
        vec_fp0 = (svfloat32_t*)((float*)fp0+(vector*4)); vec_fp1 = (svfloat32_t*)((float*)fp1+(vector*4));
        vec_fp2 = (svfloat32_t*)((float*)fp2+(vector*4)); vec_fp3 = (svfloat32_t*)((float*)fp3+(vector*4));
        vec_fm0 = (svfloat32_t*)((float*)fm0+(vector*4)); vec_fm1 = (svfloat32_t*)((float*)fm1+(vector*4));
        vec_fm2 = (svfloat32_t*)((float*)fm2+(vector*4)); vec_fm3 = (svfloat32_t*)((float*)fm3+(vector*4));
        vec_hp0 = (svfloat32_t*)((float*)hp0+(vector*4)); vec_hp1 = (svfloat32_t*)((float*)hp1+(vector*4));
        vec_hp2 = (svfloat32_t*)((float*)hp2+(vector*4)); vec_hp3 = (svfloat32_t*)((float*)hp3+(vector*4));
        vec_hm0 = (svfloat32_t*)((float*)hm0+(vector*4)); vec_hm1 = (svfloat32_t*)((float*)hm1+(vector*4));
        vec_hm2 = (svfloat32_t*)((float*)hm2+(vector*4)); vec_hm3 = (svfloat32_t*)((float*)hm3+(vector*4));
        vec_f0 = (svfloat32_t*)((float*)f0+(vector*4)); vec_f1 = (svfloat32_t*)((float*)f1+(vector*4));
        vec_f2 = (svfloat32_t*)((float*)f2+(vector*4)); vec_f3 = (svfloat32_t*)((float*)f3+(vector*4));
        vec_h0 = (svfloat32_t*)((float*)h0+(vector*4)); vec_h1 = (svfloat32_t*)((float*)h1+(vector*4));
        vec_h2 = (svfloat32_t*)((float*)h2+(vector*4)); vec_h3 = (svfloat32_t*)((float*)h3+(vector*4));
        vec_a = (svfloat32_t*)((float*)a+(vector*4));

        // Pressure - Ideal Gas Law (rho*R*T)
        P = svmul_f32_m(svptrue_b32(), *vec_p3, vec_R);
        P = svmul_f32_m(svptrue_b32(), *vec_p0, P);

        /*
        X Fluxes
        */
       //  printf("Computing x fluxes\n");

        // OK, got them. Now compute the fluxes of conserved quantities
        // Mass flux (rho*vel) - this is momentum
        *vec_f0 = svmul_f32_m(svptrue_b32(), *vec_u1, ONE);
        // Momentum flux F[:,:,1] = U[:,:,1]*P[:,:,1] + Pressure
        *vec_f1 = svmul_f32_m(svptrue_b32(), *vec_u1, *vec_p1);
        *vec_f1 = svadd_f32_m(svptrue_b32(), *vec_f1, P);
        // Momentum flux due to vy contribution
        *vec_f2 = svmul_f32_m(svptrue_b32(), *vec_u1, *vec_p2);
        // Energy P[:,:,1]*(U[:,:,3] + Pressure)
        *vec_f3 = svadd_f32_m(svptrue_b32(), *vec_u3, P);
        *vec_f3 = svmul_f32_m(svptrue_b32(), *vec_p1, *vec_f3);

        // Probably smarter to compute a and the critical number globally
        // TODO: Complete split flux calculation
        M = svdiv_f32_m(svptrue_b32(), *vec_p1, *vec_a);
        // Z1 = 0.5*(D*Fr+1.0)
        Z1 = svadd_f32_m(svptrue_b32(), M, ONE);
        Z1 = svmul_f32_m(svptrue_b32(), Z1, HALF);
        // Z3 = 0.5*(D*Fr-1.0)
        Z3 = svsub_f32_m(svptrue_b32(), M, ONE);
        Z3 = svmul_f32_m(svptrue_b32(), Z3, HALF);
        // Z2 = 0.5*D*a*(1.0-Fr*Fr)
        Z2 = svmul_f32_m(svptrue_b32(), M, M);
        Z2 = svsub_f32_m(svptrue_b32(), ONE, Z2);
        Z2 = svmul_f32_m(svptrue_b32(), Z2, *vec_a);
        Z2 = svmul_f32_m(svptrue_b32(), Z2, HALF);

        // Split fluxes - positive
        // FP[:,:,0] = F[:,:,0]*Z1 + U[:,:,0]*Z2
        FLUX_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_f0, Z1);
        DISSIPATIVE_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_u0, Z2);
        *vec_fp0 = svadd_f32_m(svptrue_b32(), FLUX_COMPONENT, DISSIPATIVE_COMPONENT);
        FLUX_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_f1, Z1);
        DISSIPATIVE_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_u1, Z2);
        *vec_fp1 = svadd_f32_m(svptrue_b32(), FLUX_COMPONENT, DISSIPATIVE_COMPONENT);
        FLUX_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_f2, Z1);
        DISSIPATIVE_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_u2, Z2);
        *vec_fp2 = svadd_f32_m(svptrue_b32(), FLUX_COMPONENT, DISSIPATIVE_COMPONENT);
        FLUX_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_f3, Z1);
        DISSIPATIVE_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_u3, Z2);
        *vec_fp3 = svadd_f32_m(svptrue_b32(), FLUX_COMPONENT, DISSIPATIVE_COMPONENT);

        // Split fluxes - negative
        // FM[:,:,0] = -F[:,:,0]*Z3 - U[:,:,0]*Z2
        FLUX_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_f0, Z3);
        DISSIPATIVE_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_u0, Z2);
        *vec_fm0 = svadd_f32_m(svptrue_b32(), FLUX_COMPONENT, DISSIPATIVE_COMPONENT);
        *vec_fm0 = svmul_f32_m(svptrue_b32(), *vec_fm0, NEG_ONE);
        FLUX_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_f1, Z3);
        DISSIPATIVE_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_u1, Z2);
        *vec_fm1 = svadd_f32_m(svptrue_b32(), FLUX_COMPONENT, DISSIPATIVE_COMPONENT);
        *vec_fm1 = svmul_f32_m(svptrue_b32(), *vec_fm1, NEG_ONE);
        FLUX_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_f2, Z3);
        DISSIPATIVE_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_u2, Z2);
        *vec_fm2 = svadd_f32_m(svptrue_b32(), FLUX_COMPONENT, DISSIPATIVE_COMPONENT);
        *vec_fm2 = svmul_f32_m(svptrue_b32(), *vec_fm2, NEG_ONE);
        FLUX_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_f3, Z3);
        DISSIPATIVE_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_u3, Z2);
        *vec_fm3 = svadd_f32_m(svptrue_b32(), FLUX_COMPONENT, DISSIPATIVE_COMPONENT);
        *vec_fm3 = svmul_f32_m(svptrue_b32(), *vec_fm3, NEG_ONE);
        /*
        Y Fluxes
        */
        //printf("Computing Y fluxes\n");
        // Mass flux (rho*vel) - this is momentum
        //printf("Mass\n");
        *vec_h0 = svmul_f32_m(svptrue_b32(), *vec_u2, ONE);
        // Momentum flux due to vx contribution
        //printf("Mom X\n");
        *vec_h1 = svmul_f32_m(svptrue_b32(), *vec_u2, *vec_p1);
        // Momentum (y)
        //printf("Mom Y\n");
        *vec_h2 = svmul_f32_m(svptrue_b32(), *vec_u2, *vec_p2);
        *vec_h2 = svadd_f32_m(svptrue_b32(), *vec_h2, P);
        // Energy
        //printf("Eng\n");
        *vec_h3 = svadd_f32_m(svptrue_b32(), *vec_u3, P);
        *vec_h3 = svmul_f32_m(svptrue_b32(), *vec_p2, *vec_h3);

        //printf("Computed h - now computing split fluxes\n");
        // Probably smarter to compute a and the critical number globally
        // TODO: Complete split flux calculation
        M = svdiv_f32_m(svptrue_b32(), *vec_p2, *vec_a);
        // Z1 = 0.5*(D*Fr+1.0)
        Z1 = svadd_f32_m(svptrue_b32(), M, ONE);
        Z1 = svmul_f32_m(svptrue_b32(), Z1, HALF);
        // Z3 = 0.5*(D*Fr-1.0)
        Z3 = svsub_f32_m(svptrue_b32(), M, ONE);
        Z3 = svmul_f32_m(svptrue_b32(), Z3, HALF);
        // Z2 = 0.5*D*a*(1.0-Fr*Fr)
        Z2 = svmul_f32_m(svptrue_b32(), M, M);
        Z2 = svsub_f32_m(svptrue_b32(), ONE, Z2);
        Z2 = svmul_f32_m(svptrue_b32(), Z2, *vec_a);
        Z2 = svmul_f32_m(svptrue_b32(), Z2, HALF);

        // Split fluxes - positive
        // FP[:,:,0] = F[:,:,0]*Z1 + U[:,:,0]*Z2
        FLUX_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_h0, Z1);
        DISSIPATIVE_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_u0, Z2);
        *vec_hp0 = svadd_f32_m(svptrue_b32(), FLUX_COMPONENT, DISSIPATIVE_COMPONENT);
        FLUX_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_h1, Z1);
        DISSIPATIVE_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_u1, Z2);
        *vec_hp1 = svadd_f32_m(svptrue_b32(), FLUX_COMPONENT, DISSIPATIVE_COMPONENT);
        FLUX_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_h2, Z1);
        DISSIPATIVE_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_u2, Z2);
        *vec_hp2 = svadd_f32_m(svptrue_b32(), FLUX_COMPONENT, DISSIPATIVE_COMPONENT);
        FLUX_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_h3, Z1);
        DISSIPATIVE_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_u3, Z2);
        *vec_hp3 = svadd_f32_m(svptrue_b32(), FLUX_COMPONENT, DISSIPATIVE_COMPONENT);

        // Split fluxes - negative
        // FM[:,:,0] = -F[:,:,0]*Z3 - U[:,:,0]*Z2
        FLUX_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_h0, Z3);
        DISSIPATIVE_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_u0, Z2);
        *vec_hm0 = svadd_f32_m(svptrue_b32(), FLUX_COMPONENT, DISSIPATIVE_COMPONENT);
        *vec_hm0 = svmul_f32_m(svptrue_b32(), *vec_hm0, NEG_ONE);
        FLUX_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_h1, Z3);
        DISSIPATIVE_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_u1, Z2);
        *vec_hm1 = svadd_f32_m(svptrue_b32(), FLUX_COMPONENT, DISSIPATIVE_COMPONENT);
        *vec_hm1 = svmul_f32_m(svptrue_b32(), *vec_hm1, NEG_ONE);
        FLUX_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_h2, Z3);
        DISSIPATIVE_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_u2, Z2);
        *vec_hm2 = svadd_f32_m(svptrue_b32(), FLUX_COMPONENT, DISSIPATIVE_COMPONENT);
        *vec_hm2 = svmul_f32_m(svptrue_b32(), *vec_hm2, NEG_ONE);
        FLUX_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_h3, Z3);
        DISSIPATIVE_COMPONENT = svmul_f32_m(svptrue_b32(), *vec_u3, Z2);
        *vec_hm3 = svadd_f32_m(svptrue_b32(), FLUX_COMPONENT, DISSIPATIVE_COMPONENT);
        *vec_hm3 = svmul_f32_m(svptrue_b32(), *vec_hm3, NEG_ONE);

    }
    // printf("(F-from-P) Completed\n");
}

void Compute_P_from_U() {
    int no_bytes = svcntb()*8; int no_vectors = (int)(N/4); // Assuming 4 elements int vector = 0;
    /*
    P[:,:,0] = U[:,:,0]                 # Water Height
    P[:,:,1] = U[:,:,1]/U[:,:,0]        # X vel
    P[:,:,2] = U[:,:,2]/U[:,:,0]        # Y vel
    P[:,:,3] = ((U[:,:,3]/U[:,:,0]) - 0.5*(P[:,:,1]*P[:,:,1]+P[:,:,2]*P[:,:,2]))/CV # Temp
    CFL = (P[:,:,1] + 2.0*np.sqrt(GAMMA*R*P[:,:,3]))*DT/DX
    */
    int vector;
    // Create SVE pointers which point to our aligned memory space
    svfloat32_t *vec_p0, *vec_p1, *vec_p2, *vec_p3, *vec_u0, *vec_u1, *vec_u2, *vec_u3, *vec_a;
    // Create some constants
    svfloat32_t HALF = svdup_f32(0.5); svfloat32_t ONE = svdup_f32(1.0);
    svfloat32_t CV = svdup_f32((R/(GAMMA-1.0)));
    svfloat32_t GAMMA_R = svdup_f32(R*GAMMA);
    svfloat32_t T_CV, KE, KE_x, KE_y, T_CV_KE;
    svfloat32_t SPECIFIC_E;
    svfloat32_t GAMMA_R_T;
    // printf("(P-from-U) Iterating over vectors of length %d, performing %d iterations\n", no_bytes, no_vectors);
    for (vector = 0; vector < no_vectors; vector++) {
        vec_p0 = (svfloat32_t*)((float*)p0+(vector*4)); vec_p1 = (svfloat32_t*)((float*)p1+(vector*4));
        vec_p2 = (svfloat32_t*)((float*)p2+(vector*4)); vec_p3 = (svfloat32_t*)((float*)p3+(vector*4));
        vec_u0 = (svfloat32_t*)((float*)u0+(vector*4)); vec_u1 = (svfloat32_t*)((float*)u1+(vector*4));
        vec_u2 = (svfloat32_t*)((float*)u2+(vector*4)); vec_u3 = (svfloat32_t*)((float*)u3+(vector*4));
        vec_a = (svfloat32_t*)((float*)a+(vector*4));

        // Now we can start - compute U from P Euler Equations Density
        // Damn it this is a waste
        *vec_p0 = svmul_f32_m(svptrue_b32(), *vec_u0, ONE);
        // Compute the speed (x)
        *vec_p1 = svdiv_f32_m(svptrue_b32(), *vec_u1, *vec_p0);
        // Compute the speed (y)
        *vec_p2 = svdiv_f32_m(svptrue_b32(), *vec_u2, *vec_p0);
        // Compute T
        KE_x = svmul_f32_m(svptrue_b32(), *vec_p1, *vec_p1);
        KE_y = svmul_f32_m(svptrue_b32(), *vec_p2, *vec_p2);
        KE = svadd_f32_m(svptrue_b32(), KE_x, KE_y);
        KE = svmul_f32_m(svptrue_b32(), KE, HALF);
        SPECIFIC_E = svdiv_f32_m(svptrue_b32(), *vec_u3, *vec_u0);
        T_CV = svsub_f32_m(svptrue_b32(), SPECIFIC_E, KE);
        *vec_p3 = svdiv_f32_m(svptrue_b32(), T_CV, CV);
        // Compute a; this is useful for computing the critical number (M)
        GAMMA_R_T = svmul_f32_m(svptrue_b32(), GAMMA_R, *vec_p3);
        // The first agument of svsqrt_f32_m is inactive; pass in 1
        *vec_a = svsqrt_f32_m(ONE, svptrue_b32(), GAMMA_R_T);
    }
    // printf("(P-from-U) Complete\n");
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

    // Save_Results();

    // Free
    Free_Memory();
}