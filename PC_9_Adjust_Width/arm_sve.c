/* https://developer.arm.com/architectures/instruction-sets/intrinsics
Euler solver using low level SVE intrinsics for computation on next
generation ARM architectures. In this case, we are targetting AWS's Graviton4 processor. */

#include <stdio.h>
#include <arm_sve.h>
#include <stdlib.h>
float *p0, *p1, *p2, *u0, *u1, *u2;
float *f0, *f1, *f2;  // Fluxes
float *fp0, *fp1, *fp2;  // Split flux (positive)
float *fm0, *fm1, *fm2;  // Split flux (minus [negative])
float *Left_f0, *Left_f1, *Left_f2;
float *Right_f0, *Right_f1, *Right_f2;
float *a;
const int N = 256;
const float R = 1.0;
const float GAMMA=1.4;
const float CV = R/(GAMMA-1.0);
const float L = 1.0;
const float DX = L/N;
const float CFL = 0.25;
float DT;
float DT_ON_DX;
int NO_STEPS = 0;
const float TOTAL_TIME = 0.2;
const int VEC_WIDTH = 8;        // 8 floating points per vector


void Allocate_and_Init_Memory() {
    size_t alignment = 32; int i;
    posix_memalign((void**)&p0, alignment, N*sizeof(float));
    posix_memalign((void**)&p1, alignment, N*sizeof(float));
    posix_memalign((void**)&p2, alignment, N*sizeof(float));
    posix_memalign((void**)&u0, alignment, N*sizeof(float));
    posix_memalign((void**)&u1, alignment, N*sizeof(float));
    posix_memalign((void**)&u2, alignment, N*sizeof(float));
    posix_memalign((void**)&f0, alignment, N*sizeof(float));
    posix_memalign((void**)&f1, alignment, N*sizeof(float));
    posix_memalign((void**)&f2, alignment, N*sizeof(float));
    posix_memalign((void**)&fp0, alignment, N*sizeof(float));
    posix_memalign((void**)&fp1, alignment, N*sizeof(float));
    posix_memalign((void**)&fp2, alignment, N*sizeof(float));
    posix_memalign((void**)&fm0, alignment, N*sizeof(float));
    posix_memalign((void**)&fm1, alignment, N*sizeof(float));
    posix_memalign((void**)&fm2, alignment, N*sizeof(float));
    posix_memalign((void**)&Left_f0, alignment, N*sizeof(float));
    posix_memalign((void**)&Left_f1, alignment, N*sizeof(float));
    posix_memalign((void**)&Left_f2, alignment, N*sizeof(float));
    posix_memalign((void**)&Right_f0, alignment, N*sizeof(float));
    posix_memalign((void**)&Right_f1, alignment, N*sizeof(float));
    posix_memalign((void**)&Right_f2, alignment, N*sizeof(float));
    posix_memalign((void**)&a, alignment, N*sizeof(float));

    // Set up the problem
    for (i = 0; i < N; i++) {
        if (i < 0.5*N) {
            p0[i] = 10.0; p1[i] = 1.0; p2[i] = 1.0;
        } else {
            p0[i] = 1.0; p1[i] = 1.0; p2[i] = 1.0;
        }
    }
}

void Free_Memory() {
    free(p0); free(p1); free(p2);
    free(u0); free(u1); free(u2);
    free(f0); free(f1); free(f2);
    free(fp0); free(fp1); free(fp2);
    free(fm0); free(fm1); free(fm2);
    free(a);
    free(Left_f0); free(Left_f1); free(Left_f2);
    free(Right_f0); free(Right_f1); free(Right_f2);
}

void Compute_U_from_P() {
    // Compute the number of bytes
    int no_bytes = svcntb()*8; int no_vectors = (int)(N/VEC_WIDTH); // Assuming VEC_WIDTH elements per vector;
    int vector;
    // Create SVE pointers which point to our aligned memory space
    svfloat32_t *vec_p0, *vec_p1, *vec_p2, *vec_u0, *vec_u1, *vec_u2;
    // Create temporary pointers to help us avoid pointer arithmetic with sve types
    float *temp_p0, *temp_p1, *temp_p2, *temp_u0, *temp_u1, *temp_u2;
    // Create some constants
    svfloat32_t HALF = svdup_f32(0.5); svfloat32_t ONE = svdup_f32(1.0);
    svfloat32_t CV = svdup_f32((R/(GAMMA-1.0)));
    svfloat32_t T_CV; svfloat32_t KE; svfloat32_t T_CV_KE;
    // printf("Iterating over vectors of length %d, performing %d iterations\n", no_bytes, no_vectors);
    for (vector = 0; vector < no_vectors; vector++) {
        // printf("Vector %d of %d\n", vector, no_vectors-1);
        // Perform pointer arithmetic on floats; this is permitted in GCC extensions
        temp_p0 = (float*)p0+(vector*VEC_WIDTH); temp_p1 = (float*)p1+(vector*VEC_WIDTH); temp_p2 = (float*)p2+(vector*VEC_WIDTH);
        temp_u0 = (float*)u0+(vector*VEC_WIDTH); temp_u1 = (float*)u1+(vector*VEC_WIDTH); temp_u2 = (float*)u2+(vector*VEC_WIDTH);
        // Now we can point our sve float types at these temp pointers
        vec_p0 = (svfloat32_t*)temp_p0; vec_p1 = (svfloat32_t*)temp_p1; vec_p2 = (svfloat32_t*)temp_p2;
        vec_u0 = (svfloat32_t*)temp_u0; vec_u1 = (svfloat32_t*)temp_u1; vec_u2 = (svfloat32_t*)temp_u2;
        // Now we can start - compute U from P Euler Equations Density
        *vec_u0 = svmul_f32_m(svptrue_b32(), *vec_p0, ONE);
        // Momentum
        *vec_u1 = svmul_f32_m(svptrue_b32(), *vec_p0, *vec_p1);
        // Energy (a little complex)
        KE = svmul_f32_m(svptrue_b32(), *vec_p1, *vec_p1);
        // Multiply by 1/2
        KE = svmul_f32_m(svptrue_b32(), KE, HALF); T_CV = svmul_f32_m(svptrue_b32(), *vec_p2, CV);
        // Add these
        T_CV_KE = svadd_f32_m(svptrue_b32(), T_CV, KE);
        *vec_u2 = svmul_f32_m(svptrue_b32(), *vec_p0, T_CV_KE);
        // 3D U[index+4] = P[index]*(P[index+4]*CV+0.5*(P[index+1]*P[index+1]+P[index+2]*P[index+2]+P[index+3]*P[index+3]));
    }

    // Update DT based on desired CFL
    // Estimated CFL = ((R + 1)*DT)/DX;
    DT = (CFL/(R+1))*DX;
    DT_ON_DX = (CFL/(R+1));   
}

void Update_U_from_F() {
    // Compute the number of bytes
    int no_bytes = svcntb()*8; int no_vectors = (int)(N/VEC_WIDTH); // Assuming 4 elements int vector = 0;
    int vector;
    int index;
    int vector_index;
    // Create SVE pointers which point to our aligned memory space
    svfloat32_t *vec_u0, *vec_u1, *vec_u2;
    svfloat32_t *vec_fp0, *vec_fp1, *vec_fp2;
    svfloat32_t *vec_fm0, *vec_fm1, *vec_fm2;
    svfloat32_t *vec_fR0, *vec_fR1, *vec_fR2;
    svfloat32_t *vec_fL0, *vec_fL1, *vec_fL2;

    // Create temporary pointers to help us avoid pointer arithmetic with sve types
    float *temp_u0, *temp_u1, *temp_u2;
    float *temp_fp0, *temp_fp1, *temp_fp2;
    float *temp_fm0, *temp_fm1, *temp_fm2;
    float *temp_fR0, *temp_fR1, *temp_fR2;
    float *temp_fL0, *temp_fL1, *temp_fL2;

    // Create some constants
    svfloat32_t DT_DX = svdup_f32(DT_ON_DX);
    svfloat32_t LEFT_FLUX, RIGHT_FLUX;
    svfloat32_t SUM_FLUXES;
    // printf("Iterating over vectors of length %d, performing %d iterations\n", no_bytes, no_vectors);

    // We shall break this down into two parts
    // i) Compute the cell left and right fluxes using serial computation, and then
    // ii) Compute the update to U based on these using SVE
    for (index = 0; index < N; index++) {
        if (index == 0) {
            // Left end boundary condition - reflective
            Left_f0[index] = -fm0[index];
            Left_f1[index] = fm1[index];
            Left_f2[index] = -fm2[index];
            // Compute Right contribution using fm(index+1)
            Right_f0[index] = fm0[index+1];
            Right_f1[index] = fm1[index+1];
            Right_f2[index] = fm2[index+1];
        } else if (index == (N-1)) {
            // Right end reflective boundary condition
            Left_f0[index] = fp0[index-1];
            Left_f1[index] = fp1[index-1];
            Left_f2[index] = fp2[index-1];
            // Reflective conditions for Right
            Right_f0[index] = -fp0[index];
            Right_f1[index] = fp1[index];
            Right_f2[index] = -fp2[index];            
        } else {
            Left_f0[index] = fp0[index-1];
            Left_f1[index] = fp1[index-1];
            Left_f2[index] = fp2[index-1];
            Right_f0[index] = fm0[index+1];
            Right_f1[index] = fm1[index+1];
            Right_f2[index] = fm2[index+1];
        }
    }

    // Update the state
    // dU = dU - PHI*(FP - FM + FR - FL)
    for (vector = 0; vector < no_vectors; vector++) {  
        vector_index = vector*VEC_WIDTH;  
        temp_u0 = (float*)u0+(vector_index); temp_u1 = (float*)u1+(vector_index); temp_u2 = (float*)u2+(vector_index);
        temp_fp0 = (float*)fp0+(vector_index); temp_fp1 = (float*)fp1+(vector_index); temp_fp2 = (float*)fp2+(vector_index);
        temp_fm0 = (float*)fm0+(vector_index); temp_fm1 = (float*)fm1+(vector_index); temp_fm2 = (float*)fm2+(vector_index);
        temp_fR0 = (float*)Right_f0+(vector_index); temp_fR1 = (float*)Right_f1+(vector_index); temp_fR2 = (float*)Right_f2+(vector_index);
        temp_fL0 = (float*)Left_f0+(vector_index); temp_fL1 = (float*)Left_f1+(vector_index); temp_fL2 = (float*)Left_f2+(vector_index);

        vec_u0 = (svfloat32_t*)temp_u0; vec_u1 = (svfloat32_t*)temp_u1; vec_u2 = (svfloat32_t*)temp_u2;
        vec_fp0 = (svfloat32_t*)temp_fp0; vec_fp1 = (svfloat32_t*)temp_fp1; vec_fp2 = (svfloat32_t*)temp_fp2;
        vec_fm0 = (svfloat32_t*)temp_fm0; vec_fm1 = (svfloat32_t*)temp_fm1; vec_fm2 = (svfloat32_t*)temp_fm2;
        vec_fR0 = (svfloat32_t*)temp_fR0; vec_fR1 = (svfloat32_t*)temp_fR1; vec_fR2 = (svfloat32_t*)temp_fR2;
        vec_fL0 = (svfloat32_t*)temp_fL0; vec_fL1 = (svfloat32_t*)temp_fL1; vec_fL2 = (svfloat32_t*)temp_fL2;

        // Update the state
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
    }
}

void Compute_F_from_P() {
    int vector;
    int vector_index;
    int no_bytes = svcntb()*8; int no_vectors = (int)(N/VEC_WIDTH);
    // Create SVE pointers which point to our aligned memory space
    svfloat32_t *vec_a;
    svfloat32_t *vec_p0, *vec_p1, *vec_p2, *vec_u0, *vec_u1, *vec_u2, *vec_f0, *vec_f1, *vec_f2;
    svfloat32_t *vec_fp0, *vec_fp1, *vec_fp2, *vec_fm0, *vec_fm1, *vec_fm2;
    // Create temporary pointers to help us avoid pointer arithmetic with sve types
    float *temp_p0, *temp_p1, *temp_p2, *temp_u0, *temp_u1, *temp_u2, *temp_f0, *temp_f1, *temp_f2;
    float *temp_fp0, *temp_fp1, *temp_fp2, *temp_fm0, *temp_fm1, *temp_fm2;
    float *temp_a;
    // Create some constants
    svfloat32_t HALF = svdup_f32(0.5); svfloat32_t ONE = svdup_f32(1.0); svfloat32_t NEG_ONE = svdup_f32(-1.0); 
    svfloat32_t CV = svdup_f32((R/(GAMMA-1.0)));
    svfloat32_t vec_R = svdup_f32(R);
    svfloat32_t T_CV; svfloat32_t KE; svfloat32_t T_CV_KE;
    svfloat32_t M; // Mach number 
    svfloat32_t Z1, Z2, Z3, P;
    svfloat32_t FLUX_COMPONENT, DISSIPATIVE_COMPONENT;

    // printf("Iterating over vectors of length %d, performing %d iterations\n", no_bytes, no_vectors);
    for (vector = 0; vector < no_vectors; vector++) {
        // printf("Vector %d of %d\n", vector, no_vectors-1);
        // Perform pointer arithmetic on floats; this is permitted in GCC extensions
        vector_index = vector*VEC_WIDTH;
        temp_p0 = (float*)p0+(vector_index); temp_p1 = (float*)p1+(vector_index); temp_p2 = (float*)p2+(vector_index);
        temp_u0 = (float*)u0+(vector_index); temp_u1 = (float*)u1+(vector_index); temp_u2 = (float*)u2+(vector_index);
        temp_f0 = (float*)f0+(vector_index); temp_f1 = (float*)f1+(vector_index); temp_f2 = (float*)f2+(vector_index);
        temp_fp0 = (float*)fp0+(vector_index); temp_fp1 = (float*)fp1+(vector_index); temp_fp2 = (float*)fp2+(vector_index);
        temp_fm0 = (float*)fm0+(vector_index); temp_fm1 = (float*)fm1+(vector_index); temp_fm2 = (float*)fm2+(vector_index);
        temp_a = (float*)a+(vector_index);
        // Now we can point our sve float types at these temp pointers
        vec_p0 = (svfloat32_t*)temp_p0; vec_p1 = (svfloat32_t*)temp_p1; vec_p2 = (svfloat32_t*)temp_p2;
        vec_u0 = (svfloat32_t*)temp_u0; vec_u1 = (svfloat32_t*)temp_u1; vec_u2 = (svfloat32_t*)temp_u2;
        vec_f0 = (svfloat32_t*)temp_f0; vec_f1 = (svfloat32_t*)temp_f1; vec_f2 = (svfloat32_t*)temp_f2;
        vec_fp0 = (svfloat32_t*)temp_fp0; vec_fp1 = (svfloat32_t*)temp_fp1; vec_fp2 = (svfloat32_t*)temp_fp2;
        vec_fm0 = (svfloat32_t*)temp_fm0; vec_fm1 = (svfloat32_t*)temp_fm1; vec_fm2 = (svfloat32_t*)temp_fm2;
        vec_a = (svfloat32_t*)temp_a;
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
        // Pressure - Ideal Gas Law (rho*R*T)
        P = svmul_f32_m(svptrue_b32(), *vec_p2, vec_R);
        P = svmul_f32_m(svptrue_b32(), *vec_p0, P);

        // OK, got them. Now compute the fluxes of conserved quantities
        // Mass flux (rho*vel) - this is momentum
        *vec_f0 = svmul_f32_m(svptrue_b32(), *vec_u1, ONE);
        // Momentum flux F[:,:,1] = U[:,:,1]*P[:,:,1] + Pressure 
        *vec_f1 = svmul_f32_m(svptrue_b32(), *vec_u1, *vec_p1);
        *vec_f1 = svadd_f32_m(svptrue_b32(), *vec_f1, P);
        // Energy P[:,:,1]*(U[:,:,3] + Pressure)
        *vec_f2 = svadd_f32_m(svptrue_b32(), *vec_u2, P);
        *vec_f2 = svmul_f32_m(svptrue_b32(), *vec_p1, *vec_f2);

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

    }
}

void Compute_P_from_U() {
    int no_bytes = svcntb()*8; int no_vectors = (int)(N/VEC_WIDTH);
    int vector_index;
    /*
    P[:,:,0] = U[:,:,0]   		# Water Height
    P[:,:,1] = U[:,:,1]/U[:,:,0]	# X vel
    P[:,:,2] = U[:,:,2]/U[:,:,0]	# Y vel
    P[:,:,3] = ((U[:,:,3]/U[:,:,0]) - 0.5*(P[:,:,1]*P[:,:,1]+P[:,:,2]*P[:,:,2]))/CV # Temp	
    CFL = (P[:,:,1] + 2.0*np.sqrt(GAMMA*R*P[:,:,3]))*DT/DX
    */
    int vector;
    // Create SVE pointers which point to our aligned memory space
    svfloat32_t *vec_p0, *vec_p1, *vec_p2, *vec_u0, *vec_u1, *vec_u2, *vec_a;
    // Create temporary pointers to help us avoid pointer arithmetic with sve types
    float *temp_p0, *temp_p1, *temp_p2, *temp_u0, *temp_u1, *temp_u2, *temp_a;
    // Create some constants
    svfloat32_t HALF = svdup_f32(0.5); svfloat32_t ONE = svdup_f32(1.0);
    svfloat32_t CV = svdup_f32((R/(GAMMA-1.0)));
    svfloat32_t GAMMA_R = svdup_f32(R*GAMMA);
    svfloat32_t T_CV; svfloat32_t KE; svfloat32_t T_CV_KE;
    svfloat32_t SPECIFIC_E;
    svfloat32_t GAMMA_R_T;
    svbool_t NONE;
    // printf("Iterating over vectors of length %d, performing %d iterations\n", no_bytes, no_vectors);
    for (vector = 0; vector < no_vectors; vector++) {
        vector_index = vector*VEC_WIDTH; 
        temp_p0 = (float*)p0+(vector_index); temp_p1 = (float*)p1+(vector_index); temp_p2 = (float*)p2+(vector_index);
        temp_u0 = (float*)u0+(vector_index); temp_u1 = (float*)u1+(vector_index); temp_u2 = (float*)u2+(vector_index);
        temp_a = (float*)a+(vector_index);
        vec_p0 = (svfloat32_t*)temp_p0; vec_p1 = (svfloat32_t*)temp_p1; vec_p2 = (svfloat32_t*)temp_p2;
        vec_u0 = (svfloat32_t*)temp_u0; vec_u1 = (svfloat32_t*)temp_u1; vec_u2 = (svfloat32_t*)temp_u2;
        vec_a = (svfloat32_t*)temp_a;

        // Now we can start - compute U from P Euler Equations Density
        // Damn it this is a waste
        *vec_p0 = svmul_f32_m(svptrue_b32(), *vec_u0, ONE);
        // Compute the speed
        *vec_p1 = svdiv_f32_m(svptrue_b32(), *vec_u1, *vec_p0);
        // Compute T
        KE = svmul_f32_m(svptrue_b32(), *vec_p1, *vec_p1);
        KE = svmul_f32_m(svptrue_b32(), KE, HALF);
        SPECIFIC_E = svdiv_f32_m(svptrue_b32(), *vec_u2, *vec_u0);
        T_CV = svsub_f32_m(svptrue_b32(), SPECIFIC_E, KE);
        *vec_p2 = svdiv_f32_m(svptrue_b32(), T_CV, CV);
        // Compute a; this is useful for computing the critical number (M)
        GAMMA_R_T = svmul_f32_m(svptrue_b32(), GAMMA_R, *vec_p2);
        // The first agument of svsqrt_f32_m is inactive; pass in 1
        *vec_a = svsqrt_f32_m(ONE, svptrue_b32(), GAMMA_R_T);
    }
}


// Replace with auto iterators
void Compute_U_from_P_AUTO() {
    int i;
    int count = 0;
    svfloat32_t ONE = svdup_f32(1.0);
    svfloat32_t CV = svdup_f32((R/(GAMMA-1.0)));
    svfloat32_t HALF = svdup_f32(0.5);
    svfloat32_t T_CV; svfloat32_t KE; svfloat32_t T_CV_KE;
    svint8_t BOOL_INSPECT;
    char mask[4];
    // svcntd counts the number of 64 bit elements present
    // For 128 bit vectors, this will be 2, which processes 4 32bit floats
    // This is where we live for 256 bits, but I just can't get it working.
    // for (int i = 0; i < N; i += 4*svcntd()) {
    
    // This is for 128 bits, as this holds 2x 64 bit entities
    for (int i = 0; i < N; i += 2*svcntd()) {
	count++;
	printf("I live here: count =  %d (index %d)\n", count, i);
        // Iterating over 32 bit elements (floats)
        svbool_t Pg = svwhilelt_b32(i, N);
        svfloat32_t vp0 = svld1(Pg, &p0[i]);
        svfloat32_t vp1 = svld1(Pg, &p1[i]);
        svfloat32_t vp2 = svld1(Pg, &p2[i]);
        svfloat32_t vu0 = svmul_f32_m(Pg, ONE, vp0);
        svfloat32_t vu1 = svmul_f32_m(Pg, vp0, vp1);
        KE = svmul_f32_m(Pg, vp1, vp1);
        KE = svmul_f32_m(Pg, KE, HALF);
        T_CV = svmul_f32_m(Pg, vp2, CV);
        T_CV_KE = svadd_f32_m(Pg, T_CV, KE);
        svfloat32_t vu2 = svmul_f32_m(svptrue_b32(), vp0, T_CV_KE);
        svst1(Pg, &u0[i], vu0);
        svst1(Pg, &u1[i], vu1);
        svst1(Pg, &u2[i], vu2);
	// BOOL_INSPECT = (svint8_t)Pg;
	// svst1(Pg, &mask[0], BOOL_INSPECT);
	// printf("Mask values = %d, %d, %d, %d\n", mask[0], mask[1], mask[2], mask[3]);
    }
}


void Save_Results() {
    FILE *fptr;
    int i;
    float cx;

    fptr = fopen("results.dat", "w");
    for (i = 0; i < N; i++) {
        cx = (i+0.5)*DX;
        fprintf(fptr, "%e\t%e\t%e\t%e\n", cx, p0[i], p1[i], p2[i]);
    }
    // Close the file
    fclose(fptr);
}


int main() {
    int i;
    float time = 0.0;
    int no_bytes = svcntb()*8;
    printf("SVCNTB() = %d, No. bytes = %d\n",svcntb(), no_bytes);
    // Allocate
    Allocate_and_Init_Memory();
    // Compute U from P
    //Compute_U_from_P();
    Compute_U_from_P_AUTO();
    //Compute_P_from_U();
    //Compute_F_from_P();
    /*
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
    */

    printf("Completed in %d steps\n", NO_STEPS);
    // Save_Results();

    /*
    printf("============ DISCONTINUITY ===============\n");
    // Let's have a look at the discontinuity
    for (i = 124; i < 132; i++) {
        printf("u0[%d] = %g, u1[%d] = %g, u2[%d] = %g\n", i, u0[i], i, u1[i], i, u2[i]);
        printf("p0[%d] = %g, p1[%d] = %g, p2[%d] = %g, a[%d] = %e\n",i,  p0[i], i,  p1[i], i,  p2[i], i,  a[i]);
        printf("fp0[%d] = %g, fp1[%d] = %g, fp2[%d] = %g\n", i, fp0[i], i,  fp1[i], i,  fp2[i], i);
        printf("fm0[%d] = %g, fm1[%d] = %g, fm2[%d] = %g\n", i, fm0[i], i,  fm1[i], i,  fm2[i], i);
    }
    */

    // Look at the left hand side bounds
    printf("============ LEFT HAND BOUNDS ===============\n");
    for (i = 0; i < 15; i++) {
	printf("-----\n");    
        printf("u0[%d] = %g, u1[%d] = %g, u2[%d] = %g\n", i, u0[i], i, u1[i], i, u2[i]);
        printf("p0[%d] = %g, p1[%d] = %g, p2[%d] = %g, a[%d] = %e\n",i,  p0[i], i,  p1[i], i,  p2[i], i,  a[i]);
        // printf("fp0[%d] = %g, fp1[%d] = %g, fp2[%d] = %g\n", i, fp0[i], i,  fp1[i], i,  fp2[i], i);
        // printf("fm0[%d] = %g, fm1[%d] = %g, fm2[%d] = %g\n", i, fm0[i], i,  fm1[i], i,  fm2[i], i);
    }

    // Look at the right hand side bounds
    printf("============ RIGHT HAND BOUNDS ===============\n");
    for (i = N-15; i < N; i++) {
	printf("--------\n");
        printf("u0[%d] = %g, u1[%d] = %g, u2[%d] = %g\n", i, u0[i], i, u1[i], i, u2[i]);
        printf("p0[%d] = %g, p1[%d] = %g, p2[%d] = %g, a[%d] = %e\n",i,  p0[i], i,  p1[i], i,  p2[i], i,  a[i]);
        //printf("fp0[%d] = %g, fp1[%d] = %g, fp2[%d] = %g\n", i, fp0[i], i,  fp1[i], i,  fp2[i], i);
        //printf("fm0[%d] = %g, fm1[%d] = %g, fm2[%d] = %g\n", i, fm0[i], i,  fm1[i], i,  fm2[i], i);
    }

    // Free
    Free_Memory();
}
