/* https://developer.arm.com/architectures/instruction-sets/intrinsics
Euler solver using low level SVE intrinsics for computation on next
generation ARM architectures. In this case, we are targetting AWS's Graviton4 processor.
This is the base C code equivalent to the SVE code. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
            p0[i] = 10.0; p1[i] = 0.0; p2[i] = 1.0;
        } else {
            p0[i] = 1.0; p1[i] = 0.0; p2[i] = 1.0;
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
    int cell;
    for (cell = 0; cell < N; cell++) {
        u0[cell] = p0[cell];
        u1[cell] = p0[cell]*p1[cell];
        u2[cell] = p0[cell]*(p2[cell]*CV + 0.5*p1[cell]*p1[cell]);
    }
    // Update DT based on desired CFL
    // Estimated CFL = ((R + 1)*DT)/DX;
    DT = (CFL/(R+1))*DX;
    DT_ON_DX = (CFL/(R+1));   
}

void Update_U_from_F() {
    int index;
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
    for (index = 0; index < N; index++) {    
        u0[index] = u0[index] - DT_ON_DX*(fp0[index] - fm0[index] + Right_f0[index] - Left_f0[index]);
        u1[index] = u1[index] - DT_ON_DX*(fp1[index] - fm1[index] + Right_f1[index] - Left_f1[index]);
        u2[index] = u2[index] - DT_ON_DX*(fp2[index] - fm2[index] + Right_f2[index] - Left_f2[index]);
    }
}

void Compute_F_from_P() {
    int cell;
    float Z1, Z2, Z3, M, P;

    for (cell = 0; cell < N; cell++) {
        M = p1[cell]/a[cell];
        // Z invariants
        Z1 = 0.5*(M + 1.0);
        Z2 = 0.5*a[cell]*(1.0-M*M);
        Z3 = 0.5*(M - 1.0);
        // Pressure
        P = p0[cell]*R*p2[cell];
        // Fluxes of conserved quantities
        f0[cell] = u1[cell];
        f1[cell] = u1[cell]*p1[cell] + P;
        f2[cell] = p1[cell]*(u2[cell] + P);

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
        p2[cell] = ((u2[cell]/u0[cell]) - 0.5*p1[cell]*p1[cell])/CV;
        a[cell] = sqrt(GAMMA*R*p2[cell]);
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

