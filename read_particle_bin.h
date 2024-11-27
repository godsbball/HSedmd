#ifndef _read_particle_bin_h_
#define _read_particle_bin_h_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#ifndef __cplusplus
#define bool _Bool
#endif

#define true	1
#define false	0
#define DIM 3
#define voigtDIM ((DIM * (DIM + 1) / 2))


#if defined(checkMacroFlag)
#define spaceIdx2voigt(alpha, beta)            \
({                                         \
    if ((alpha) < 0 || (alpha) >= DIM)     \
        exit(EXIT_FAILURE);                \
    if ((beta) < 0 || (beta) >= DIM)       \
        exit(EXIT_FAILURE);                \
    if ((alpha) > (beta))                  \
        exit(EXIT_FAILURE);                \
    ((alpha) + (beta) * ((beta) + 1) / 2); \
})
#else
#define spaceIdx2voigt(alpha, beta) ((alpha) + (beta) * ((beta) + 1) / 2)
#endif


typedef double doubleVector[DIM];
typedef int intVector[DIM];
typedef double uptriMat[voigtDIM];//up-trianglar part of symmetric matrix.



// bool isEmpty(char *str, int maxlen) {
//     if (maxlen <= 0) Abort("Fatal Error!");
//     if (strlen(str) >= maxlen) Abort("Fatal Error");
//     if (strlen(str) == 0) return true;

//     char *start = str, *stop = str + strlen(str) - 1;
//     while (isspace(*stop) && stop >= start) stop--;
//     while (isspace(*start) && start <= stop) start++;

//     if (start > stop) return true;
//     return false;
// }
typedef struct int2 {
    int first, second;
} int2;

typedef struct Box {
    int dim;
    uptriMat boxH, invBoxH;
    double volume;

    doubleVector boxEdge[DIM];

    doubleVector cornerLo;
    doubleVector cornerHi;

    bool isShapeFixed;
} Box;

typedef struct Particle {
    int nAtom, nAtomType;

    doubleVector *pos;
    doubleVector *veloc;
    intVector *img;
    doubleVector *force;

    doubleVector *uPos;

    // the diameter in code is diameterScale[iatom] * meandiameter.
    double *diameterScale, meanDiameter;
    
    //===just for future feature===
    int *type;
    double *mass;
    double *massPerType;
    //===just for future feature===

    int *id2tag, *tag2id;// tag is the permanent label (id2tag[id]), id is the memory index (tag2id[tag]);
    int sortFlag;//increased when the index is reordered.
    
    // The library MUST NOT rewrite isSortForbidden.
    bool isSortForbidden; //default: false. True: Never doing sorting and (label == index).
    
    bool isSizeFixed;
    bool isForceValid;
} Particle;

typedef struct idPosRadius {
    int id;
    doubleVector pos;
    double radius;
} idPosRadius;

typedef struct NebrList {
    //to check consistency
    Box *boxPtrSave;
    Particle *particlePtrSave;
    
    doubleVector normPlane[DIM];  // 3D: 0:yoz; 1: xoz; 2:xoy; 2D: 0:oy; 1: ox;
    int nImage, maxImage;         // image particle
    // used to generate image pos
    int *imageParent;
    int imagePartCount[2 * DIM];
    // used to get source particle
    int *imgageSource;

    double maxDiameterScale, minDiameterScale;
    double skinSet, rskin, maxRcut;

    doubleVector binLen;
    intVector nbin;      // box
    intVector nbinExt;   // box + stencil
    intVector nStencil;  // stencil
    int totBin, totBinExt, allocBinExt;
    bool *isSourceBin;

    int2 *stencilSource;
    int exLayerNum[2 * DIM];

    doubleVector *xyzHold;
    double meanDiameterHold;
    uptriMat invBoxHold;

    idPosRadius *binHead, *binList;
    int maxBinHead;
    int nAdjBin, allocAdjBin;
#if (DIM == 2 || DIM == 3)
    int *adjBinList;
#else
    intVector *adjBinList;
#endif

    //====Half style NebrList: compute force====
    int *list;
    int2 *nNebr;
    int maxAllocNebr;

    int nDelay, nRebuildMax;  // for thermal run
    //===list for sorting===
    int *binHead4sort, *binList4sort;
    int totBin4sort, allocBin4sort;
    int *oid2nid;
    void *buffer;

    long int nBuild, nDangerous, nForce;
    int cntForce;
    bool isValid, doSort;
    bool compelInit;
} NebrList;





// typedef struct Update {
//     // $\tilde{A}_{sim} = A_{real} \ast A_{units}$
//     double massUnits, energyUnits, distanceUnits;
//     double timeUnits, forceUnits, velocityUnits;
//     double pressureUnits, volumeUnits;

//     // thermal properties
//     double tempKin, pVirKin, ePair, pVir, maxOvlp;  // pVirKin: both Virial and kinetic contributions.
//     uptriMat kinTens, pVirKinTens, fabricTensor, pVirTens;

//     int nContact;
//     double dof, volFrac;
//     bool Edone, Pdone, Tdone, Wdone;

//     NebrList nebrList;
//     bool isThermalRun;
//     int evFlag;  // -1: check; 0: none; 2: energy and virial;

//     Toolkit toolkit;
// } Update;

// typedef struct cmdArg {
//     char *cmdType;
//     int cmdArgc;
//     char **cmdArgv;
// } cmdArg;
// typedef struct Variable {
//     int nVar, maxVar;
//     cmdArg *cmd;
    
//     char *cwd, *sf;
// } Variable;

void Abort(const char *str);
bool isEmpty(char *str, int maxlen);
void readConf_data(Box *box, Particle *particle, char *fname);
// void readConf(Box *box, Particle *particle, Update *update, Variable *var);



#endif