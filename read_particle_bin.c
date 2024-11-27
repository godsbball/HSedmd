#include "read_particle_bin.h"

void Abort(const char *str){
    // fprintf(stderr,str);
    exit(EXIT_FAILURE);  // EXIT_FAILURE 表示程序异常退出
}

bool isEmpty(char *str, int maxlen) {
    if (maxlen <= 0) Abort("Fatal Error!");
    if (strlen(str) >= maxlen) Abort("Fatal Error");
    if (strlen(str) == 0) return true;

    char *start = str, *stop = str + strlen(str) - 1;
    while (isspace(*stop) && stop >= start) stop--;
    while (isspace(*start) && start <= stop) start++;

    if (start > stop) return true;
    return false;
}

void readConf_data(Box *box, Particle *particle, char *fname) {
    FILE *fp = fopen(fname, "r");
    // if (particle->pos != NULL)
    //     Abort("--rf or --rd");
    
    char str[4096];
    fgets(str, 4096, fp);
    if (strcmp(str, "binary") == 0) {
        bool hasMeanDiameter = false, hasDimension = false;
        while (fread(str, sizeof(char), 32, fp) == 32) {
            if (strcmp(str, "dimension") == 0) {
                fread(&box->dim, sizeof(int), 1, fp);
                if (box->dim != DIM)
                    Abort("The file is for");
                hasDimension = true;
            } else if (strcmp(str, "atoms") == 0) {

                fread(&particle->nAtom, sizeof(int), 1, fp);
                if (particle->nAtom <= 0)
                    Abort("Wrong File!");
                
                particle->pos =
                (doubleVector *)calloc(particle->nAtom, sizeof(doubleVector));
                particle->veloc =
                (doubleVector *)calloc(particle->nAtom, sizeof(doubleVector));
                particle->force =
                (doubleVector *)calloc(particle->nAtom, sizeof(doubleVector));
                particle->mass = (double *)calloc(particle->nAtom, sizeof(double));
                particle->img = (intVector *)calloc(particle->nAtom, sizeof(intVector));
                particle->type = (int *)calloc(particle->nAtom, sizeof(int));
                particle->diameterScale =
                (double *)calloc(particle->nAtom, sizeof(double));
                particle->id2tag = (int *)calloc(particle->nAtom, sizeof(int));
                particle->tag2id = (int *)calloc(particle->nAtom, sizeof(int));
            } else if (strcmp(str, "atom types") == 0) {
        printf("atom types\n");

                fread(&particle->nAtomType, sizeof(int), 1, fp);
                if (particle->nAtomType <= 0)
                    Abort("Wrong File!");
                
                particle->massPerType =
                (double *)calloc(particle->nAtomType, sizeof(double));
                for (int itype = 0; itype < particle->nAtomType; itype++)
                    particle->massPerType[itype] = 1.0;
            } else if (strcmp(str, "box Hvoigt") == 0) {
        printf("box Hvoigt\n");

                if (!hasDimension) {
                    box->dim = 3;
                    if (box->dim != DIM)
                        Abort("Dimension is not consistent!");
                    
                    fread(&box->boxH[spaceIdx2voigt(0, 0)], sizeof(double), 1, fp);  // xx
                    fread(&box->boxH[spaceIdx2voigt(1, 1)], sizeof(double), 1, fp);  // yy
                    fread(&box->boxH[spaceIdx2voigt(2, 2)], sizeof(double), 1, fp);  // zz
                    fread(&box->boxH[spaceIdx2voigt(1, 2)], sizeof(double), 1, fp);  // yz
                    fread(&box->boxH[spaceIdx2voigt(0, 2)], sizeof(double), 1, fp);  // xz
                    fread(&box->boxH[spaceIdx2voigt(0, 1)], sizeof(double), 1, fp);  // xy
                } else {
                    fread(&box->boxH, sizeof(uptriMat), 1, fp);
                }
            } else if (strcmp(str, "mean diameter") == 0) {
                printf("mean diameter\n");
                fread(&particle->meanDiameter, sizeof(double), 1, fp);
                hasMeanDiameter = true;
            } else if (strcmp(str, "Atoms") == 0) {
                printf("Atoms\n");
                for (int iatom = 0; iatom < particle->nAtom; iatom++) {
                    fread(&particle->type[iatom], sizeof(int), 1, fp);
                    fread(&particle->diameterScale[iatom], sizeof(double), 1, fp);
                    fread(&particle->pos[iatom], sizeof(doubleVector), 1, fp);
                    fread(&particle->img[iatom], sizeof(intVector), 1, fp);
                    
                    particle->mass[iatom] = particle->massPerType[particle->type[iatom]];
                    
                    particle->tag2id[iatom] = iatom;
                    particle->id2tag[iatom] = iatom;
                }
            } 
        }
        if (!hasMeanDiameter) {
            double meanDiameter = 0.0;
            for (int iatom = 0; iatom < particle->nAtom; iatom++) {
                meanDiameter += particle->diameterScale[iatom];
            }
            meanDiameter = meanDiameter / particle->nAtom;
            for (int iatom = 0; iatom < particle->nAtom; iatom++) {
                particle->diameterScale[iatom] /= meanDiameter;
            }
            particle->meanDiameter = meanDiameter;
        }
    } else if (strstr(str, "LAMMPS compatible data file.")) {
        // Lammps style data file
        box->dim = DIM;
        while (fgets(str, 4096, fp) != NULL) {
            if (strstr(str, "atoms") != NULL) {
                particle->nAtom = atoi(str);
                if (particle->nAtom <= 0)
                    Abort("No atom!");
                
                particle->pos =
                (doubleVector *)calloc(particle->nAtom, sizeof(doubleVector));
                particle->veloc =
                (doubleVector *)calloc(particle->nAtom, sizeof(doubleVector));
                particle->force =
                (doubleVector *)calloc(particle->nAtom, sizeof(doubleVector));
                particle->mass = (double *)calloc(particle->nAtom, sizeof(double));
                particle->img = (intVector *)calloc(particle->nAtom, sizeof(intVector));
                particle->type = (int *)calloc(particle->nAtom, sizeof(int));
                particle->diameterScale =
                (double *)calloc(particle->nAtom, sizeof(double));
                particle->id2tag = (int *)calloc(particle->nAtom, sizeof(int));
                particle->tag2id = (int *)calloc(particle->nAtom, sizeof(int));
            }
            if (strstr(str, "atom types") != NULL) {
                particle->nAtomType = atoi(str);
                if (particle->nAtomType <= 0)
                    Abort("Wrong DATA file!");
                
                particle->massPerType =
                (double *)calloc(particle->nAtomType, sizeof(double));
                for (int itype = 0; itype < particle->nAtomType; itype++)
                    particle->massPerType[itype] = 1.0;
            }
            
            if (strstr(str, "xlo xhi") != NULL) {
                double xlo, xhi;
                sscanf(str, "%lf %lf", &xlo, &xhi);
                box->boxH[spaceIdx2voigt(0, 0)] = xhi - xlo;
            }
            if (strstr(str, "ylo yhi") != NULL) {
                double ylo, yhi;
                sscanf(str, "%lf %lf", &ylo, &yhi);
                box->boxH[spaceIdx2voigt(1, 1)] = yhi - ylo;
            }
#if (DIM == 3)
            if (strstr(str, "zlo zhi") != NULL) {
                double zlo, zhi;
                sscanf(str, "%lf %lf", &zlo, &zhi);
                box->boxH[spaceIdx2voigt(2, 2)] = zhi - zlo;
            }
#endif
            if (strstr(str, "xy xz yz") != NULL) {
                double xy, xz, yz;
                sscanf(str, "%lf %lf %lf", &xy, &xz, &yz);
#if (DIM == 3)
                box->boxH[spaceIdx2voigt(1, 2)] = yz;
                box->boxH[spaceIdx2voigt(0, 2)] = xz;
                box->boxH[spaceIdx2voigt(0, 1)] = xy;
#elif (DIM == 2)
                box->boxH[spaceIdx2voigt(0, 1)] = xy;
#else
                Abort("Only 2D and 3D are vailid!");
#endif
            }
            
            if (strstr(str, "Atoms") != NULL) {
                double meanDiameter = 0.0;
                for (int iatom = 0; iatom < particle->nAtom;) {
                    if (feof(fp) && iatom < particle->nAtom)
                        Abort("Wrong dataFile!");
                    fgets(str, 4096, fp);
                    if (isEmpty(str, 4096))
                        continue;
                    
                    int num, id, type, ix = 0, iy = 0, iz = 0;
                    double diam, density, x, y, z;
                    
                    num = sscanf(str, "%d %d %lf %lf %lf %lf %lf %d %d %d", &id, &type,
                                 &diam, &density, &x, &y, &z, &ix, &iy, &iz);
                    if (num != 10)
                        Abort("Wrong format!");
                    if (id <= 0 || id > particle->nAtom)
                        Abort("Wrong atom ID.");
                    if (type <= 0 || type > particle->nAtomType)
                        Abort("Wrong atom type.");
                    
#if (DIM == 3)
                    particle->pos[id - 1][0] = x;
                    particle->pos[id - 1][1] = y;
                    particle->pos[id - 1][2] = z;
                    printf("%lf %lf %lf\n", x, y, z);
#elif (DIM == 2)
                    particle->pos[id - 1][0] = x;
                    particle->pos[id - 1][1] = y;
#endif
                    
                    particle->type[id - 1] = type - 1;
                    particle->mass[id - 1] =
                    particle->massPerType[particle->type[id - 1]];
                    particle->diameterScale[id - 1] = diam;
#if (DIM == 3)
                    particle->img[id - 1][0] = ix;
                    particle->img[id - 1][1] = iy;
                    particle->img[id - 1][2] = iz;
#elif (DIM == 2)
                    particle->img[id - 1][0] = ix;
                    particle->img[id - 1][1] = iy;
#endif
                    
                    particle->tag2id[id - 1] = id - 1;
                    particle->id2tag[id - 1] = id - 1;
                    meanDiameter += diam;
                    
                    iatom++;
                }
                meanDiameter = meanDiameter / particle->nAtom;
                for (int iatom = 0; iatom < particle->nAtom; iatom++) {
                    particle->diameterScale[iatom] /= meanDiameter;
                }
                particle->meanDiameter = meanDiameter;
            }
            if (strstr(str, "Velocities") != NULL) {
                for (int iatom = 0; iatom < particle->nAtom;) {
                    if (feof(fp) && iatom < particle->nAtom)
                        Abort("Wrong dataFile!");
                    fgets(str, 4096, fp);
                    if (isEmpty(str, 4096))
                        continue;
                    
                    int num, id;
                    double vx, vy, vz;
                    num = sscanf(str, "%d %lf %lf %lf", &id, &vx, &vy, &vz);
                    
#if (DIM == 3)
                    particle->veloc[id - 1][0] = vx;
                    particle->veloc[id - 1][1] = vy;
                    particle->veloc[id - 1][2] = vz;
#elif (DIM == 2)
                    particle->veloc[id - 1][0] = vx;
                    particle->veloc[id - 1][1] = vy;
#endif
                    
                    iatom++;
                }
            }
        }
    } else
        Abort("Wrong File!");
    fclose(fp);
    
    box->isShapeFixed = true;
    particle->isSizeFixed = true;
    particle->isForceValid = false;
    // update->Edone = update->Pdone = update->Tdone = false;
    // update->nebrList.isValid = false;
    // update->nebrList.compelInit = true;
    // update->nebrList.nForce = 0;
    // update->nebrList.cntForce = 0;
    // update->nebrList.doSort = true;
    
    // setBoxPara(box);
    // initConfInfo(box, particle, update);
    // setUnits(update, particle->meanDiameter);
    // adjustImg(box, particle);
}

// void readConf(Box *box, Particle *particle, Update *update, Variable *var) {
//     update->nebrList.skinSet = -1.0;
//     update->nebrList.nDelay = -1;
    
//     int cntR = 0;
//     cmdArg *cmd = findVariable(var, "--rf");
//     if (cmd) {
//         readConf_data(box, particle, update, var);
//         cntR++;
//     }
//     cmd = findVariable(var, "--rd");
//     if (cmd && cmd->cmdArgc == 2) {
//         if (cntR == 1)
//             Abort("--rf or --rd");
//         readConf_dump(box, particle, update, var);
//         cntR++;
//     }
//     if (cntR != 1)
//         Abort("--rd dump.bin step or --rf conf.bin");  //--rd dump.bin or no "--rd"
//     // and "--rf"
    
//     cmd = findVariable(var, "--skin");
//     if (cmd) {
//         if (cmd->cmdArgc <= 0)
//             Abort("--skin 0.1");
//         update->nebrList.skinSet = atof(cmd->cmdArgv[0]) / 1.1;
//     } else {
//         update->nebrList.skinSet = 0.1 / 1.1;
//     }
    
// #ifdef __orthBox__
//     for (int idim = 0; idim < DIM; idim++) {
//         for (int jdim = idim + 1; jdim < DIM; jdim++) {
//             if (fabs(box->boxH[spaceIdx2voigt(idim, jdim)]) >= 5E-16)
//                 Abort("Not orthogonal Box!");
//             box->boxH[spaceIdx2voigt(idim, jdim)] = 0;
//         }
//     }
// #endif
    
//     if (screenOutputReadConf) {
//         printf("===========System Info==========\n");
//         printf("Dimension: %d\n", DIM);
//         printf("Number of Particles: %d;\n", particle->nAtom);
//         printf("Volume Fraction: %g;\n", update->volFrac);
//         printf("Min(diameter): %g;\nMax(diameter): %g;\n",
//                update->nebrList.minDiameterScale,
//                update->nebrList.maxDiameterScale);
//         printf("Edges of Simulation Box: \n");
//         for (int iedge = 0; iedge < DIM; iedge++) {
//             printf("\t");
//             for (int jdim = 0; jdim < DIM; jdim++) {
//                 printf("%-8.6e\t", box->boxEdge[iedge][jdim] / update->distanceUnits);
//             }
//             printf("\n");
//         }
// #if defined(Harmonic)
//         printf("Interaction between particles: Harmonic\n");
// #elif defined(Hertzian)
//         printf("Interaction between particles: Hertzian\n");
// #else
// #error "No Interaction is defined!"
// #endif
//         printf("===========System Info==========\n");
        
//         if (screenOutputReadConf >= 10000)
//             screenOutputReadConf -= 10000;
//     }
// }