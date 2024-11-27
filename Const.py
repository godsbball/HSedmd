class Const:
        def __init__(self,):
                pass
        def gppdynamics(num):
                return  "g++ -DN=%d -o mdgrow main.cpp mdgrow.cpp mdcell.cpp read_particle_bin.c   -lm"%num


#     gccPre = "gcc -o genrem genrem2.c -lgsl -lgslcblas -lm -lrt; \
#             gcc -fPIC -shared -o libdos.so dos2.c -lm; " # ; g++ -o tree_barrier tree_barrier.cpp -lgsl -lgslcblas -lm; gcc -fPIC -shared -o libdos.so dos2.c -lm; "  
#     gppSimple = "g++ -o remrw10 remrw_simple.cpp basin.cpp tree.cpp -std=c++17 -lgsl -lgslcblas -lm -lrt;" + gccPre
#     gppActivationSimple = "g++ -o remrw_activation_simple remrw_activation_simple.cpp basin.cpp tree.cpp -std=c++17 -lgsl -lgslcblas -lm -lrt;"  + gccPre
#     gppObs = "g++ -o remrw_obs_t remrw_obs_t.cpp basin.cpp tree.cpp -std=c++17 -lgsl -lgslcblas -lm -lrt;" + gccPre
#     gppCorrelationTdtw = "g++ -o remrw_corr_t remrw_corr_t.cpp basin.cpp tree.cpp -std=c++17 -lgsl -lgslcblas -lm -lrt;" + gccPre
#     gppNextBasin = "g++ -o rem_nextBasin_t rem_nextBasin_t.cpp basin.cpp tree.cpp -std=c++17 -lgsl -lgslcblas -lm -lrt;" + gccPre
#     gppTreeBarrier = "g++ -o tree_barrier2 tree_barrier2.cpp basin.cpp tree.cpp -std=c++17 -lgsl -lgslcblas -lm -lrt;" + gccPre
    
#     nvccCurand = "nvcc curand.c  -lcurand  -o curand"