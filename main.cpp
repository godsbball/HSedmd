#include "mdgrow.hpp"

int main(int argc, char **argv){
    double unit_time = 1e-3;
    int STEP_MAX = 35;
    double phi, phi0;
    double print_tw, print_tptw, target_time;
    double print_time_ratio = 1.5, tw_ratio = 1.5;
    int samples=6, replica=1000;
    double MSD_tw_tptw = 0., MSD_ab =0., MSD_ab_half =0.,
        MSD_0_tptw = 0., MSD_0_tptw_half = 0., correlation_tw_tptw = 0., 
        correlation_tptw_self = 0., correlation = 0., delta = 0.,
        delta_0_tptw, correlation_0_tptw, delta_tw_tptw;
    long *idx = new long[N];
    double r_arr[N];
    char fname[1024] = "N125/phig630/equHSrho630N125_s1040.bin";
    char fname_out[1024];
    if (argc != 6){
        printf("input error\n");
        exit(1);
    }
    char **tailptr = nullptr;
    phi = strtod(argv[1], tailptr);
    samples = strtol(argv[2], tailptr, 0);
    replica = strtol(argv[3], tailptr, 0);
    STEP_MAX = strtol(argv[4], tailptr, 0);
    phi0 = strtod(argv[5], tailptr);
    sparticle *particles[samples];
    pos_t pos_tw[N], posbar_tptw[N], pos_0_samples[samples][N];
    // replica = 1040;
    // samples = 6;
    sprintf(fname,"N%d/phig%d/equHSrho%dN%d_s%d.bin",N,int(phi0*1000),int(phi0*1000),N,replica);
    sprintf(fname_out,"../data/N%d_phi=%.2le_StepMax=%02d/-t_samples=%06d/replica=%06d.dat",N,phi,STEP_MAX,samples,replica);
    FILE *fout = fopen(fname_out,"w+");
    fprintf(fout, "#%-15s %-15s %-10s %-15s %-25s %-15s %-15s %-25s %-25s %-25s %-25s\n",
        "print_tw", "MSD_0_tptw", "MSD_ab", "delta_0_tptw", "correlation_0_tptw",
        "MSD_tw_tptw", "delta_tw_tptw", "correlation_tptw_self", "correlation_tw_tptw + self"
        , "MSD_0_tptw_half","MSD_ab_half");
    // printf("%d %d %d %d\n",sizeof(MDGrow),sizeof(mdgrow),sizeof(sparticle),sizeof(particles));

    // mdgrow.init_read_confs(fname);

    std::vector<MDGrow> mdgrow_vec0;
    mdgrow_vec0.reserve(samples);
    for (int spl = 0; spl < samples; ++spl) {
        mdgrow_vec0.emplace_back(1e-3, phi); 
        mdgrow_vec0[spl].init_read_confs(fname);
    }//construct class and init confs from fname.
    Box_t box;
    box = mdgrow_vec0[0].get_box();

    // To reach the target packing fraction:
    target_time = mdgrow_vec0[0].get_target_time();
    for (int spl=0; spl<samples; spl++){
        particles[spl] = mdgrow_vec0[spl].get_partciles_at_print_time(target_time);
    }

    //construct the idx of sorted radius 
    for(int i=0;i<N;i++){
        r_arr[i] = particles[0][i].r;
    }
    argsort(r_arr,N,idx);


    //calculate overlap after reaching target packing fraction
    for(int spl=0; spl<samples; spl++){
        int idx = 0;
        int j;
        for(j=0; j<N; ++j){
            idx = mdgrow_vec0[spl].overlap(particles[spl]+j);
            if(idx) break;
        }
        particles[spl] = mdgrow_vec0[spl].get_paritcles();
        printf("sample %d: overlap = %d   %d. T = %lf\n",spl, idx, j,mdgrow_vec0[spl].get_temperature());
        printf("sample %d: %d   %d\n",spl, idx, j);
    }
    fflush(stdout);





    //nve for decrease overlap?
    double nooverlap_time = target_time;

    //pos at tw=t=0
    for(int spl=0; spl<samples; spl++){
        for(int j=0; j<N; j++){
            pos_0_samples[spl][j] = {
                    particles[spl][j].x, particles[spl][j].y, particles[spl][j].z
                    , particles[spl][j].r
                    , particles[spl][j].boxestraveledx
                    , particles[spl][j].boxestraveledy
                    , particles[spl][j].boxestraveledz
                };
        }
    }

    std::vector<MDCell> mdcell_vec;
    mdcell_vec.reserve(samples);
    for (int spl = 0; spl < samples; ++spl) {
        mdcell_vec.emplace_back();
        mdcell_vec[spl].init(box,particles[spl]);
    }//construct class and init confs from fname.
    mdgrow_vec0.clear();



    target_time = 0.;
    particle_c *particles_c[samples];
    // NVE evolution
    for(int i = 0; i<STEP_MAX; i++){
        print_tw = pow(print_time_ratio,1.*i)*unit_time + target_time;
        print_tptw = print_tw * tw_ratio + target_time;
        printf("%le\n",print_tw - target_time);

        //init summation quantities
        for(int j=0; j<N; j++){
            posbar_tptw[j] = {0, 0, 0} ;
        }
        MSD_tw_tptw = 0;
        MSD_0_tptw = 0.;
        MSD_0_tptw_half = 0.;
        correlation_tw_tptw = 0;
        correlation_0_tptw = 0;
        for (int spl=0; spl<samples; spl++){
            //get positions at tw and tptw
            if (print_time_ratio - tw_ratio>1e-10 || i==0){
                particles_c[spl] = mdcell_vec[spl].get_partciles_at_print_time(print_tw);
            }else if(print_time_ratio - tw_ratio < -1e-6) {
                printf("print_time_ratio should > tw_ratio");
                exit(1);
            }

            for(int j=0; j<N; j++){
                pos_tw[j] = {
                    particles_c[spl][j].x, particles_c[spl][j].y, particles_c[spl][j].z
                    , particles_c[spl][j].r
                    , particles_c[spl][j].boxestraveledx
                    , particles_c[spl][j].boxestraveledy
                    , particles_c[spl][j].boxestraveledz
                };
            }
            particles_c[spl] = mdcell_vec[spl].get_partciles_at_print_time(print_tptw);
            //calculate self quantities at tptw: MSD, correlation, position vector
            MSD_tw_tptw += MSD_2time_1spl(pos_tw, particles_c[spl],box)/samples;
            MSD_0_tptw += MSD_2time_1spl(pos_0_samples[spl], particles_c[spl],box)/samples;
            MSD_0_tptw_half += MSD_2time_half_1spl(pos_0_samples[spl], particles_c[spl],box, idx)/samples;
            correlation_tw_tptw += correlation_2time_1spl(pos_tw, particles_c[spl],box)/samples;
            correlation_0_tptw += correlation_2time_1spl(pos_0_samples[spl], particles_c[spl],box)/samples;
            for(int j=0; j<N; j++){
                posbar_tptw[j].x += (pos_tw[j].x + pos_tw[j].boxestraveledx*box.xsize)/samples;
                posbar_tptw[j].y += (pos_tw[j].y + pos_tw[j].boxestraveledy*box.ysize)/samples;
                posbar_tptw[j].z += (pos_tw[j].z + pos_tw[j].boxestraveledz*box.zsize)/samples;
            }
            printf("sample %d: overlap = %d\n",spl, mdcell_vec[spl].overlap(particles_c[spl]));
        }
        //calculate final quantities and output
        MSD_ab = MSD_AB(particles_c, box, samples);

        MSD_ab_half = MSD_AB_half(particles_c,box,samples,idx);
        correlation_tptw_self = correlation_self(posbar_tptw);
        correlation_tw_tptw = correlation_tw_tptw - correlation_tptw_self;
        correlation_0_tptw = correlation_0_tptw - correlation_tptw_self;
        delta_0_tptw = MSD_ab - MSD_0_tptw;
        delta_tw_tptw = MSD_ab - MSD_tw_tptw;
        fprintf(fout, "%-15le %-15le %-10le %-15le %-25le %-15le %-15le %-25le %-25le %-25le %-25le\n",
            print_tw - target_time, 
            MSD_0_tptw, MSD_ab, delta_0_tptw, correlation_0_tptw, 
            MSD_tw_tptw, delta_tw_tptw,
            correlation_tptw_self, correlation_tw_tptw + correlation_tptw_self, MSD_0_tptw_half,MSD_ab_half);
        fflush(fout);

    }

    mdcell_vec.clear();
    delete[] idx;
    fclose(fout);
    return 0;
}