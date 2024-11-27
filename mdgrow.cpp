#include "mdgrow.hpp"
#include "read_particle_bin.h"



// int main()
// {
//     MDGrow mdgrow(0.1, 0.58);
//     mdgrow.run();
//     return 0;
// }


// int main(int argc, char **argv){
//     double unit_time = 1e-3;
//     int STEP_MAX = 35;
//     double phi;
//     double print_tw, print_tptw, target_time;
//     double print_time_ratio = 1.5, tw_ratio = 1.5;
//     int samples=6, replica=1000;
//     sparticle *particles[samples];
//     pos_t pos_tw[N], posbar_tptw[N], pos_0[samples][N];
//     double MSD_tw_tptw = 0., MSD_ab =0., 
//         MSD_0_tptw = 0., correlation_tw_tptw = 0., 
//         correlation_tptw_self = 0., correlation = 0., delta = 0.,
//         delta_0_tptw, correlation_0_tptw, delta_tw_tptw;
//     char fname[1024] = "N125/phig630/equHSrho630N125_s1040.bin";
//     char fname_out[1024];
//     // if (argc != 5){
//     //     printf("input error\n");
//     //     exit(1);
//     // }
//     char **tailptr = nullptr;
//     phi = strtod(argv[1], tailptr);
//     samples = strtol(argv[2], tailptr, 0);
//     replica = strtol(argv[3], tailptr, 0);
//     STEP_MAX = strtol(argv[4], tailptr, 0);
//     // phi = 0.63;
//     // replica = 1040;
//     // samples = 6;
//     sprintf(fname,"N%d/phig%d/equHSrho%dN%d_s%d.bin",N,int(phi*1000),int(phi*1000),N,replica);
//     sprintf(fname_out,"../data/N%d_phi=%.2le_StepMax=%02d/-t_samples=%06d/replica=%06d.dat",N,phi,STEP_MAX,samples,replica);
//     FILE *fout = fopen(fname_out,"w+");
//     fprintf(fout, "#%-15s %-15s %-10s %-15s %-25s %-15s %-15s %-25s %-25s\n",
//         "print_tw", "MSD_0_tptw", "MSD_ab", "delta_0_tptw", "correlation_0_tptw",
//         "MSD_tw_tptw", "delta_tw_tptw", "correlation_tptw_self", "correlation_tw_tptw + self");

//     // printf("%d %d %d %d\n",sizeof(MDGrow),sizeof(mdgrow),sizeof(sparticle),sizeof(particles));

//     // mdgrow.init_read_confs(fname);

//     std::vector<MDGrow> mdgrow_vec0;
//     mdgrow_vec0.reserve(samples);
//     for (int spl = 0; spl < samples; ++spl) {
//         mdgrow_vec0.emplace_back(5e-4, 0.67); 
//         mdgrow_vec0[spl].init_read_confs(fname);
//     }//construct class and init confs from fname.


//     // To reach the target packing fraction:
//     target_time = mdgrow_vec0[0].get_target_time();
//     for (int spl=0; spl<samples; spl++){
//         particles[spl] = mdgrow_vec0[spl].get_partciles_at_print_time(target_time);
//     }


//     for(int spl=0; spl<samples; spl++){
//         int idx = 0;
//         int j;
//         for(j=0; j<N; ++j){
//             idx = mdgrow_vec0[spl].overlap(particles[spl]+j);
//             if(idx) break;
//         }
//         printf("sample %d: overlap = %d   %d\n",spl, idx, j);
//     }
//     // mdgrow_vec.clear();


//     std::vector<MDGrow> mdgrow_vec;
//     mdgrow_vec.reserve(samples);
//     for (int spl = 0; spl < samples; ++spl) {
//         mdgrow_vec.emplace_back(0.0, 0.64); 
//         mdgrow_vec[spl].init_copy_confs(particles[spl]);
//     }//construct class and init confs from fname.

//     //transfer to NVT and evolution:
//     // for (int spl=0; spl<samples; spl++){
//     //     // mdgrow_vec[spl].nvt(target_time+1000);
//     //     mdgrow_vec[spl].quench();
//     //     // mdgrow_vec[spl].run();
//     //     printf("sample %d: phi=%lf T=%lf\n",spl,mdgrow_vec[spl].get_packing_fraction(),mdgrow_vec[spl].get_temperature());
//     // }

//     //transfer to NVE:
//     // for (int spl=0; spl<samples; spl++){
//     //     mdgrow_vec[spl].nve();
//     // }

//     for(int spl=0; spl<samples; spl++){
//         particles[spl] = mdgrow_vec[spl].get_paritcles();
//         for(int j=0; j<N; j++){
//             pos_0[spl][j] = {particles[spl][j].x, particles[spl][j].y, particles[spl][j].z};
//         }
//     }
//     //     int idx = 0;
//     //     int j;
//     //     for(j=0; j<N; ++j){
//     //         idx = mdgrow_vec[spl].overlap(particles[spl]+j);
//     //         if(idx) break;
//     //     }
//     //     printf("sample %d: overlap = %d   %d\n",spl, idx, j);
//     // }

//     target_time = 0.;
//     // NVE evolution
//     for(int i = 0; i<STEP_MAX; i++){
//         print_tw = pow(print_time_ratio,1.*i)*unit_time + target_time;
//         print_tptw = print_tw * tw_ratio + target_time;
//         printf("%le\n",print_tw - target_time);

//         //init summation quantities
//         for(int j=0; j<N; j++){
//             posbar_tptw[j] = {0, 0, 0} ;
//         }
//         MSD_tw_tptw = 0;
//         MSD_0_tptw = 0.;
//         correlation_tw_tptw = 0;
//         correlation_0_tptw = 0;
//         for (int spl=0; spl<samples; spl++){
//             //get positions at tw and tptw
//             if (print_time_ratio - tw_ratio>1e-10 || i==0){
//                 particles[spl] = mdgrow_vec[spl].get_partciles_at_print_time(print_tw);
//             }else if(print_time_ratio - tw_ratio < -1e-6) {
//                 printf("print_time_ratio should > tw_ratio");
//                 exit(1);
//             }

//             for(int j=0; j<N; j++){
//                 pos_tw[j] = {particles[spl][j].x, particles[spl][j].y, particles[spl][j].z};
//             }
//             particles[spl] = mdgrow_vec[spl].get_partciles_at_print_time(print_tptw);
//             //calculate self quantities at tptw: MSD, correlation, position vector
//             MSD_tw_tptw += MSD_2time_1spl(pos_tw, particles[spl])/samples;
//             MSD_0_tptw += MSD_2time_1spl(pos_0[spl], particles[spl])/samples;
//             correlation_tw_tptw += correlation_2time_1spl(pos_tw, particles[spl])/samples;
//             correlation_0_tptw += correlation_2time_1spl(pos_0[spl], particles[spl])/samples;
//             for(int j=0; j<N; j++){
//                 posbar_tptw[j].x += pos_tw[j].x/samples;
//                 posbar_tptw[j].y += pos_tw[j].y/samples;
//                 posbar_tptw[j].z += pos_tw[j].z/samples;
//             }
//             printf("sample %d: overlap = %d\n",spl, mdgrow_vec[spl].overlap(particles[spl]));

//         }
//         //calculate final quantities and output
//         MSD_ab = MSD_AB(particles, samples);
//         correlation_tptw_self = correlation_self(posbar_tptw);
//         correlation_tw_tptw = correlation_tw_tptw - correlation_tptw_self;
//         correlation_0_tptw = correlation_0_tptw - correlation_tptw_self;
//         delta_0_tptw = MSD_ab - MSD_0_tptw;
//         delta_tw_tptw = MSD_ab - MSD_tw_tptw;
//         fprintf(fout, "%-15le %-15le %-10le %-15le %-25le %-15le %-15le %-25le %-25le\n",
//             print_tw - target_time, 
//             MSD_0_tptw, MSD_ab, delta_0_tptw, correlation_0_tptw, 
//             MSD_tw_tptw, delta_tw_tptw,
//             correlation_tptw_self, correlation_tw_tptw + correlation_tptw_self);
//         fflush(fout);
//     }

//     mdgrow_vec.clear();
//     fclose(fout);
//     return 0;
// }
template<class T>
void argsort(T *array, long num, long *index) {
    const auto function = [array](long a, long b) noexcept -> bool {
        return array[a] < array[b];
    };
    assert(num < INT_MAX);
    long *temp = new long[num];
    std::iota(temp, temp + num, 0);
    std::sort(temp, temp + num, function);
    memcpy(index, temp, num * sizeof(long));
    delete[] temp;
}
template void argsort(double *array, long num, long *index);

int MDGrow::run()
{
    // init();
    // if (stop) return 1;				//Stop is set to 1 whenever the simulation should stop
    // printf("Starting\n");
    while (!stop)
    {
        step();
    }
    printstuff();
    return 0;
}

double MSD_2time_1spl(pos_t *pos_tw, particle_c *p_tptw, Box_t box){
    double MSD = 0;
    double dx,dy,dz;
    for(int i = 0; i<N; i++){
        dx = pos_tw[i].x + pos_tw[i].boxestraveledx*box.xsize - p_tptw[i].x - p_tptw[i].boxestraveledx*box.xsize;
        dy = pos_tw[i].y + pos_tw[i].boxestraveledy*box.ysize - p_tptw[i].y - p_tptw[i].boxestraveledy*box.ysize;
        dz = pos_tw[i].z + pos_tw[i].boxestraveledz*box.zsize - p_tptw[i].z - p_tptw[i].boxestraveledz*box.zsize;
        MSD += (dx*dx + dy*dy + dz*dz);
    }
    MSD = MSD / N;
    return MSD;
}

double MSD_2time_half_1spl(pos_t *pos_tw, particle_c *p_tptw, Box_t box, long *idx){
    double MSD = 0;
    double dx,dy,dz;
    int i;
    for(int j = 0; j<N/2; j++){
        i = idx[N-j-1];
        dx = pos_tw[i].x + pos_tw[i].boxestraveledx*box.xsize - p_tptw[i].x - p_tptw[i].boxestraveledx*box.xsize;
        dy = pos_tw[i].y + pos_tw[i].boxestraveledy*box.ysize - p_tptw[i].y - p_tptw[i].boxestraveledy*box.ysize;
        dz = pos_tw[i].z + pos_tw[i].boxestraveledz*box.zsize - p_tptw[i].z - p_tptw[i].boxestraveledz*box.zsize;
        MSD += (dx*dx + dy*dy + dz*dz);
    }
    MSD = MSD / int(N/2);
    return MSD;
}

double correlation_2time_1spl(pos_t *pos_tw, particle_c *p_tptw, Box_t box){
    double cor = 0;
    double x1x2,y1y2,z1z2;
    for(int i = 0; i<N; i++){
        x1x2 =  (pos_tw[i].x + pos_tw[i].boxestraveledx*box.xsize) * (p_tptw[i].x + p_tptw[i].boxestraveledx*box.xsize);
        y1y2 =  (pos_tw[i].y + pos_tw[i].boxestraveledy*box.ysize) * (p_tptw[i].y + p_tptw[i].boxestraveledy*box.ysize);
        z1z2 =  (pos_tw[i].z + pos_tw[i].boxestraveledz*box.zsize) * (p_tptw[i].z + p_tptw[i].boxestraveledz*box.zsize);
        cor += x1x2 + y1y2 + z1z2;
    }
    cor = cor / N;
    return cor;
}

double correlation_self(pos_t *posbar_tptw){
    double cor = 0;
    // double x,y,z;
    for(int i = 0; i<N; i++){
        cor += posbar_tptw[i].x * posbar_tptw[i].x + posbar_tptw[i].y * posbar_tptw[i].y + posbar_tptw[i].z * posbar_tptw[i].z;
    }
    cor = cor / N;
    return cor;
}


double MSD_AB(particle_c **ps_tptw, Box_t box, int samples){
    double MSD = 0;
    double dx,dy,dz;
    for(int i = 0; i<samples; i++){
        for(int j = 0; j<i; j++){
            for(int k = 0; k<N; k++){
                dx = ps_tptw[i][k].x + ps_tptw[i][k].boxestraveledx*box.xsize - ps_tptw[j][k].x - ps_tptw[j][k].boxestraveledx*box.xsize;
                dy = ps_tptw[i][k].y + ps_tptw[i][k].boxestraveledy*box.ysize - ps_tptw[j][k].y - ps_tptw[j][k].boxestraveledy*box.ysize;
                dz = ps_tptw[i][k].z + ps_tptw[i][k].boxestraveledz*box.zsize - ps_tptw[j][k].z - ps_tptw[j][k].boxestraveledz*box.zsize;
                MSD += (dx * dx + dy * dy + dz * dz);            
            }
        }
    }
    MSD = 2 * MSD / samples / (samples-1) / N;
    return MSD;
}

double MSD_AB_half(particle_c **ps_tptw, Box_t box, int samples, long *idx){
    double MSD = 0;
    int k;
    double dx,dy,dz ;
    for(int i = 0; i<samples; i++){
        for(int j = 0; j<i; j++){
            for(int usk = 0; usk<N/2; usk++){
                k = idx[N-usk-1];
                dx = ps_tptw[i][k].x + ps_tptw[i][k].boxestraveledx*box.xsize - ps_tptw[j][k].x - ps_tptw[j][k].boxestraveledx*box.xsize;
                dy = ps_tptw[i][k].y + ps_tptw[i][k].boxestraveledy*box.ysize - ps_tptw[j][k].y - ps_tptw[j][k].boxestraveledy*box.ysize;
                dz = ps_tptw[i][k].z + ps_tptw[i][k].boxestraveledz*box.zsize - ps_tptw[j][k].z - ps_tptw[j][k].boxestraveledz*box.zsize;
                MSD += (dx * dx + dy * dy + dz * dz);
            }
        }
    }
    MSD = 2 * MSD / samples / (samples-1) / int(N/2);
    return MSD;
}

double MSD_AB_half2(particle_c *ps_tptw, Box_t box, int samples, long *idx){
    double MSD = 0;
    int k;
    double dx,dy,dz;
    for(int i = 0; i<samples; i++){
        for(int j = 0; j<i; j++){
            for(int usk = 0; usk<N/2; usk++){
                k = idx[N-usk-1];
                dx = ps_tptw[ i*N +k ].x + ps_tptw[ i*N +k ].boxestraveledx*box.xsize - ps_tptw[ j*N +k ].x - ps_tptw[ j*N +k ].boxestraveledx*box.xsize;
                dy = ps_tptw[ i*N +k ].y + ps_tptw[ i*N +k ].boxestraveledy*box.ysize - ps_tptw[ j*N +k ].y - ps_tptw[ j*N +k ].boxestraveledy*box.ysize;
                dz = ps_tptw[ i*N +k ].z + ps_tptw[ i*N +k ].boxestraveledz*box.zsize - ps_tptw[ j*N +k ].z - ps_tptw[ j*N +k ].boxestraveledz*box.zsize;
                MSD += (dx * dx + dy * dy + dz * dz);
            }
        }
    }
    MSD = 2 * MSD / samples / (samples-1) / int(N/2);
    return MSD;
}

void MDGrow::nvt(double maxt_){
    sparticle *p;
    maxt = maxt_;
    growthspeed = 0.;
    usethermostat = 1;
    stop = 0;
    thermostat(nullptr);
    for (int i = 0; i < N; i++) {
        p = &(particles[i]);
        p->vr = growthspeed * p->r;
        // p->t = 0;						//r and v known at t=0
    }
}

void MDGrow::quench(){
    sparticle *p;
    growthspeed = 0.;
    nempty = 0; 
    for (int i = 0; i < N; i++) {
        p = &(particles[i]);
        p->vr = growthspeed * p->r;
        // p->t = 0;						//r and v known at t=0
    }
    clear_except_particles();
    assign_except_particles();
    init_varibles();
    init_particles();
    init_without_confs();
}

void MDGrow::init_varibles(){
    maxscheduletime = 5;
    eventlisttime = 2 / (float)N;
    maxdt = 1;
    //Neighbor lists
    shellsize = 1.5;
    //Internal variables
    time = 0;
    maxt = -1;
    reftime = 0;
    currentlist = 0;
    listcounter1 = 0, listcounter2 = 0, mergecounter = 0;
    nempty = 0;
    stop = 0;
    dvtot = 0;
}
void MDGrow::nve(){
    sparticle *p;
    growthspeed = 0.;
    usethermostat = 1;
    for (int i = 0; i < N; i++) {
        p = &(particles[i]);
        p->vr = growthspeed * p->r;
        // p->t = 0;						//r and v known at t=0
    }
}

void MDGrow::update_to_print_time(particle* p1, double print_time)
{
    double dt = print_time - p1->t;
    p1->t = print_time;
    p1->x += dt * p1->vx;
    p1->y += dt * p1->vy;
    p1->z += dt * p1->vz;
    p1->r += dt * p1->vr;
}
sparticle *MDGrow::get_partciles_at_print_time(double print_time){
    // sparticle *particles;
    maxt = print_time;
    printf("%lf\n",time);
    stop = run_2(); // determine   time (now) < print_time < ev->time (next collision time)

    for(int i=0; i<N; i++){
        update_to_print_time(particles + i, print_time);
    }
    return particles;
}

MDGrow::MDGrow(double growthspeed_, double targetpackfrac_)
:rng_64(std::random_device{}()),double_01(0,1), normal(0,1){
    // N = N_;
    // sevent ev_null = {0., 
    //     nullptr, nullptr,
    //     nullptr, nullptr, 
    //     nullptr, nullptr, 
    //     nullptr,0,0,0};
    growthspeed = growthspeed_;
    targetpackfrac = targetpackfrac_;
    particles = (sparticle *)malloc(sizeof(sparticle) * (N ));
    // particles = new sparticle[N];
    eventlists = (sevent **) malloc(sizeof(sevent*) * (MAXNUMEVENTLISTS + 1));
    eventlist = (sevent *) malloc(sizeof(sevent) * MAXEVENTS);
    eventpool = (sevent **) malloc(sizeof(sevent*) * (MAXEVENTS));
    for(int i=0; i<MAXEVENTS; i++){
        // eventpool[i] = (event *) malloc(sizeof(event) * (MAXEVENTS));
        eventpool[i] = nullptr;
        // eventlists[i] = (event *) malloc(sizeof(event) * (MAXNUMEVENTLISTS + 1));
        eventlists[i] = nullptr;
    }
    // particles = new sparticle[N];
    // eventlist = new sevent[MAXEVENTS];
    // eventlist = (sevent *)malloc()
    // particles2 = new sparticle[N];
}

MDGrow::~MDGrow(){
    int i;
    particles = nullptr;
    free(particles);

    for(i=0;i<cx*cy*cz;i++) celllist[i] = nullptr;
    celllist = nullptr;
    free(celllist);

    for(i=0;i<(MAXNUMEVENTLISTS + 1);i++) eventlists[i] = nullptr;
    eventlists = nullptr;
    free(eventlists);

    eventlist = nullptr;//??
    free(eventlist);

    for(i=0;i<MAXEVENTS;i++) eventpool[i] = nullptr;
    eventpool = nullptr;
    free(eventpool);
    // for(int i=0; i<MAXEVENTS; i++){
    //     free(eventpool[i]);
    //     free(eventpool[i]);
    // }

    // delete[] particles;
    // // delete[] particles2;
    // delete[] eventlist;
}

void MDGrow::clear_except_particles(){
    int i;
    // event *ev_null = {0., 
    //     nullptr, nullptr,
    //     nullptr, nullptr, 
    //     nullptr, nullptr, 
    //     nullptr,0,0,0};
    for(i=0;i<cx*cy*cz;i++) celllist[i] = nullptr;
    free(celllist);

    for(i=0;i<(MAXNUMEVENTLISTS + 1);i++) eventlists[i] = nullptr;
    free(eventlists);

    eventlist = nullptr;//??
    // for(i=0;i<MAXEVENTS;i++) eventlist[i] = ev_null;
    free(eventlist);

    for(i=0;i<MAXEVENTS;i++) eventpool[i] = nullptr;
    free(eventpool);
}
void MDGrow::assign_except_particles(){
    eventlists = (event **) malloc(sizeof(event*) * (MAXNUMEVENTLISTS + 1));
    eventlist = (event *) malloc(sizeof(event) * MAXEVENTS);
    eventpool = (event **) malloc(sizeof(event*) * (MAXEVENTS));
}
void MDGrow::init_particles(){
    for(int i=0;i<N;i++){
        particles[i].firstcollision = nullptr;
        particles[i].prev = nullptr;
        particles[i].next = nullptr;
        for(int j=0;j<MAXNEIGH;j++){
            particles[i].neighbors[j] = nullptr;
        }
    }
}
int MDGrow::run_2()
{
    while (!stop)
    {
        step();
    }
    printstuff();
    return 0;
}

double MDGrow::get_target_time(){
    return target_time;
}

double MDGrow::get_time(){
    return time;
}
double MDGrow::get_packing_fraction(){
    double sum_ball_volume = 0.;
    double radius = 0.0;

    for (int i = 0; i < N; i++) {
        radius = particles[i].r;
        sum_ball_volume += radius*radius*radius*4/3*M_PI;
    }
    return sum_ball_volume/(xsize * ysize * zsize);
}

double MDGrow::get_temperature(){
    double en = 0.;
    sparticle *p;
    for (int i = 0; i < N; i++)
    {
        p = particles + i;
        // update(p);
        en += p->mass * (p->vx * p->vx + p->vy * p->vy + p->vz * p->vz);
    }
    temperature = 0.5 * en / (float)N / 1.5;
    return temperature;
}

sparticle *MDGrow::get_paritcles(){
    return particles;
}




pos_t *MDGrow::get_pos_deep(pos_t *pos){
    for (int i = 0; i < N; i++) {
        pos[i].x = particles[i].x;
        pos[i].y = particles[i].y;
        pos[i].z = particles[i].z;
        pos[i].r = particles[i].r;
    }
    return pos;
}

particle_c *MDGrow::get_particles_c_deep(particle_c *p){
    for (int i = 0; i < N; i++) {
        p[i].x = particles[i].x;
        p[i].y = particles[i].y;
        p[i].z = particles[i].z;
        p[i].r = particles[i].r;
        p[i].t = particles[i].t;
        p[i].boxestraveledx = particles[i].boxestraveledx;
        p[i].boxestraveledy = particles[i].boxestraveledy;
        p[i].boxestraveledz = particles[i].boxestraveledz;
    }
    return p;
}

int MDGrow::read_confs(char *fname){
    /* 
     THE FILE FORMAT IS BINARY
     In conf data, particle position is -0.5->0.5 
     */
    Box *box = (Box *)calloc(1, sizeof(Box));
    Particle *particle_ = (Particle *)calloc(1, sizeof(Particle));
    // char fname[1024] = "N125/phig598/equHSrho598N125_s1000.bin";
    particle_->nAtom = N;
    // particle->pos = (doubleVector *)calloc(particle->nAtom, sizeof(doubleVector));
    // xsize = ysize = zsize = 1.0;
    // initcelllist_without_confs();

    readConf_data(box, particle_, fname);
    double scale_sum = 0.;
    double sum_ball_volume = 0.;
    double radius = 0.0, radiusbar = 0.;
    sparticle *p;

    for (int i = 0; i < particle_->nAtom; i++) {
        radius = particle_->diameterScale[i]/2.0 * particle_->meanDiameter;
        radiusbar += 2.*radius/N;   //actually it is the diameter
        sum_ball_volume += radius*radius*radius*4/3*M_PI;
    }

    xsize = ysize = zsize = 1.0/radiusbar;
    printf("xsize=%lf  diamater=%lf",xsize,radiusbar);
    printf("\n");
    initcelllist_without_confs();

    for (int i = 0; i < particle_->nAtom; i++) {
        p = &(particles[i]);
        p->r = particle_->meanDiameter*particle_->diameterScale[i]/2./radiusbar;
        p->x = (particle_->pos[i][0])/radiusbar + xsize/2;
        p->y = (particle_->pos[i][1])/radiusbar + ysize/2;
        p->z = (particle_->pos[i][2])/radiusbar + zsize/2;
        p->rtarget = cbrt(targetpackfrac/sum_ball_volume)*p->r;//??
        p->mass = 1.;
        p->vr = growthspeed * p->r;
        double mt = (p->rtarget - p->r) / p->vr;
        target_time = mt;

        if(i==0){printf("mt=%lf\n",mt);}
        if (mt < 0) 
            // printf("Particles should not exceed size 1!\n");
        if (maxt < 0 || maxt > mt) {maxt = mt;printf("mt=%lf\n",mt); }

        p->cellx = p->x / cxsize;				//Find particle's cell
        p->celly = p->y / cysize;
        p->cellz = p->z / czsize;

        double sqm = 1.0 / sqrt(p->mass);
        p->xn = p->x; p->yn = p->y; p->zn = p->z;
        p->vx = (double_01(rng_64) - 0.5) / sqm;
        p->vy = (double_01(rng_64) - 0.5) / sqm;
        p->vz = (double_01(rng_64) - 0.5) / sqm;
        p->t = 0;						//r and v known at t=0
        p->next = celllist[p->cellx*cy*cz + p->celly*cz + p->cellz];	//Add particle to celllist
        if (p->next) p->next->prev = p;			//Link up list
        celllist[p->cellx*cy*cz + p->celly*cz + p->cellz] = p;
        p->prev = nullptr;

        // printf("%le %le %le  %le\n",
        //     particle_->pos[i][0], particle_->pos[i][1], particle_->pos[i][2], 
        //     particle_->diameterScale[i]
        //     );
    }
    

    printf("initial density = %lf\n",sum_ball_volume);
    // printf("mean diameter = %lf\n",particle_->meanDiameter);
    // printf("mean diameter = %lf\n",particle_->meanDiameter);
    // printf("%lf\n",particle_->nAtomType);
    // printf("scale_sum = %lf\n",scale_sum);
    free(box);
    free(particle_->pos);
    free(particle_->veloc);
    free(particle_->force);
    free(particle_->mass);
    free(particle_->img);
    free(particle_->type);
    free(particle_->diameterScale);
    free(particle_->id2tag);
    free(particle_->tag2id);
    free(particle_);
    return 0;
}
template <typename T>
int MDGrow::copy_confs(T *particles_){
    /* 
     THE FILE FORMAT IS BINARY
     In conf data, particle position is -0.5->0.5 
     */
    // xsize = ysize = zsize = 1.0;
    // initcelllist_without_confs();

    double scale_sum = 0.;
    double sum_ball_volume = 0.;
    double radius = 0.0, radiusbar = 0.;
    sparticle *p;

    for (int i = 0; i < N; i++) {
        radius = particles_[i].r;
        radiusbar += radius/N;
        sum_ball_volume += radius*radius*radius*4/3*M_PI;
    }

    xsize = ysize = zsize = 9.913121729529303;
    // = cbrt(sum_ball_volume/targetpackfrac);
    // =9.913121729529303;
    initcelllist_without_confs();

    for (int i = 0; i < N; i++) {
        p = &(particles[i]);
        p->r = particles_[i].r;
        p->x = particles_[i].x;
        p->y = particles_[i].y;
        p->z = particles_[i].z;
        if (p->x >= xsize) { p->x -= xsize; }
        else if(p->x < 0) { p->x += xsize; }
        if (p->y >= ysize) { p->y -= ysize; }
        else if(p->y < 0) { p->y += ysize;  }
        if (p->z >= zsize) { p->z -= zsize;  }
        else if(p->z < 0) { p->z += zsize;  }

        p->rtarget = cbrt(targetpackfrac/sum_ball_volume)*p->r;//??
        p->mass = 1.;
        p->vr = growthspeed * p->r;
        double mt = (p->rtarget - p->r) / p->vr;
        target_time = mt;

        if(i==0){printf("mt=%lf\n",mt);}
        if (mt < 0) 
            // printf("Particles should not exceed size 1!\n");
        if (maxt < 0 || maxt > mt) {maxt = mt; }

        p->cellx = p->x / cxsize;				//Find particle's cell
        p->celly = p->y / cysize;
        p->cellz = p->z / czsize;

        double sqm = 1.0 / sqrt(p->mass);
        p->xn = p->x; p->yn = p->y; p->zn = p->z;
        p->vx = (double_01(rng_64) - 0.5) / sqm;
        p->vy = (double_01(rng_64) - 0.5) / sqm;
        p->vz = (double_01(rng_64) - 0.5) / sqm;
        p->t = 0;						//r and v known at t=0
        p->next = celllist[p->cellx*cy*cz + p->celly*cz + p->cellz];	//Add particle to celllist
        if (p->next) p->next->prev = p;			//Link up list
        celllist[p->cellx*cy*cz + p->celly*cz + p->cellz] = p;
        p->prev = nullptr;

        // printf("%le %le %le  %le\n",
        //     particle_->pos[i][0], particle_->pos[i][1], particle_->pos[i][2], 
        //     particle_->diameterScale[i]
        //     );
    }
    

    printf("initial density = %lf\n",sum_ball_volume);
    // printf("mean diameter = %lf\n",particle_->meanDiameter);
    // printf("mean diameter = %lf\n",particle_->meanDiameter);
    // printf("%lf\n",particle_->nAtomType);
    // printf("scale_sum = %lf\n",scale_sum);

    return 0;
}
template int MDGrow::copy_confs<sparticle>(sparticle* particles);
template int MDGrow::copy_confs<pos_t>(pos_t* particles);

Box_t MDGrow::get_box(){
    return Box_t({xsize, ysize, zsize});
}

int MDGrow::without_confs(){
    /* 
     THE FILE FORMAT IS BINARY
     In conf data, particle position is -0.5->0.5 
     */

    double scale_sum = 0.;
    double sum_ball_volume = 0.;
    double radius = 0.0;
    sparticle *p;

    for (int i = 0; i < N; i++) {
        radius = particles[i].r;
        sum_ball_volume += radius*radius*radius*4/3*M_PI;
    }

    for (int i = 0; i < N; i++) {
        p = &(particles[i]);
        if (p->x >= xsize) { p->x -= xsize; }
        else if(p->x < 0) { p->x += xsize; }
        if (p->y >= ysize) { p->y -= ysize; }
        else if(p->y < 0) { p->y += ysize;  }
        if (p->z >= zsize) { p->z -= zsize;  }
        else if(p->z < 0) { p->z += zsize;  }
        
        p->rtarget = cbrt(targetpackfrac/sum_ball_volume)*p->r;//??
        p->mass = 1.;
        p->type = 0;
        p->vr = growthspeed * p->r;
        double mt = (p->rtarget - p->r) / p->vr;
        target_time = mt;

        if(i==0){printf("mt=%lf\n",mt);}
        if (mt < 0) printf("Particles should not exceed size 1!\n");
        if (maxt < 0 || maxt > mt) {maxt = mt; }

        p->cellx = p->x / cxsize;				//Find particle's cell
        p->celly = p->y / cysize;
        p->cellz = p->z / czsize;

        double sqm = 1.0 / sqrt(p->mass);
        p->xn = p->x; p->yn = p->y; p->zn = p->z;
        p->vx = (double_01(rng_64) - 0.5) / sqm;
        p->vy = (double_01(rng_64) - 0.5) / sqm;
        p->vz = (double_01(rng_64) - 0.5) / sqm;
        p->t = 0;						//r and v known at t=0
        p->next = celllist[p->cellx*cy*cz + p->celly*cz + p->cellz];	//Add particle to celllist
        if (p->next) p->next->prev = p;			//Link up list
        celllist[p->cellx*cy*cz + p->celly*cz + p->cellz] = p;
        p->prev = nullptr;

        // printf("%le %le %le  %le\n",
        //     particle_->pos[i][0], particle_->pos[i][1], particle_->pos[i][2], 
        //     particle_->diameterScale[i]
        //     );
    }
    

    printf("initial density = %lf\n",sum_ball_volume);
    // printf("mean diameter = %lf\n",particle_->meanDiameter);
    // printf("mean diameter = %lf\n",particle_->meanDiameter);
    // printf("%lf\n",particle_->nAtomType);
    // printf("scale_sum = %lf\n",scale_sum);

    return 0;
}


/**************************************************
**                 PRINTSTUFF
** Some data at the end of the simulation
**************************************************/
void MDGrow::printstuff()
{
    int i;
    particle* p;
    double v2tot = 0;
    double vfilled = 0;

    for (i = 0; i < N; i++)
    {
        p = particles + i;
        v2tot += p->mass * (p->vx * p->vx + p->vy * p->vy + p->vz * p->vz);
        vfilled += p->r * p->r * p->r * 8;
    }
    vfilled *= M_PI / 6.0;
    printf("Average kinetic energy: %lf\n", 0.5 * v2tot / N);
    double volume = xsize * ysize * zsize;
    double dens = N / volume;
    double press = -dvtot / (3.0 * volume * time);
    double pressid = dens;
    double presstot = press + pressid;
    printf("Total time simulated  : %lf\n", time);
    //  printf ("Density               : %lf\n", (double) N / volume);
    printf("Packing fraction      : %lf\n", vfilled / volume);
    printf("Measured pressure     : %lf + %lf = %lf\n", press, pressid, presstot);

}


/**************************************************
**                    INIT
**************************************************/
void MDGrow::init()
{
    int i;
    unsigned long seed = 1;
    //   FILE *fp=fopen("/dev/urandom","r");
    //   int tmp = fread(&seed,1,sizeof(unsigned long),fp);
    //   if (tmp != sizeof(unsigned long)) printf ("error with seed\n");
    //   fclose(fp);
    printf("Seed: %u\n", (int)seed);
    // init_genrand(seed);
    initeventpool();

    for (i = 0; i < N; i++)
    {
        particle* p = particles + i;
        p->number = i;
        p->boxestraveledx = 0;
        p->boxestraveledy = 0;
        p->boxestraveledz = 0;
        p->nneigh = 0;
        p->counter = 0;
        p->t = 0;
    }


    randomparticles();
    randommovement();
    hx = 0.5 * xsize; hy = 0.5 * ysize; hz = 0.5 * zsize;	//Values used for periodic boundary conditions



    for (i = 0; i < N; i++)
    {
        particle* p = particles + i;
        p->xn = p->x;
        p->yn = p->y;
        p->zn = p->z;
    }

    for (i = 0; i < N; i++)
    {
        makeneighborlist(particles + i, 1);
    }
    printf("Done adding collisions: %d events\n", MAXEVENTS - nempty);


    if (usethermostat)
    {
        thermostat(nullptr);
        printf("Started thermostat: %d events\n", MAXEVENTS - nempty);
    }

}

void MDGrow::init_read_confs(char *fname)
{
    int i;
    unsigned long seed = 1;
    //   FILE *fp=fopen("/dev/urandom","r");
    //   int tmp = fread(&seed,1,sizeof(unsigned long),fp);
    //   if (tmp != sizeof(unsigned long)) printf ("error with seed\n");
    //   fclose(fp);
    printf("Seed: %u\n", (int)seed);
    // init_genrand((unsigned long)(&seed));
    initeventpool();

    for (i = 0; i < N; i++)
    {
        particle* p = particles + i;
        p->number = i;
        p->boxestraveledx = 0;
        p->boxestraveledy = 0;
        p->boxestraveledz = 0;
        p->nneigh = 0;
        p->counter = 0;
        p->t = 0;
    }



    read_confs(fname);
    printf("!\n");
    fflush(stdout);
    randommovement();
    hx = 0.5 * xsize; hy = 0.5 * ysize; hz = 0.5 * zsize;	//Values used for periodic boundary conditions


    printf("!!\n");
    fflush(stdout);
    for (i = 0; i < N; i++)
    {
        particle* p = particles + i;
        p->xn = p->x;
        p->yn = p->y;
        p->zn = p->z;
    }
    printf("!!!\n");
    fflush(stdout);
    for (i = 0; i < N; i++)
    {
        makeneighborlist(particles + i, 1);
    }
    printf("Done adding collisions: %d events\n", MAXEVENTS - nempty);
    printf("!!!!\n");
    fflush(stdout);

    if (usethermostat)
    {
        thermostat(nullptr);
        printf("Started thermostat: %d events\n", MAXEVENTS - nempty);
    }
    printf("!!!!!\n");
    fflush(stdout);
}

void MDGrow::init_without_confs()
{
    int i;
    initeventpool();

    for (i = 0; i < N; i++)
    {
        particle* p = particles + i;
        p->number = i;
        p->boxestraveledx = 0;
        p->boxestraveledy = 0;
        p->boxestraveledz = 0;
        p->nneigh = 0;
        p->counter = 0;
        p->t = 0;
    }


    xsize = ysize = zsize = 1.0;
    initcelllist_without_confs();//assign cellist
    
    without_confs();
    randommovement();
    hx = 0.5 * xsize; hy = 0.5 * ysize; hz = 0.5 * zsize;	//Values used for periodic boundary conditions


    for (i = 0; i < N; i++)
    {
        particle* p = particles + i;
        p->xn = p->x;
        p->yn = p->y;
        p->zn = p->z;
    }
    fflush(stdout);
    for (i = 0; i < N; i++)
    {
        makeneighborlist(particles + i, 1);
    }
    // printf("Done adding collisions: %d events\n", MAXEVENTS - nempty);

    if (usethermostat)
    {
        thermostat(nullptr);
        printf("Started thermostat: %d events\n", MAXEVENTS - nempty);
    }
}
template <typename T>
void MDGrow::init_copy_confs(T *particles_)
{
    int i;
    unsigned long seed = 1;
    //   FILE *fp=fopen("/dev/urandom","r");
    //   int tmp = fread(&seed,1,sizeof(unsigned long),fp);
    //   if (tmp != sizeof(unsigned long)) printf ("error with seed\n");
    //   fclose(fp);
    printf("Seed: %u\n", (int)seed);
    // init_genrand((unsigned long)(&seed));
    initeventpool();

    for (i = 0; i < N; i++)
    {
        particle* p = particles + i;
        p->number = i;
        p->boxestraveledx = 0;
        p->boxestraveledy = 0;
        p->boxestraveledz = 0;
        p->nneigh = 0;
        p->counter = 0;
        p->t = 0;
    }



    copy_confs(particles_);
    printf("!\n");
    fflush(stdout);
    randommovement();
    hx = 0.5 * xsize; hy = 0.5 * ysize; hz = 0.5 * zsize;	//Values used for periodic boundary conditions


    printf("!!\n");
    fflush(stdout);
    for (i = 0; i < N; i++)
    {
        particle* p = particles + i;
        p->xn = p->x;
        p->yn = p->y;
        p->zn = p->z;
    }
    printf("!!!\n");
    fflush(stdout);
    for (i = 0; i < N; i++)
    {
        makeneighborlist(particles + i, 1);
    }
    printf("Done adding collisions: %d events\n", MAXEVENTS - nempty);
    printf("!!!!\n");
    fflush(stdout);

    if (usethermostat)
    {
        thermostat(nullptr);
        printf("Started thermostat: %d events\n", MAXEVENTS - nempty);
    }
    printf("!!!!!\n");
    fflush(stdout);
}
template void MDGrow::init_copy_confs<sparticle>(sparticle* particles);
template void MDGrow::init_copy_confs<pos_t>(pos_t* pos);
/******************************************************
**               MYGETLINE
** Reads a single line, skipping over lines
** commented out with #
******************************************************/
int MDGrow::mygetline(char* str, FILE* f)
{
    int comment = 1;
    while (comment)
    {
        if (!fgets(str, 255, f)) return -1;
        if (str[0] != '#') comment = 0;
    }
    return 0;
}




/**************************************************
**                    RANDOMPARTICLES
** Positions particles randomly in the box
**************************************************/
void MDGrow::randomparticles()
{
    double x = composition;
    double alpha = sizeratio;
    double vol = N * M_PI / 6 * (x + (1 - x) * alpha * alpha * alpha) / targetpackfrac;

    printf("Volume: %lf\n", vol);

    xsize = cbrt(vol);
    ysize = xsize;
    zsize = ysize;
    initcelllist();
    int i;
    particle* p;
    for (i = 0; i < N; i++)				//First put particles at zero
    {
        particles[i].x = 0; particles[i].y = 0; particles[i].z = 0;
    }
    for (i = 0; i < N; i++)
    {
        p = &(particles[i]);
        p->rtarget = 1;
        if (i >= x * N - 0.00000001) p->rtarget = alpha;
        p->rtarget *= 0.5;
        p->r = 0.5 * p->rtarget;
        p->mass = 1;
        p->type = 0;
        if (p->rtarget < 0.5) p->type = 1;
        p->number = i;
        p->vr = growthspeed * p->r;
        double mt = (p->rtarget - p->r) / p->vr;
        if (mt < 0) printf("Particles should not exceed size 1!\n");
        if (maxt < 0 || maxt > mt) maxt = mt;
        do
        {
            p->x = double_01(rng_64) * xsize;			//Random location and speed
            p->y = double_01(rng_64) * ysize;
            p->z = double_01(rng_64) * zsize;
            p->cellx = p->x / cxsize;				//Find particle's cell
            p->celly = p->y / cysize;
            p->cellz = p->z / czsize;
        } while (overlaplist(p, 0));
        double sqm = 1.0 / sqrt(p->mass);
        p->xn = p->x; p->yn = p->y; p->zn = p->z;
        p->vx = (double_01(rng_64) - 0.5) / sqm;
        p->vy = (double_01(rng_64) - 0.5) / sqm;
        p->vz = (double_01(rng_64) - 0.5) / sqm;
        p->t = 0;						//r and v known at t=0
        p->next = celllist[p->cellx*cy*cz + p->celly*cz + p->cellz];	//Add particle to celllist
        if (p->next) p->next->prev = p;			//Link up list
        celllist[p->cellx*cy*cz + p->celly*cz + p->cellz] = p;
        p->prev = nullptr;
    }
}


/**************************************************
**                RANDOMMOVEMENT
**************************************************/
void MDGrow::randommovement()
{
    particle* p;
    double v2tot = 0, vxtot = 0, vytot = 0, vztot = 0;
    double mtot = 0;
    int i;

    for (i = 0; i < N; i++)
    {
        p = particles + i;
        double imsq = 1.0 / sqrt(p->mass);

        p->vx = imsq * normal(rng_64);
        p->vy = imsq * normal(rng_64);
        p->vz = imsq * normal(rng_64);
        vxtot += p->mass * p->vx;					//Keep track of total v
        vytot += p->mass * p->vy;
        vztot += p->mass * p->vz;
        mtot += p->mass;
    }


    vxtot /= mtot; vytot /= mtot; vztot /= mtot;
    for (i = 0; i < N; i++)
    {
        p = &(particles[i]);
        p->vx -= vxtot;					//Make sure v_cm = 0
        p->vy -= vytot;
        p->vz -= vztot;
        v2tot += p->mass * (p->vx * p->vx + p->vy * p->vy + p->vz * p->vz);
    }
    double fac = sqrt(3.0 / (v2tot / N));
    v2tot = 0;
    vxtot = vytot = vztot = 0;
    for (i = 0; i < N; i++)
    {
        p = &(particles[i]);
        p->vx *= fac;					//Fix energy
        p->vy *= fac;
        p->vz *= fac;
        v2tot += p->mass * (p->vx * p->vx + p->vy * p->vy + p->vz * p->vz);
        vxtot += p->mass * p->vx;					//Keep track of total v
        vytot += p->mass * p->vy;
        vztot += p->mass * p->vz;

    }
    printf("average v2: %lf (%lf, %lf, %lf)\n", v2tot / N, vxtot / N, vytot / N, vztot / N);
}

/**************************************************
**                UPDATE
**************************************************/
void MDGrow::update(particle* p1)
{
    double dt = time - p1->t;
    p1->t = time;
    p1->x += dt * p1->vx;
    p1->y += dt * p1->vy;
    p1->z += dt * p1->vz;
    p1->r += dt * p1->vr;

}


/**************************************************
**                 INITCELLLIST
**************************************************/
void MDGrow::initcelllist()
{
    int i, j, k;
    cx = (int)(xsize - 0.0001) / shellsize;				//Set number of cells
    cy = (int)(ysize - 0.0001) / shellsize;
    cz = (int)(zsize - 0.0001) / shellsize;
    printf("Cells: %d, %d, %d\n", cx, cy, cz);
    if (cx >= CEL || cy >= CEL || cz >= CEL)
    {
        printf("Too many cells!\n");
        stop = 1; return;
    }
    celllist = (sparticle **)malloc(sizeof(sparticle *) * cx*cy*cz);

    cxsize = xsize / cx;						//Set cell size
    cysize = ysize / cy;
    czsize = zsize / cz;
    for (i = 0; i < cx; i++)					//Clear celllist
        for (j = 0; j < cy; j++)
            for (k = 0; k < cz; k++)
            {
                celllist[i*cz*cy + j*cz + k] = nullptr;
            }

}

void MDGrow::initcelllist_without_confs()
{
    int i, j, k;
    double x = composition;
    double alpha = sizeratio;
    cx = cy = cz = (int)(cbrt(N * M_PI / 6 * (x + (1 - x) * alpha * alpha * alpha) / targetpackfrac) / shellsize);
        // cx = cy = cz = 3;	
    if(cx<CEM) cx=CEM;
    if(cy<CEM) cy=CEM;
    if(cz<CEM) cz=CEM;	
    celllist = (sparticle **)malloc(sizeof(sparticle *) * cx*cy*cz);

    printf("Cells: %d, %d, %d\n", cx, cy, cz);
    if (cx >= CEL || cy >= CEL || cz >= CEL)
    {
        printf("Too many cells!\n");
        stop = 1; return;
    }
    cxsize = xsize / cx;						//Set cell size
    cysize = ysize / cy;
    czsize = zsize / cz;
    for (i = 0; i < cx; i++)					//Clear celllist
        for (j = 0; j < cy; j++)
            for (k = 0; k < cz; k++)
            {
                celllist[i*cz*cy + j*cz + k] = nullptr;
            }

}




/**************************************************
**               REMOVEFROMCELLLIST
**************************************************/
void MDGrow::removefromcelllist(particle* p1)
{
    if (p1->prev) p1->prev->next = p1->next;    //Remove particle from celllist
    else celllist[p1->cellx*cy*cz + p1->celly*cz + p1->cellz] = p1->next;
    if (p1->next) p1->next->prev = p1->prev;
}

/**************************************************
**                    ADDTOCELLLIST
**************************************************/
void MDGrow::addtocelllist(particle* p)
{
    p->cellx = p->x / cxsize;				//Find particle's cell
    p->celly = p->y / cysize;
    p->cellz = p->z / czsize;
    p->next = celllist[p->cellx*cy*cz + p->celly*cz + p->cellz];	//Add particle to celllist
    if (p->next) p->next->prev = p;			//Link up list
    celllist[p->cellx*cy*cz + p->celly*cz + p->cellz] = p;
    p->prev = nullptr;
    p->edge = (p->cellx == 0 || p->celly == 0 || p->cellz == 0 || p->cellx == cx - 1 || p->celly == cy - 1 || p->cellz == cz - 1);

}

/**************************************************
**                     STEP
**************************************************/
void MDGrow::step()
{
    event* ev;
    ev = root->child2;
    if (ev == nullptr)
    {
        addnexteventlist();
        ev = root->child2;
    }

    while (ev->child1) ev = ev->child1;		//Find first event

    // printf("type = %d\n",ev->type);

    //if (ev->type == 0)
    //{
      //printf("Time: %lf, ev: %d, part: %d, %d\n", ev->time, ev->type, ev->p1 ? ev->p1->number : -1, ev->p2 ? ev->p2->number : -1);
    //}

   //if ((ev->p1 && ev->p1->number == 591) && (ev->p2 && ev->p2->number == 701) || 
   //     (ev->p1 && ev->p1->number == -1) || (ev->p2 && ev->p2->number == -1))
   // {
   //      printf ("Time: %.10lf, ev: %d, part: %d, %d (%lf, %d)\n", ev->time, ev->type, ev->p1?ev->p1->number:-1, ev->p2?ev->p2->number:-1, particles[621].z, particles[374].cellz);
   // }

   // if (ev->time < time)
   // {
   //     printf("Time: %lf, ev: %d, part: %d, %d\n", ev->time, ev->type, ev->p1 ? ev->p1->number : -1, ev->p2 ? ev->p2->number : -1);
   //     exit(3);
   // }

   // if (ev->time < time)
   // {
   //     printf(" time error: %lf, %lf \n", time, ev->time);
   //     printf("Time: %.10lf, ev: %d, part: %d, %d (%lf, %d)\n", ev->time, ev->type, ev->p1 ? ev->p1->number : -1, ev->p2 ? ev->p2->number : -1, particles[621].z, particles[374].cellz);

   //     exit(3);
   // }
//    ev->time> time  ev->time  is the next collision time, time is the now.
    // printf("time = %lf  %lf  dt=%le\n",  time, ev->time,-time+ev->time);
    if (ev->time > maxt)
    {
        time = maxt;
        // write();
        // writelast();
        // printf("Time is up!\n");
        stop = 1;
    }

    if (ev->type == 100) //??
    {
        time = ev->time;
        removeevent(ev);
        // write();
    }
    else if(ev->type == 200  &&  usethermostat)//??
    {
        thermostat(ev);
    }
    else if(ev->type == 200  &&  usethermostat==0)//??
    {
        removeevent(ev);
    }
    else if(ev->type == 8)//??
    {
        time = ev->time;
        removeevent(ev);
        makeneighborlist(ev->p1, 0);
    }
    else
    {
        if(colcounter<20 && colcounter>0){
            // int totvx = 0., totvy = 0., totvz = 0.;
            // for(int i = 0; i < N; i++){
            //     if(particles2[i].vx != particles[i].vx || particles2[i].vy != particles[i].vy || particles2[i].vz != particles[i].vz){
            //         printf("Particle %d: %le %le %le\n",i,particles2[i].vx,particles2[i].vy,particles2[i].vz);
            //         printf("Particle %d: %le %le %le\n\n",i,particles[i].vx,particles[i].vy,particles[i].vz);
            //     }
            // }
            // for(int i = 0; i < N; i++){
            //     totvx += particles[i].mass * particles[i].vx;
            //     totvy += particles[i].mass * particles[i].vy;
            //     totvz += particles[i].mass * particles[i].vz;  
            //     particles2[i] = particles[i];
            // }

        // printf("%d\t\t%le %le %le\n",colcounter,totvx,totvy,totvz);
        }
        collision(ev);
    }
}


void MDGrow::step_specific_time()
{
    event* ev;
    ev = root->child2;
    if (ev == nullptr)
    {
        addnexteventlist();
        ev = root->child2;
    }

    while (ev->child1) ev = ev->child1;		//Find first event


    //if (ev->type == 0)
    //{
      //printf("Time: %lf, ev: %d, part: %d, %d\n", ev->time, ev->type, ev->p1 ? ev->p1->number : -1, ev->p2 ? ev->p2->number : -1);
    //}

   //if ((ev->p1 && ev->p1->number == 591) && (ev->p2 && ev->p2->number == 701) || 
   //     (ev->p1 && ev->p1->number == -1) || (ev->p2 && ev->p2->number == -1))
   // {
   //      printf ("Time: %.10lf, ev: %d, part: %d, %d (%lf, %d)\n", ev->time, ev->type, ev->p1?ev->p1->number:-1, ev->p2?ev->p2->number:-1, particles[621].z, particles[374].cellz);
   // }

   // if (ev->time < time)
   // {
   //     printf("Time: %lf, ev: %d, part: %d, %d\n", ev->time, ev->type, ev->p1 ? ev->p1->number : -1, ev->p2 ? ev->p2->number : -1);
   //     exit(3);
   // }

   // if (ev->time < time)
   // {
   //     printf(" time error: %lf, %lf \n", time, ev->time);
   //     printf("Time: %.10lf, ev: %d, part: %d, %d (%lf, %d)\n", ev->time, ev->type, ev->p1 ? ev->p1->number : -1, ev->p2 ? ev->p2->number : -1, particles[621].z, particles[374].cellz);

   //     exit(3);
   // }
//    ev->time> time  ev->time  is the next collision time, time is the now.
    // printf("time = %lf  %lf  dt=%le\n",  time, ev->time,-time+ev->time);
    if (ev->time > maxt)
    {
        time = maxt;
        write();
        writelast();
        // printf("Time is up!\n");
        stop = 1;
    }

    if (ev->type == 100) //??
    {
        time = ev->time;
        removeevent(ev);
        write();
    }
    else if(ev->type == 200)//??
    {
        // thermostat(ev);
    }
    else if(ev->type == 8)//??
    {
        time = ev->time;
        removeevent(ev);
        makeneighborlist(ev->p1, 0);
    }
    else
    {

        collision(ev);
    }
}

/**************************************************
**                MAKENEIGHBORLIST
**************************************************/
void MDGrow::makeneighborlist(particle* p1, int firsttime)
{

    int cdx, cdy, cdz, cellx, celly, cellz;
    particle* p2;
    double dx, dy, dz, r2, rm;

    update(p1);

    if (p1->x >= xsize) { p1->x -= xsize; p1->boxestraveledx++; }
    else if(p1->x < 0) { p1->x += xsize; p1->boxestraveledx--; }
    if (p1->y >= ysize) { p1->y -= ysize; p1->boxestraveledy++; }
    else if(p1->y < 0) { p1->y += ysize; p1->boxestraveledy--; }
    if (p1->z >= zsize) { p1->z -= zsize; p1->boxestraveledz++; }
    else if(p1->z < 0) { p1->z += zsize; p1->boxestraveledz--; }
    p1->xn = p1->x;
    p1->yn = p1->y;
    p1->zn = p1->z;



    removefromcelllist(p1);
    addtocelllist(p1);


    int i, j;
    for (i = 0; i < p1->nneigh; i++)
    {
        p2 = p1->neighbors[i];
        for (j = 0; j < p2->nneigh; j++)
        {
            if (p2->neighbors[j] == p1)
            {
                p2->nneigh--;
                p2->neighbors[j] = p2->neighbors[p2->nneigh];
                break;
            }
        }
    }


    cellx = p1->cellx + cx;
    celly = p1->celly + cy;
    cellz = p1->cellz + cz;

    p1->nneigh = 0;

    for (cdx = cellx - 1; cdx < cellx + 2; cdx++)
        for (cdy = celly - 1; cdy < celly + 2; cdy++)
            for (cdz = cellz - 1; cdz < cellz + 2; cdz++)
            {
                p2 = celllist[(cdx % cx)*cy*cz + (cdy % cy)*cz + cdz % cz];
                while (p2)
                {
                    if (p2 != p1)
                    {
                        update(p2);
                        dx = p1->xn - p2->xn;
                        dy = p1->yn - p2->yn;
                        dz = p1->zn - p2->zn;
                        if (p1->edge)
                        {
                            if (dx > hx) dx -= xsize; else if(dx < -hx) dx += xsize;  //periodic boundaries
                            if (dy > hy) dy -= ysize; else if(dy < -hy) dy += ysize;
                            if (dz > hz) dz -= zsize; else if(dz < -hz) dz += zsize;
                        }
                        r2 = dx * dx + dy * dy + dz * dz;
                        rm = (p1->r + p1->vr*maxdt + p2->rtarget + p2->vr*maxdt) * shellsize;
                        if (r2 < rm * rm)
                        {
                            p1->neighbors[p1->nneigh++] = p2;
                            p2->neighbors[p2->nneigh++] = p1;
                        }
                    }
                    p2 = p2->next;
                }
            }

    //printf("Neighbors: %d, %d\n", p1->number, p1->nneigh);
    //if (p1->nneigh > MAXNEIGH) exit(3);

    findcollisions(p1);


}


/**************************************************
**                FINDNEIGHBORLISTUPDATE
** Assumes p1 is up to date
** Note that the particle is always in the same
** box as its neighborlist MDGrow::position(p->xn)
**************************************************/
double MDGrow::findneighborlistupdate(particle* p1)
{
    double dx = p1->x - p1->xn;
    double dy = p1->y - p1->yn;
    double dz = p1->z - p1->zn;

    double dvx = p1->vx, dvy = p1->vy, dvz = p1->vz;

    double b = dx * dvx + dy * dvy + dz * dvz;                  //dr.dv

    double dv2 = dvx * dvx + dvy * dvy + dvz * dvz;
    double dr2 = dx * dx + dy * dy + dz * dz;
    double md = (shellsize - 1) * p1->rtarget;

    double disc = b * b - dv2 * (dr2 - md * md);
    if(disc<0) return never;
    double t = (-b + sqrt(disc)) / dv2;
    //printf("Predicting nlistupdate %d, %lf\n", p1->number, t);
    if (t<0 || isnan(t) || isinf(t)){
        printf("%lf\n",t);
    }
    // printf("%lf\n",t);

    return t;
}

/**************************************************
**                FINDCOLLISION
** Detect the next collision for two particles
** Note that p1 is always up to date in
** findcollision
**************************************************/
double MDGrow::findcollision(particle* p1, particle* p2, double tmin)
{
    double dt2 = time - p2->t;
    double dx = p1->x - p2->x - dt2 * p2->vx;    //relative distance at current time
    double dy = p1->y - p2->y - dt2 * p2->vy;
    double dz = p1->z - p2->z - dt2 * p2->vz;
    if (dx > hx) dx -= xsize; else if(dx < -hx) dx += xsize;  //periodic boundaries
    if (dy > hy) dy -= ysize; else if(dy < -hy) dy += ysize;
    if (dz > hz) dz -= zsize; else if(dz < -hz) dz += zsize;
    double dvx = p1->vx - p2->vx;                               //relative velocity
    double dvy = p1->vy - p2->vy;
    double dvz = p1->vz - p2->vz;
    double dvr = p1->vr + p2->vr;
    double md = p1->r + p2->r + dt2 * p2->vr;


    double b = dx * dvx + dy * dvy + dz * dvz - dvr * md;     //dr.dv
    if (b > 0) return never;
    double dv2 = dvx * dvx + dvy * dvy + dvz * dvz;
    double dr2 = dx * dx + dy * dy + dz * dz;

    double twoa = dv2 - dvr * dvr;

    double disc = b * b - twoa * (dr2 - md * md);
    if (disc < 0) return never;
    double t = (-b - sqrt(disc)) / twoa;
    if (t < 0. && time > 0) 
        printf("overlap = %d\n",overlap(particles));
        // printf("t<0");
        // return 0.0;
    if (isinf(t) || isnan(t))
        printf("t is inf or  nan\n");
    return t;

}




/**************************************************
**                FINDCOLLISIONS
** Find all collisions for particle p1.
** The particle 'not' isn't checked.
**************************************************/
void MDGrow::findcollisions(particle* p1)    //All collisions of particle p1
{
    int i;
    double t;
    double tmin = findneighborlistupdate(p1);
    if (tmin > maxdt) tmin = maxdt;
    int type = 8;
    particle* partner = p1;
    particle* p2;
    for (i = 0; i < p1->nneigh; i++)
    {
        p2 = p1->neighbors[i];
        if (p1==p2){
            printf("Warning!p1 is in the neighbor list of p1!");
        }
        t = findcollision(p1, p2, tmin);
        if (t<0 && time>0){
            double Ek_sum=0.;
            for(int i=0;i<N;i++){
                Ek_sum +=  particles[i].vx * particles[i].vx + particles[i].vy * particles[i].vy + particles[i].vz * particles[i].vz;
            }
            printf("same p1p2. radius: %d %d %lf\n",p1->number,p2->number,Ek_sum);
        }
        if (t < tmin)
        {
            tmin = t;
            partner = p2;
            type = 0;
        }
    }
    if (!p1->nneigh) {}
        // printf("There is no neighbor for some particle!\n");

    // if (tmin<0 && time>0){
    //     for (int i=0;i<N;i++){
    //         fprintf(fout_bad,"%le %le %le  %le  %le %le %le\n",particles[i].x,particles[i].y,particles[i].z
    //         ,particles[i].r,particles[i].vx,particles[i].vy,particles[i].vz);
    //         fflush(fout_bad);

    //     }
    //     printf("Warning! tmin<0!");
    // }
    // if(tmin<0) tmin = 0;
    event* ev = createevent(tmin + time, p1, partner, type);
    p1->firstcollision = ev;
    ev->counter2 = partner->counter;

}



/**************************************************
**                FINDALLCOLLISION
** All collisions of all particle pairs
**************************************************/
void MDGrow::findallcollisions()       //All collisions of all particle pairs
{
    int i, j;


    for (i = 0; i < N; i++)
    {
        particle* p1 = particles + i;
        particle* partner = p1;
        double tmin = findneighborlistupdate(p1);
        if(isnan(tmin)){
            printf("tmin is nan\n");
        }
        int type = 8;
        for (j = 0; j < p1->nneigh; j++)
        {
            particle* p2 = p1->neighbors[j];
            if (p2 > p1)
            {
                double t = findcollision(p1, p2, tmin);
                if (t < tmin)
                {
                    tmin = t;
                    partner = p2;
                    type = 0;
                }
            }
        }
        if (partner)
        {
            event* ev = createevent(tmin, p1, partner, type);
            p1->firstcollision = ev;
            ev->counter2 = partner->counter;
        }

    }
}







/**************************************************
**                  COLLISION
** Process a single collision event
**************************************************/
void MDGrow::collision(event* ev)
{
    time = ev->time;
    particle* p1 = ev->p1;
    particle* p2 = ev->p2;
    update(p1);
    removeevent(ev);
    if (ev->counter2 != p2->counter)//??
    {
        findcollisions(p1);
        return;
    }

    //     if (ev->counter1 != p1->counter)
    //     {
    //         printf("Huh?\n");
    //     }
    update(p2);
    p1->counter++;
    p2->counter++;



    double m1 = p1->mass, r1 = p1->r;
    double m2 = p2->mass, r2 = p2->r;

    double r = r1 + r2;
    double rinv = 1.0 / r;
    double dx = (p1->x - p2->x);			//Normalized distance vector
    double dy = (p1->y - p2->y);
    double dz = (p1->z - p2->z);
    if (p1->edge)
    {
        if (dx > hx) dx -= xsize; else if(dx < -hx) dx += xsize;  //periodic boundaries
        if (dy > hy) dy -= ysize; else if(dy < -hy) dy += ysize;
        if (dz > hz) dz -= zsize; else if(dz < -hz) dz += zsize;
    }
    dx *= rinv;  dy *= rinv;  dz *= rinv;

    // if (colcounter % 1000 == 0 || colcounter < 10){
    //     double totvx = 0., totvy = 0., totvz = 0.;
    //     for(int i = 0; i < N; i++){
    //         totvx += particles[i].mass * particles[i].vx;
    //         totvy += particles[i].mass * particles[i].vy;
    //         totvz += particles[i].mass * particles[i].vz;  
    //     }
    //     fprintf(fout_bad,"%d\t\t%le %le %le\n",colcounter,totvx,totvy,totvz);
    //     fflush(fout_bad);
    
    // }

    double dvx = p1->vx - p2->vx;                               //relative velocity
    double dvy = p1->vy - p2->vy;
    double dvz = p1->vz - p2->vz;
    double dvr = p1->vr + p2->vr;

    double b = dx * dvx + dy * dvy + dz * dvz - dvr;                  //dr.dv
    b *= 1.0 / (m1 + m2);
    double dv1 = 2 * b * m2;
    double dv2 = 2 * b * m1;
    dvtot += 2 * m1 * m2 * b;

    p1->vx -= dv1 * dx;         //Change velocities after collision
    p1->vy -= dv1 * dy;         //delta v = (-) dx2.dv2
    p1->vz -= dv1 * dz;
    p2->vx += dv2 * dx;
    p2->vy += dv2 * dy;
    p2->vz += dv2 * dz;

    colcounter++;



    // if ((colcounter-1) % 1000 == 0 || (colcounter-1) < 10){
    //     double totvx = 0., totvy = 0., totvz = 0.;
    //     for(int i = 0; i < N; i++){
    //         totvx += particles[i].mass * particles[i].vx;
    //         totvy += particles[i].mass * particles[i].vy;
    //         totvz += particles[i].mass * particles[i].vz;  
    //     }
    //     fprintf(fout_bad,"%d\t\t%le %le %le\n\n",colcounter,totvx,totvy,totvz);
    //     fflush(fout_bad);
    // }

    if (p2->firstcollision && p2->firstcollision != ev)
    {
        removeevent(p2->firstcollision);
    }

    findcollisions(p1);
    findcollisions(p2);





}






/**************************************************
**                 INITEVENTPOOL
** Creates two first events, and sets up
**************************************************/
void MDGrow::initeventpool()
{
    numeventlists = ceil(maxscheduletime / eventlisttime);  //about 2N
    maxscheduletime = numeventlists * eventlisttime;
    if (numeventlists > MAXNUMEVENTLISTS)
    {
        printf("Number of event lists too large: increase MAXNUMEVENTLISTS to at least %d\n", numeventlists);
        exit(3);
    }
    printf("number of lists: %d\n", numeventlists);


    int i;
    event* e;
    for (i = 0; i < MAXEVENTS; i++)			//Clear eventpool
    {
        e = &(eventlist[MAXEVENTS - i - 1]);			//Fill in backwards, so the first few events are 'special'
        eventpool[i] = e;					//This includes root, the write events, in that order
        eventpool[i]->child1 = nullptr;			//...  Not really used for now, but it might be useful at some point
        eventpool[i]->child2 = nullptr;			//Clear children
        nempty++;						//All events empty so far
    }
    root = eventpool[--nempty];				//Create root event
    root->time = -99999999999.99;				//Root event is empty, but makes sure every other event has a parent
    root->type = 200;					//This makes sure we don't have to keep checking this when adding/removing events
    root->parent = nullptr;
    event* writeevent = eventpool[--nempty];		//Pick first unused event
    writeevent->time = 0;
    writeevent->type = 100;
    root->child2 = writeevent;
    writeevent->parent = root;
    printf("Event tree initialized: %d events\n", MAXEVENTS - nempty);
}

/**************************************************
**                  ADDEVENTTOTREE
**************************************************/
void MDGrow::addeventtotree(event* newevent)
{
    double time = newevent->time;
    event* loc = root;
    int busy = 1;
    while (busy)						//Find location to add event into MDGrow::tree(loc)
    {
        if (time < loc->time)				//Go left
        {
            if (loc->child1) loc = loc->child1;
            else
            {
                loc->child1 = newevent;
                busy = 0;
            }
        }
        else						//Go right
        {
            if (loc->child2) loc = loc->child2;
            else
            {
                loc->child2 = newevent;
                busy = 0;
            }
        }
    }
    newevent->parent = loc;

}

/**************************************************
**                  ADDEVENT
**************************************************/
void MDGrow::addevent(event* newevent)
{
    double dt = newevent->time - reftime;

    if (dt < eventlisttime) //Put it in the tree
    {
        newevent->queue = currentlist;
        addeventtotree(newevent);
    }
    else
    {
        int list_id = currentlist + dt / eventlisttime;
        if (list_id >= numeventlists)
        {
            list_id -= numeventlists;
            if (list_id > currentlist - 1) list_id = numeventlists; //Overflow
        }

        newevent->queue = list_id;

        newevent->nextq = eventlists[list_id]; //Add to linear list
        newevent->prevq = nullptr;
        if (newevent->nextq) 
            newevent->nextq->prevq = newevent;
        eventlists[list_id] = newevent;
    }
}
/**************************************************
**                  CREATEEVENT
**************************************************/
event* MDGrow::createevent(double time, particle* p1, particle* p2, int type)
{
    event* newevent = eventpool[--nempty];		//Pick first unused event
    newevent->time = time;
    newevent->p1 = p1;
    newevent->type = type;
    newevent->p2 = p2;
    addevent(newevent);
    return newevent;
}

/**************************************************
**                     ADDNEXTEVENTLIST
**************************************************/
void MDGrow::addnexteventlist()
{
    do
    {
        currentlist++;
        if (currentlist == numeventlists) currentlist = 0;
        reftime += eventlisttime;
    } while (eventlists[currentlist] == nullptr);

    //   printf("Currentlist is now %d (%lf)\n", currentlist, reftime);

    event* ev = eventlists[currentlist];
    event* nextev;
    while (ev)
    {
        nextev = ev->nextq;
        //         if (ev->type != 0 || ev->counter2 == ev->p2->counter) 
        addeventtotree(ev);
        ev = nextev;
        listcounter1++;
    }
    eventlists[currentlist] = nullptr;
    ev = eventlists[numeventlists];//Overflow queue
    eventlists[numeventlists] = nullptr;
    while (ev)
    {
        nextev = ev->nextq;
        addevent(ev);
        ev = nextev;
        listcounter2++;
    }
    mergecounter++;
}

/**************************************************
**                  REMOVEEVENT
**************************************************/
void MDGrow::removeevent(event* oldevent)
{

    //event* ev = oldevent;
    //if (ev->type != 8) printf("Removing event: %lf, ev: %d, part: %d, %d\n", ev->time, ev->type, ev->p1 ? ev->p1->number : -1, ev->p2 ? ev->p2->number : -1);

    if (oldevent->queue != currentlist)
    {
        if (oldevent->nextq) oldevent->nextq->prevq = oldevent->prevq;
        if (oldevent->prevq) oldevent->prevq->nextq = oldevent->nextq;
        else
        {
            eventlists[oldevent->queue] = oldevent->nextq;
        }
        eventpool[nempty++] = oldevent;     //Put the removed event back in the event pool.
        return;
    }

    event* parent = oldevent->parent;
    event* node;					//This node will be attached to parent in the end


    if (oldevent->child1 == nullptr)			//Only one child: easy to delete
    {
        node = oldevent->child2;			//Child2 is attached to parent
        if (node)
        {
            node->parent = parent;
            oldevent->child2 = nullptr;			//Clear child, so createevent doesn't have to do it
        }
    }
    else if(oldevent->child2 == nullptr)		//Only one child again
    {
        node = oldevent->child1;			//Child1 is attached to parent
        node->parent = parent;
        oldevent->child1 = nullptr;
    }
    else		  //Node to delete has 2 children
    {               //In this case: a) Find first node after MDGrow::oldevent(This node will have no child1)
                    //              b) Remove this node from the MDGrow::tree(Attach node->child2 to node->parent)
                    //              c) Put this node in place of MDGrow::oldevent(Oldevent's children are adopted by node)
        node = oldevent->child2;
        while (node->child1) node = node->child1;	//Find first node of right tree of descendants of oldevent
        event* pnode = node->parent;
        if (pnode != oldevent)			//node is not a child of oldevent
        {						//Both of oldevent's children should be connected to node
            pnode->child1 = node->child2;		//Remove node from right tree
            if (node->child2) node->child2->parent = pnode;
            oldevent->child1->parent = node;
            node->child1 = oldevent->child1;
            oldevent->child2->parent = node;
            node->child2 = oldevent->child2;
        }
        else					//This means node == oldevent->child2
        {						//Only child1 has to be attached to node
            oldevent->child1->parent = node;
            node->child1 = oldevent->child1;
        }
        node->parent = parent;
        oldevent->child1 = nullptr;
        oldevent->child2 = nullptr;
    }
    if (parent->child1 == oldevent) parent->child1 = node;
    else                            parent->child2 = node;
    eventpool[nempty++] = oldevent;     //Put the removed event back in the event pool.
}

/**************************************************
**                  SHOWTREE
** Gives a rough view of the event tree.
** Not so useful except for very small trees
**************************************************/
void MDGrow::showtree()
{
    shownode(root);
}

void MDGrow::shownode(event* ev)
{
    int c1 = 0, c2 = 0, p = 0;
    if (ev->child1) c1 = ev->child1->type;
    if (ev->child2) c2 = ev->child2->type;
    if (ev->parent) p = ev->parent->type;
    printf("%3d => %3d => %d (p: %d, %d)\n", p, ev->type, c1, ev->p1->number, ev->p2->number);
    printf("           => %d \n", c2);

    if (ev->child1) shownode(ev->child1);
    if (ev->child2) shownode(ev->child2);
}

/**************************************************
**                  CHECKTREE
** Checks the tree for possible errors.
**  1 ) node's parent doesn't point to node
**  1b) node->t > parent->t
**  1c) node->t < parent->t
**  2 ) A non-root node lacks a parent
**  3 ) node's child1 doesn't point to parent
**  3b) node->t < child1->t
**  4 ) node's child2 doesn't point to parent
**  4b) node->t > child2->t
** Also checks if all events are in the tree
**************************************************/
void MDGrow::checktree()
{
    return;
    int t = checknode(root);
    if (t != MAXEVENTS - nempty) printf("Error: %d, %d\n", t, MAXEVENTS - nempty);
}

int MDGrow::checknode(event* node)
{
    static int count = 0;
    if (node->parent)
    {
        if (node->parent->child1 != node && node->parent->child2 != node) printf("Error 1\n");
        if (node->parent->child1 == node && node->time > node->parent->time) printf("Error 1b\n");
        if (node->parent->child2 == node && node->time < node->parent->time) printf("Error 1c\n");
    }
    else
    {
        if (root != node) printf("Error 2\n");
        count = 0;
    }
    if (node->child1)
    {
        checknode(node->child1);
        if (node->child1->parent != node) printf("Error 3\n");
        if (node->child1->time > node->time) printf("Error 3b\n");
    }
    if (node->child2)
    {
        checknode(node->child2);
        if (node->child2->parent != node) printf("Error 4\n");
        if (node->child2->time < node->time) printf("Error 4b\n");
    }
    count++;
    return count;
}

/**************************************************
**                    OVERLAP
** Checks for overlaps
** Should write one that uses the cell list...
**************************************************/
int MDGrow::overlap(particle* part)
{
    particle* p;
    double dx, dy, dz, r2, rm;
    int i;
    double dl = pow(10.0, -10);
    // double dl = 0.0;

    for (i = 0; i < N; i++)
    {
        if (i == part->number) continue;
        p = &(particles[i]);
        dx = part->x - p->x;
        dy = part->y - p->y;
        dz = part->z - p->z;
        if (dx > 0.5 * xsize) dx -= xsize; else if(dx < -0.5 * xsize) dx += xsize;  //periodic boundaries
        if (dy > 0.5 * ysize) dy -= ysize; else if(dy < -0.5 * ysize) dy += ysize;
        if (dz > 0.5 * zsize) dz -= zsize; else if(dz < -0.5 * zsize) dz += zsize;
        r2 = dx * dx + dy * dy + dz * dz;
        rm = p->r + part->r;
        if (r2 < rm * rm - dl)
        {
            printf("Overlap: %lf, %d, %d\n", r2 - rm * rm, part->number, p->number);
            return 1;
        }
    }
    return 0;
}

/**************************************************
**                    OVERLAPLIST
** Checks for overlaps
** if error is one, allow a small margin of error
**
**************************************************/
int MDGrow::overlaplist(particle* part, int error)
{
    int cdx, cdy, cdz, cellx, celly, cellz, num;
    particle* p;
    double dx, dy, dz, r2, rm;
    double dl = error * pow(10.0, -10);

    cellx = part->cellx + cx;
    celly = part->celly + cy;
    cellz = part->cellz + cz;
    num = part->number;

    for (cdx = cellx - 1; cdx < cellx + 2; cdx++)
        for (cdy = celly - 1; cdy < celly + 2; cdy++)
            for (cdz = cellz - 1; cdz < cellz + 2; cdz++)
            {
                p = celllist[(cdx % cx)*cy*cz + (cdy % cy)*cz + cdz % cz];
                while (p)
                {
                    if (p->number != num)
                    {
                        dx = part->x - p->x;
                        dy = part->y - p->y;
                        dz = part->z - p->z;
                        if (dx > 0.5 * xsize) dx -= xsize; else if(dx < -0.5 * xsize) dx += xsize;  //periodic boundaries
                        if (dy > 0.5 * ysize) dy -= ysize; else if(dy < -0.5 * ysize) dy += ysize;
                        if (dz > 0.5 * zsize) dz -= zsize; else if(dz < -0.5 * zsize) dz += zsize;
                        r2 = dx * dx + dy * dy + dz * dz;
                        rm = p->r + part->r;
                        if (r2 < rm * rm - dl)
                        {
                            //            printf ("Overlap: %lf, %d, %d\n", r2, part->number, p->number);
                            return 1;
                        }
                    }
                    p = p->next;
                }
            }
    return 0;
}




/**************************************************
**                    WRITE
** Writes a movie
**************************************************/
void MDGrow::write()
{
    static int counter = 0;
    static int first = 1;
    static double lastsnapshottime = -999999999.9;
    int i;
    particle* p;
    FILE* file;

    double en = 0;
    for (i = 0; i < N; i++)
    {
        p = particles + i;
        update(p);
        en += p->mass * (p->vx * p->vx + p->vy * p->vy + p->vz * p->vz);
    }
    temperature = 0.5 * en / (float)N / 1.5;


    //     checktree();
    //     checkcells();
    double volume = xsize * ysize * zsize;
    double dens = N / volume;
    double timeint = time;
    if (timeint < 0) timeint = time;
    double press = -dvtot / (3.0 * volume * timeint);
    double pressid = dens;
    double presstot = colcounter ? press + pressid : 0;

    double listsize1 = (double)listcounter1 / mergecounter;
    double listsize2 = (double)listcounter2 / mergecounter;
    if (mergecounter == 0) listsize1 = listsize2 = 0.0;
    listcounter1 = listcounter2 = mergecounter = 0;

    // printf("Simtime: %lf, Collisions: %u, Press: %lf (%lf), T: %lf, Listsizes: (%lf, %lf)\n", time, colcounter, presstot, presstot / dens, temperature, listsize1, listsize2);
    char filename[200];
    if (makesnapshots && time - lastsnapshottime > snapshotinterval - 0.001)
    {
        sprintf(filename, "mov.n%d.v%.4lf.sph", N, xsize * ysize * zsize);
        if (first) { first = 0; file = fopen(filename, "w"); }
        else                     file = fopen(filename, "a");
        fprintf(file, "%d\n%.12lf %.12lf %.12lf\n", (int)N, xsize, ysize, zsize);
        for (i = 0; i < N; i++)
        {
            p = &(particles[i]);

            fprintf(file, "%c %.12lf  %.12lf  %.12lf  %lf\n", 'a' + p->type, p->x + xsize * p->boxestraveledx, p->y + ysize * p->boxestraveledy, p->z + zsize * p->boxestraveledz, p->r);
        }
        fclose(file);
        lastsnapshottime = time;
    }
    if (temperature > 1.5) thermostatinterval *= 0.5;

    counter++;

    createevent(time + writeinterval, nullptr, nullptr, 100);     //Add next write interval
}


/**************************************************
**                    CHECKCELLS
** Checks the cell list for possible errors
**************************************************/
void MDGrow::checkcells()
{
    int x, y, z, count = 0;
    double dl = 0.0000001;
    particle* p;
    for (x = 0; x < cx; x++)
        for (y = 0; y < cy; y++)
            for (z = 0; z < cz; z++)
            {
                p = celllist[x*cy*cz + y*cz + z];
                if (p)
                {
                    if (p->prev) printf("First part has a MDGrow::prev(%d, %d, %d) %d, %d\n",
                        x, y, z, p->number, p->prev->number);
                    while (p)
                    {
                        count++;
                        if (p->cellx != x || p->celly != y || p->cellz != z)
                        {
                            printf("Cell error: %d, %d, %d / %d, %d, %d\n", x, y, z, p->cellx, p->celly, p->cellz);
                            exit(3);
                        }
                        if (p->x < cxsize * x - dl || p->x > cxsize * (x + 1) + dl)
                        {
                            printf("wrong cell x: %lf, %d, %d, %lf\n", p->x, x, p->number, p->vx);
                            exit(3);
                        }
                        if (p->y < cysize * y - dl || p->y > cysize * (y + 1) + dl)
                        {
                            printf("wrong cell y: %lf, %d, %d, %lf\n", p->y, y, p->number, p->vy);
                            exit(3);
                        }
                        if (p->z < czsize * z - dl || p->z > czsize * (z + 1) + dl)
                        {
                            printf("wrong cell z: %lf, %d, %d, %lf\n", p->z, z, p->number, p->vz);
                            exit(3);
                        }
                        if (p->next)
                        {
                            if (p->next->prev != p) printf("link error: %d, %d, %d\n",
                                p->number, p->next->number, p->next->prev->number);
                        }
                        p = p->next;
                    }
                }
            }
    if (count != N) printf("error in number of MDGrow::particles(%d)\n", count);
}

/**************************************************
**                    BACKINBOX
** Just for initialization
**************************************************/
void MDGrow::backinbox(particle* p)
{
    p->x -= xsize * floor(p->x / xsize);
    p->y -= ysize * floor(p->y / ysize);
    p->z -= zsize * floor(p->z / zsize);
}


/**************************************************
**                    THERMOSTAT
**************************************************/
void MDGrow::thermostat(event* ev)
{
    if (ev)
    {
        int i, num;
        particle* p;
        time = ev->time;
        int freq = N / 100;
        if (freq == 0) freq = 1;
        for (i = 0; i < freq; i++)
        {
            num = double_01(rng_64) * N;			//Random particle
            p = particles + num;
            double imsq = 1.0 / sqrt(p->mass);
            update(p);
            p->vx = normal(rng_64) * imsq;			//Kick it
            p->vy = normal(rng_64) * imsq;
            p->vz = normal(rng_64) * imsq;
            p->counter++;
            removeevent(p->firstcollision);
            findcollisions(p);
        }
        removeevent(ev);
        temperature = get_temperature();
        if(fabs(temperature-1.)<1e-6) stop = 1;
    }
    createevent(time + thermostatinterval, nullptr, nullptr, 200);     //Add next write interval
}

/**************************************************
**                    WRITELAST
**************************************************/
void MDGrow::writelast()
{
    int i;
    particle* p;
    FILE* file;
    char filename[200];
    sprintf(filename, "last.sph");
    file = fopen(filename, "w");
    fprintf(file, "%d\n%lf %lf %lf\n", (int)N, xsize, ysize, zsize);
    for (i = 0; i < N; i++)
    {
        p = &(particles[i]);
        fprintf(file, "%c %.12lf  %.12lf  %.12lf  %.12lf\n", 'a' + p->type, p->x, p->y, p->z, p->r);
    }
    fclose(file);
}

double MDGrow::random_gaussian()
{
    static int have_deviate = 0;
    static double u1, u2;
    double  x1, x2, w;

    if (have_deviate)
    {
        have_deviate = 0;
        return u2;
    }
    else
    {
        do
        {
            x1 = 2.0 * double_01(rng_64) - 1.0;
            x2 = 2.0 * double_01(rng_64) - 1.0;
            w = x1 * x1 + x2 * x2;
        } while (w >= 1.0);
        w = sqrt((-2.0 * log(w)) / w);
        u1 = x1 * w;
        u2 = x2 * w;
        have_deviate = 1;
        return u1;
    }
}


