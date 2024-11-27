// Event driven MD, headers.
#ifndef _MDGROW_HPP_
#define _MDGROW_HPP_

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
// #include <list>
#include <random>
#include <cassert>
#include <algorithm>
#include <cstring>
#include <climits>
#include <numeric>
#include "mdcell.hpp"
#include "types.hpp"

//Code for creating densely packed initial configurations. Particles start small in a random configuration, then grow to the desired size.
//The code is set up for binary mixtures. The parameters below control the number of particles, size ratio, composition, target packing fraction, etc.

//Number of particles:
// #define N 125


template<class T>
void argsort(T *array, long num, long *index);

double MSD_2time_1spl(pos_t *pos_tw, particle_c *p_tptw, Box_t box);
double MSD_2time_half_1spl(pos_t *pos_tw, particle_c *p_tptw, Box_t box, long *idx);
double correlation_2time_1spl(pos_t *pos_tw, particle_c *p_tptw, Box_t box);
double correlation_self(pos_t *posbar_tptw);
double MSD_AB(particle_c **ps_tptw, Box_t box, int samples);
double MSD_AB_half(particle_c **ps_tptw, Box_t box, int samples, long *idx);
double MSD_AB_half2(particle_c *ps_tptw, Box_t box, int samples, long *idx);
int main(int, char**);

class MDGrow{
public:
    // MDGrow(int N_=1000, double growthspeed_=0.1, double targetpackfrac_=0.58);
    MDGrow(double growthspeed_=0.0, double targetpackfrac_=0.64);
    ~MDGrow();

    void printstuff();
    void init();
    void cubicfcc();
    void cubicfcc2();

    void initeventpool();
    void fcc();
    int mygetline(char* str, FILE* f);
    void randomparticles();
    void randommovement();
    void update_specific_t(particle* p1, double stime);
    void update(particle* p1);
    void initcelllist();
    void removefromcelllist(particle* p1);
    void addtocelllist(particle*);

    void step();
    void step_specific_time();
    double findcollision(particle*, particle*, double);
    void findallcollisions();
    void findcollisions(particle*);
    void collision(event*);

    int run_2();
    void update_to_print_time(particle* p1, double print_time);
    sparticle *get_partciles_at_print_time(double print_time);
    int read_confs(char *fname);
    void init_read_confs(char *fname);
    void initcelllist_without_confs();
    void nvt(double);
    void nve();
    void quench();
    template <typename T>
    int copy_confs(T *);
    template <typename T>
    void init_copy_confs(T *);
    void clear_except_particles();
    void assign_except_particles();
    int without_confs();
    void init_without_confs();
    double get_time();
    void init_varibles();
    void init_particles();
    Box_t get_box();





    double get_target_time();
    double get_packing_fraction();
    double get_temperature();
    sparticle *get_paritcles();
    pos_t *get_pos_deep(pos_t *);
    particle_c *get_particles_c_deep(particle_c *p);
    particle_c *get_particles_c_head(particle_c *p);


    void addeventtotree(event* newevent);
    void addevent(event*);
    void removeevent(event*);
    event* createevent(double time, particle* p1, particle* p2, int type);
    void addnexteventlist();
    double findneighborlistupdate(particle* p1);
    void makeneighborlist(particle* p1, int firsttime);



    void showtree();
    void shownode(event*);
    void checktree();
    int checknode(event*);
    int overlap(particle*);
    int overlaplist(particle* part, int error);
    void outputsnapshot();
    void write();
    void writelast();
    void checkcells();
    void thermostat(event* ev);
    double random_gaussian();
    void backinbox(particle* p);
    // void writelast();

    int run();

protected:
    // int N;
    double targetpackfrac = 0.58; //Target packing fraction (if too high, simulation will not finish or crash)  
    double composition = 0.3;     //Fraction of large particles
    double sizeratio = 0.85;      //small diameter / large diameter (must be <= 1)
    double growthspeed = 0.1;     //Factor determining growth speed (slower growth means higher packing fractions can be reached)
    double thermostatinterval = 1e-3;  //Time between applications of thermostat, which gets rid of excess heat generated while growing

    int makesnapshots = 0;        //Whether to make snapshots during the run (yes = 1, no = 0)
    double writeinterval = 1;     //Time between output to screen / data file
    double snapshotinterval = 1;  //Time between snapshots (should be a multiple of writeinterval)


    //Variables related to the event queueing system. These can affect efficiency.
    //The system schedules only events in the current block of time with length "eventlisttime" into a sorted binary search tree. 
    //The rest are scheduled in unordered linked lists associated with the "numeventlists" next blocks.
    //"numeventlists" is roughly equal to maxscheduletime / eventlisttime
    //Any events occurring even later are put into an overflow list
    //After every time block with length "eventlisttime", the set of events in the next linear list is moved into the binary search try.
    //All events in the overflow list are also rescheduled.

    //After every "writeinterval", the code will output two listsizes to screen. 
    //The first is the average number of events in the first that gets moved into the event tree after each block.
    //The second is the average length of the overflow list.
    //Ideally, we set maxscheduletime large enough that the average overflow list size is negligible (i.e. <10 events)
    //Also, there is some optimum value for the number of events per block (scales approximately linearly with "eventlisttime").
    //I seem to get good results with an eventlisttime chosen such that there are a few hundred events per block, and dependence is pretty weak (similar performance in the range of e.g. 5 to 500 events per block...)
    
    double maxscheduletime = 5;
    int numeventlists;
    double eventlisttime = 2 / (float)N;

    double maxdt = 1;

    //Neighbor lists
    double shellsize = 1.5;
    sevent ev_null = {0., 
        nullptr, nullptr,
        nullptr, nullptr, 
        nullptr, nullptr, 
        nullptr,0,0,0};
    //Internal variables
    double time = 0;
    double maxt = -1;

    double reftime = 0;
    int currentlist = 0;
    const double never = 9999999999999999999999.9;


    int listcounter1 = 0, listcounter2 = 0, mergecounter = 0;
    int nbig;

    event** eventlists; //Last one is overflow list. MAXNUMEVENTLISTS + 1

    // particle particles[N+1];
    particle *particles, *particles2;
    particle** celllist;  // cx*cy*cz
    event *eventlist;  // MAXEVENTS
    event* root;
    event** eventpool;//MAXEVENTS
    int nempty = 0;
    double density;
    double xsize, ysize, zsize; //Box size
    double hx, hy, hz; //Half box size
    double cxsize, cysize, czsize; //Cell size
    int    cx, cy, cz;  //Number of cells
    double dvtot = 0;   //Momentum transfer (for calculating pressure)
    unsigned int colcounter = 0; //Collision counter (will probably overflow in a long run...)
    int stop = 0;
    double target_time = 0.;
    double temperature = 1.;

    int usethermostat = 0; //Whether to use a thermostat
    // FILE *fout_bad = fopen("bad_pos.dat","w+");

    //random:
    std::mt19937_64 rng_64;
    std::uniform_real_distribution<double> double_01;
    std::normal_distribution<double> normal;


};

#endif
