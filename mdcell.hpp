#ifndef _MDCELL_HPP_
#define _MDCELL_HPP_

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <random>
#include "types.hpp"
// #include "mt19937ar.c"
// #include "mdCell.h"




// int main();


class MDCell{
protected:
    double maxtime = 10;           //Simulation stops at this time
    int makesnapshots = 0;          //Whether to make snapshots during the run (yes = 1, no = 0)
    double writeinterval = 1;     //Time between output to screen / data file
    double snapshotinterval = 1;  //Time between snapshots (should be a multiple of writeinterval)

    int initialconfig = 2;    //= 0 load from file, 1 = FCC crystal
    char inputfilename[100] = "init.sph"; //File to read as input snapshot (for initialconfig = 0)
    double packfrac = 0.49;                     //Packing fraction (for initialconfig = 1)
    // int N = 4000;             //Number of particles (for FCC)

    //Variables related to the event queueing system. These can affect efficiency.
    //The system schedules only events in the current block of time with length "eventlisttime" into a sorted binary search tree. 
    //The rest are scheduled in unordered linked lists associated with the "numeventlists" next blocks.
    //"numeventlists" is roughly equal to maxscheduletime / eventlisttime
    //Any events occurring even later are put into an overflow list
    //After every time block with length "eventlisttime", the set of events in the next linear list is moved into the binary search try.
    //All events in the overflow list are also rescheduled.

    //After every "writeinterval", the code will output two listsizes to screen. 
    //The first is the average number of events in the first that gets moved into the event tree after each block.
    //The second is the length of the overflow list at the last time it was looped over.
    //Ideally, we set maxscheduletime large enough that the average overflow list size is negligible (i.e. <10 events)
    //Also, there is some optimum value for the number of events per block (scales approximately linearly with "eventlisttime").
    //I seem to get good results with an eventlisttime chosen such that there are a few hundred events per block, and dependence is pretty weak (similar performance in the range of e.g. 5 to 500 events per block...)
    double maxscheduletime = 1.0;
    int numeventlists;
    double eventlisttimemultiplier = 1;  //event list time will be this / N
    double eventlisttime;

    //Internal variables
    double simtime = 0;
    double reftime = 0;
    int currentlist = 0;
    int totalevents;

    int listcounter1 = 0, listcounter2 = 0, mergecounter = 0;

    event_c** eventlists; //Last one is overflow list

    particle_c* particles;
    // particle_c* celllist[MAXCEL][MAXCEL][MAXCEL];
    particle_c** celllist;
    event_c* eventlist;
    event_c* root;
    event_c** eventpool;
    int nempty = 0;
    double xsize, ysize, zsize; //Box size
    double hx, hy, hz; //Half box size
    double cxsize, cysize, czsize; //Cell size
    int    cx, cy, cz;  //Number of cells
    double dvtot = 0;   //Momentum transfer (for calculating pressure)
    unsigned int colcounter = 0; //Collision counter (will probably overflow in a long run...)


    const int usethermostat = 0; //Whether to use a thermostat
    double thermostatinterval = 0.01;    //Time interval between applications of thermostat
    //random:
    std::mt19937_64 rng_64;
    std::uniform_real_distribution<double> double_01;
    std::normal_distribution<double> normal;

public:
    MDCell();
    ~MDCell();
    void printstuff();
    // template <typename T>    
    void init(Box_t ,particle*);
    void run();

    // template <typename T>
    void copy_confs(particle*);
    void copy_box(Box_t box);
    void update_to_print_time(particle_c* p1, double print_time);
    particle_c *get_partciles_at_print_time(double print_time);
    int overlap(particle_c* part);
    particle_c *get_particles_c_deep(particle_c *p);


    void initparticles(int n);
    void update(particle_c* p1);
    void removefromcelllist(particle_c* p1);
    void addeventtotree(event_c* newevent);


    void randomparticles();
    void initeventpool();
    void fcc();
    void loadparticles();
    void randommovement();
    void initcelllist();
    void addtocelllist(particle_c* p);
    void step();
    void findcollision(particle_c*, particle_c*);
    void findallcollisions();
    void findcollisions(particle_c*, particle_c*);
    void findcollisioncell(particle_c*, int);
    void findcrossing(particle_c* part);
    void collision(event_c*);
    void fakecollision(event_c*);
    void addfakecollision(particle_c*);
    void findlistupdate(particle_c*);
    void cellcross(event_c*);
    void addevent (event_c*);
    void createevent (double time, particle_c* p1, particle_c* p2, int type);
    void addnexteventlist();

    void removeevent (event_c*);
    void outputsnapshot();
    void write();
    void thermostat();
    void backinbox(particle_c* p);

};
// double random_gaussian();
#endif