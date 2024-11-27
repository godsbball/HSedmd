#ifndef _TYPE_HPP_
#define _TYPE_HPP_

#include <stdint.h>


typedef struct Box_c
{
    double xsize, ysize, zsize;				//Particle type
} Box_t;

typedef struct spos
{
    double x, y, z;
    double r;
    int boxestraveledx, boxestraveledy, boxestraveledz;
} pos_t;


//GROW:
#define MAXNEIGH 48
//Size of array allocated for the event tree (currently overkill)
#define MAXEVENTS (N*5)
#define MAXNUMEVENTLISTS (5*N)
//Maximum number of cells in each direction
#define CEL 75
#define CEM 3
//Pi (if not already defined)
#ifndef M_PI
#define M_PI 3.1415926535897932
#endif
typedef struct sevent
{
	double time;
	struct sevent* child1;
	struct sevent* child2;
	struct sevent* parent;
	struct sparticle* p1;
	struct sparticle* p2;
	struct sevent* prevq;
    struct sevent* nextq;
	uint8_t type;
	int queue;
	unsigned int counter2;
} event;


typedef struct sparticle
{
	double x, y, z;
	double vx, vy, vz;
	double xn, yn, zn;
	double vr;
	struct sparticle* neighbors[MAXNEIGH];
	uint8_t nneigh;
	double t;
	double r;
	double rtarget;
	double mass;
	uint8_t edge;
	uint8_t cellx, celly, cellz;
	int boxestraveledx, boxestraveledy, boxestraveledz;
	event* firstcollision;
	unsigned int counter;
	struct sparticle* prev, * next;
	int number;
	uint8_t type;
} particle;





//CELL:
//Size of array allocated for the event tree (currently overkill)
#define MAXEVENTS_C (20)
//Maximum number of cells in each direction
#ifndef MAXCEL
  #define MAXCEL 75
#endif
//Pi (if not already defined)
#ifndef M_PI
  #define M_PI 3.1415926535897932
#endif
//Event structure
typedef struct sevent_c
{
    double eventtime;
    struct sevent_c* left;		//Left child in tree or previous event in event list
    struct sevent_c* right;		//Right child in tree or next event in event list
    struct sevent_c* parent;		//Parent in tree
    struct sparticle_c* p1;		//Particles involved in the event
    struct sparticle_c* p2;
    struct sevent_c* prevp1, *nextp1;	//Circular linked list for all events involving p1 as the first particle
    struct sevent_c* prevp2, *nextp2; //Circular linked list for all events involving p2 as the second particle
    int eventtype;
    int queue;					//Index of the event queue this event is in
} event_c;

//Particle structure
typedef struct sparticle_c
{
    double x, y, z;				//Position
    double vx, vy, vz;			//Velocity
  	double t;					//Last update time
    double r;
    double mass;
    uint8_t nearboxedge;		//Is this particle in a cell near the box edge?
    int cellx, celly, cellz; 					//Current cell
  	int boxestraveledx, boxestraveledy, boxestraveledz;	//Keeps track of dynamics across periodic boundaries
  	event_c* cellcrossing;		//Cell crossing event
	  struct sparticle_c* prev, *next;	//Doubly linked cell list
	  uint8_t type;				//Particle type
} particle_c;



#endif