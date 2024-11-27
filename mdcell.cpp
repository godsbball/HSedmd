#include "mdcell.hpp"
// int main()
// {
//     MDCell md;
//     md.run();
//   return 0;
// }

MDCell::MDCell()
:rng_64(std::random_device{}()),double_01(0,1), normal(0,1){
    // N = N_;
    // sevent ev_null = {0., 
    //     nullptr, nullptr,
    //     nullptr, nullptr, 
    //     nullptr, nullptr, 
    //     nullptr,0,0,0};
    // growthspeed = growthspeed_;
    // targetpackfrac = targetpackfrac_;

    // particles = (sparticle *)malloc(sizeof(sparticle) * (N ));
    // // particles = new sparticle[N];
    // // eventlists = (sevent **) malloc(sizeof(sevent_c*) * (MAXNUMEVENTLISTS + 1));
    // eventlist = (sevent *) malloc(sizeof(sevent) * MAXEVENTS_C);
    // eventpool = (sevent **) malloc(sizeof(sevent_c*) * (MAXEVENTS_C));
    // for(int i=0; i<MAXEVENTS_C; i++){
    //     // eventpool[i] = (event *) malloc(sizeof(event) * (MAXEVENTS_C));
    //     eventpool[i] = nullptr;
    //     // eventlists[i] = (event *) malloc(sizeof(event) * (MAXNUMEVENTLISTS + 1));
    //     eventlists[i] = nullptr;
    // }

    // particles = new sparticle[N];
    // eventlist = new sevent[MAXEVENTS_C];
    // eventlist = (sevent *)malloc()
    // particles2 = new sparticle[N];
}

MDCell::~MDCell(){
    int i;
    particles = nullptr;
    free(particles);

    for(i=0;i<cx*cy*cz;i++) celllist[i] = nullptr;
    celllist = nullptr;
    free(celllist);

    for(i=0;i<(numeventlists + 1);i++) eventlists[i] = nullptr;
    eventlists = nullptr;
    free(eventlists);

    eventlist = nullptr;//??
    free(eventlist);

    for(i=0;i<totalevents;i++) eventpool[i] = nullptr;
    eventpool = nullptr;
    free(eventpool);
}
void MDCell::run(){
    // Box_c b;
    // particle_c* p;
    //     init(b,p);
    // printf("Starting\n");

    while (simtime <= maxtime)
    {
      step();
    }
    simtime = maxtime;

    printstuff();
    outputsnapshot();
}
/**************************************************
**                 PRINTSTUFF
** Some data at the end of the simulation
**************************************************/
void MDCell::printstuff()
{
  int i;
  particle_c* p;
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
  double press = -dvtot / (3.0 * volume * simtime);
  double pressid = dens;
  double presstot = press + pressid;
  printf("Total time simulated  : %lf\n", simtime);
//  printf ("Density               : %lf\n", (double) N / volume);
  printf("Packing fraction      : %lf\n", vfilled / volume);
  printf("Measured pressure     : %lf + %lf = %lf\n", press, pressid, presstot);
  
}


/**************************************************
**                    INIT
**************************************************/
void MDCell::init(Box_t box, particle* confs)
{
    int i; 

    if (initialconfig == 0) loadparticles();
    else if(initialconfig == 1)  fcc();
    else if (initialconfig == 2) {copy_box(box); copy_confs(confs);}
    randommovement();  
    hx = 0.5 * xsize; hy = 0.5 * ysize; hz = 0.5 * zsize;	//Values used for periodic boundary conditions

    initeventpool();
    for (i = 0; i < N; i++)
    {
        particle_c* p = particles + i;
        // p->boxestraveledx = 0;
        // p->boxestraveledy = 0;
        // p->boxestraveledz = 0;
        p->t = 0;
    }
    initcelllist();
    for (i = 0; i < N; i++)
    {
        particle_c* p = particles + i;
        p->cellcrossing = eventpool[nempty - 1];		//Connect cellcrossing event to particle
        findcrossing(p);					//Find the next cell crossing
    }

    printf("Done adding cell crossings: %d events\n", totalevents - nempty);

    findallcollisions();
    printf("Done adding collisions: %d events\n", totalevents - nempty);
  
  
  if (usethermostat)
  {
      createevent(thermostatinterval, nullptr, nullptr, 101);     //Add next write interval
      printf("Started thermostat: %d events\n", totalevents - nempty);
  }

}
// template void MDCell::init<particle_c>(Box_t box, particle_c* confs);
// template void MDCell::init<spos>(Box_t box, spos* confs);

/******************************************************
**               INITPARTICLES
** Allocate memory for particle array
******************************************************/
void MDCell::initparticles(int n)
{
    // N = n;
    particles = (particle_c*)calloc(N, sizeof(particle_c));
    if (!particles)
    {
        printf("Failed to allocate memory for particles\n");
        exit(3);
    }
}


/******************************************************
**               MYGETLINE
** Reads a single line, skipping over lines
** commented out with #
******************************************************/
int mygetline(char* str, FILE* f)
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
**                    FCC
** Initialize system on a face-centered cubic
** lattice
**************************************************/
void MDCell::fcc()
{
    int i, j, k;
    particle_c* p;

    int ncell = cbrt(N / 4) + 0.0001;
    
    if (ncell * ncell * ncell * 4 != N)
    {
        printf("N should be 4 * a perfect cube! (e.g. %d)\n", ncell * ncell * ncell * 4);
        exit(3);
    }
    
    double volume = N / (6.0 / M_PI * packfrac);
    xsize = cbrt(volume);
    ysize = xsize;
    zsize = xsize;
    
    double step = xsize / ncell;

    printf("step: %lf\n", step);
    initparticles(N);
    printf("Placing particles\n");
    
    p = particles;
    for (i = 0; i < ncell; i++) for (j = 0; j < ncell; j++) for (k = 0; k < ncell; k++)
    {
        p->x = (i + 0.25) * step;
        p->y = (j + 0.25) * step;
        p->z = (k + 0.25) * step;
        p++;
        p->x = (i + 0.75) * step;
        p->y = (j + 0.75) * step;
        p->z = (k + 0.25) * step;
        p++;
        p->x = (i + 0.75) * step;
        p->y = (j + 0.25) * step;
        p->z = (k + 0.75) * step;
        p++;
        p->x = (i + 0.25) * step;
        p->y = (j + 0.75) * step;
        p->z = (k + 0.75) * step;
        p++;
    }

    for (i = 0; i < N; i++)
    {
        p = particles + i;
        p->r = 0.5;
        p->type = 0;
        p->mass = 1;
    }
    
    printf("Packing fraction: %lf\n", M_PI / (6.0 * xsize * ysize * zsize) * N);
    printf("Starting configuration from fcc crystal\n");
}

/**************************************************
**                    LOADPARTICLES
** Read particles from file
** First line: Number of particles
** Second line: box size along all three axes
** Rest: particle data
** Each particle line consists of :
** - A character indicating type (a = 0, b = 1, etc.)
** - 3 coordinates (x, y, z)
** - The r
**************************************************/
void MDCell::loadparticles()
{
    char tmp;
    int i, npart;
    particle_c* p;
    char buffer[255];
    
    FILE* file;
    file = fopen(inputfilename, "r");
    if (!file)
    {
      printf("File not found!\n");
      exit(3);
    }
    mygetline(buffer, file);
    int ftmp = sscanf(buffer, "%d", &npart);
    if (ftmp != 1) { printf("Read error (n or box)\n"); exit(3); }
    mygetline(buffer, file);
    ftmp = sscanf(buffer, "%lf %lf %lf\n", &xsize, &ysize, &zsize);
    if (ftmp != 3) { printf("Read error (n or box)\n"); exit(3); }


    initparticles(npart);
    printf("Placing particles\n");
    double vfilled = 0;
    for (i = 0; i < N; i++)
    {
        p = &(particles[i]);
        mygetline(buffer, file);
        ftmp = sscanf(buffer, "%c %lf  %lf  %lf %lf\n", &tmp, &(p->x), &(p->y), &(p->z), &(p->r));
        backinbox(p);
        if (ftmp != 5) { printf("Read error (particle) %d \n String: %s\n", ftmp, buffer); exit(3); }
        p->type = tmp - 'a';
        p->mass = 1;
        vfilled += 8*p->r*p->r*p->r;
        p->cellcrossing = nullptr;
    }
    fclose(file);

    printf("Packing fraction: %lf\n", M_PI / (6.0 * xsize * ysize * zsize) * vfilled);
}

// template <typename T>
void MDCell::copy_confs(particle* confs)
{
    int i, npart;
    particle_c* p;

    initparticles(npart);
    printf("Placing particles\n");
    double vfilled = 0;
    for (i = 0; i < N; i++)
    {
        p = &(particles[i]);
        p->x = confs[i].x;
        p->y = confs[i].y;
        p->z = confs[i].z;
        p->r = confs[i].r;
        p->boxestraveledx = confs[i].boxestraveledx;
        p->boxestraveledy = confs[i].boxestraveledy;
        p->boxestraveledz = confs[i].boxestraveledz;
        backinbox(p);

        p->mass = 1;
        vfilled += 8*p->r*p->r*p->r;
        p->cellcrossing = nullptr;
    }
    printf("Packing fraction: %lf\n", M_PI / (6.0 * xsize * ysize * zsize) * vfilled);
}
// template void MDCell::copy_confs<particle_c>(particle_c* confs);
// template void MDCell::copy_confs(pos_t* confs);

void MDCell::copy_box(Box_t box)
{
    xsize = box.xsize;
    ysize = box.ysize;
    zsize = box.zsize;
}

// template void MDCell::copy_confs(pos_t* confs);

/**************************************************
**                RANDOMMOVEMENT
** Assign random velocities to all particles
** Center-of-mass velocity = 0
** Kinetic energy per particle = 3kT/2
**************************************************/
void MDCell::randommovement()
{
    particle_c* p;
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
    for (i = 0; i < N; i++)
    {
        p = &(particles[i]);
        p->vx *= fac;					//Fix energy
        p->vy *= fac;
        p->vz *= fac;
    }
    printf("Starting configuration read from %s\n", inputfilename);
}

/**************************************************
**                UPDATE
** Update particle to the current time
**************************************************/
void MDCell::update(particle_c* p1)
{
    double dt = simtime - p1->t;
    p1->t = simtime;
    p1->x += dt * p1->vx;
    p1->y += dt * p1->vy;
    p1->z += dt * p1->vz;
}

void MDCell::update_to_print_time(particle_c* p1, double print_time)
{
    double dt = print_time - p1->t;
    p1->t = print_time;
    p1->x += dt * p1->vx;
    p1->y += dt * p1->vy;
    p1->z += dt * p1->vz;
    // p1->r += dt * p1->vr;
}
particle_c *MDCell::get_partciles_at_print_time(double print_time){
    // sparticle *particles;
    maxtime = print_time;
    printf("simtime = %lf\n",simtime);
    run(); // determine   time (now) < print_time < ev->time (next collision time)

    for(int i=0; i<N; i++){
        update_to_print_time(particles + i, print_time);
    }
    return particles;
}

particle_c *MDCell::get_particles_c_deep(particle_c *p){
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

/**************************************************
**                 INITCELLLIST
** Initialize the cell list
** Cell size should be at least:
**    shellsize * [diameter of largest sphere]
** Can use larger cells at low densities
**************************************************/
void MDCell::initcelllist()
{
    int i, j, k;
    cx = (int)(xsize - 0.0001) ;				//Set number of cells
    cy = (int)(ysize - 0.0001) ;
    cz = (int)(zsize - 0.0001) ;

	while (cx*cy*cz > 8*N)
    {
        cx *= 0.9;
        cy *= 0.9;
        cz *= 0.9;
    }   
    printf("Cells: %d, %d, %d\n", cx, cy, cz);
    int hitmax = 0;
    if (cx > MAXCEL) { cx = MAXCEL; hitmax = 1; }
    if (cy > MAXCEL) { cy = MAXCEL; hitmax = 1; }
    if (cz > MAXCEL) { cz = MAXCEL; hitmax = 1; }
    if (hitmax) printf("Maximum cellsize reduced. Consider increasing MAXCEL. New values:  %d, %d, %d\n", cx, cy, cz);
cx=cy=cz = ceil(pow(1.*N/125,1./3)*3.);

    celllist = (particle_c**)malloc(cx*cy*cz * sizeof(particle_c*));		//Allocate cell list
    cxsize = xsize / cx;						//Set cell size
    cysize = ysize / cy;
    czsize = zsize / cz;
    for (i = 0; i < cx; i++)					//Clear celllist
       for (j = 0; j < cy; j++)
           for (k = 0; k < cz; k++)
            {
                celllist[(i)*cy*cz + (j)*cz + k] = nullptr;
            }
    for (i = 0; i < N; i++) addtocelllist(particles + i);
}

int MDCell::overlap(particle_c* part)
{
    particle_c* p;
    double dx, dy, dz, r2, rm;
    int i;
    double dl = pow(10.0, -10);
    // double dl = 0.0;

    for (i = 0; i < N; i++)
    {
        p = &(particles[i]);
        if (p == part) continue;

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
            printf("Overlap: %lf, %d \n", r2 - rm * rm, i);
            return 1;
        }
    }
    return 0;
}

/**************************************************
**                    ADDTOCELLLIST
** Add particle to the cell list.
** Note that each cell is implemented as a doubly
** linked list.
**************************************************/
void MDCell::addtocelllist(particle_c* p)
{
    p->cellx = p->x / cxsize;				//Find particle's cell
    p->celly = p->y / cysize;
    p->cellz = p->z / czsize;
    p->next = celllist[(p->cellx)*cy*cz + (p->celly)*cz + p->cellz];	//Add particle to celllist
    if (p->next) p->next->prev = p;			//Link up list
    celllist[(p->cellx)*cy*cz + (p->celly)*cz + p->cellz] = p;
    p->prev = nullptr;
    p->nearboxedge = (p->cellx == 0 || p->celly == 0 || p->cellz == 0 || p->cellx == cx - 1 || p->celly == cy - 1 || p->cellz == cz - 1);
    
}

/**************************************************
**               REMOVEFROMCELLLIST
**************************************************/
void MDCell::removefromcelllist(particle_c* p1)
{
    if (p1->prev) p1->prev->next = p1->next;    //Remove particle from celllist
    else          celllist[(p1->cellx)*cy*cz + (p1->celly)*cz + p1->cellz] = p1->next;
    if (p1->next) p1->next->prev = p1->prev;
}

/**************************************************
**                     STEP
** Process a single event
**************************************************/
void MDCell::step()
{
    event_c* ev;
    ev = root->right;
    while (ev == nullptr)                  //Need to include next event list?
    {
        addnexteventlist();
        ev = root->right;
    }
    
    while (ev->left) ev = ev->left;		//Find first event

    simtime = ev->eventtime;
    switch(ev->eventtype)
    {
        case 0:
            collision(ev);
            break;
        case 100:
            removeevent(ev);
            write();
            break;
        case 101:
            removeevent(ev);
            thermostat();
            break;
        default:
            cellcross(ev);
    }
} 
  


/**************************************************
**                FINDCOLLISION
** Detect the next collision for two particles
** Note that p1 is always up to date in
** findcollision
**************************************************/
void MDCell::findcollision(particle_c* p1, particle_c* p2)
{
	double dt2 = simtime - p2->t;
	double dx = p1->x - p2->x - dt2 * p2->vx;    //relative distance at current time
	double dy = p1->y - p2->y - dt2 * p2->vy;
	double dz = p1->z - p2->z - dt2 * p2->vz;
	if (p1->nearboxedge)
	{
		if (dx > hx) dx -= xsize; else if (dx < -hx) dx += xsize;  //periodic boundaries
		if (dy > hy) dy -= ysize; else if (dy < -hy) dy += ysize;
		if (dz > hz) dz -= zsize; else if (dz < -hz) dz += zsize;
	}
	double dvx = p1->vx - p2->vx;                               //relative velocity
	double dvy = p1->vy - p2->vy;
	double dvz = p1->vz - p2->vz;
	
	double b = dx * dvx + dy * dvy + dz * dvz;                  //dr.dv
	if (b > 0) return;
	
	double dv2 = dvx * dvx + dvy * dvy + dvz * dvz;
	double dr2 = dx * dx + dy * dy + dz * dz;
	double md = p1->r + p2->r;

	double disc = b * b - dv2 * (dr2 - md * md);
	if (disc < 0) return;
	double t = simtime + (-b - sqrt(disc)) / dv2;
	createevent(t, p1, p2, 0);
}

/**************************************************
**                FINDCOLLISIONS
** Find all collisions for particle p1.
** The particle 'notthis' isn't checked.
**************************************************/
void MDCell::findcollisions(particle_c* p1, particle_c* notthis)    //All collisions of particle p1
{
	int dx,dy,dz, cellx, celly, cellz,ccx,ccy,ccz;
	cellx = p1->cellx;
	celly = p1->celly;
	cellz = p1->cellz;
	particle_c* p2;

	for ( dx = -1; dx < 2; dx++)
	{
		ccx = cellx + dx;
		if (dx<0 && ccx < 0) ccx += cx; else if (dx >0 &&ccx >= cx) ccx -= cx;
		{
			for ( dy = -1; dy < 2; dy++)
			{
				ccy = celly + dy;
				if (dy<0 &&ccy < 0) ccy += cy; else if (dy >0 &&ccy >= cy) ccy -= cy;
				for ( dz = -1; dz < 2; dz++)
				{
					ccz = cellz + dz;
					if (dz<0 &&ccz < 0) ccz += cz; else if (dz >0 &&ccz >= cz) ccz -= cz;
					for (p2 = celllist[(ccx)*cy*cz + (ccy)*cz + ccz]; p2; p2 = p2->next)
						if (p2 != p1 && p2 != notthis) findcollision(p1,p2);
				}
			}
		}
	}
}


/**************************************************
**                FINDALLCOLLISION
** All collisions of all particle pairs
**************************************************/
void MDCell::findallcollisions()       //All collisions of all particle pairs
{
	int i,dx,dy,dz, cellx, celly, cellz;
	particle_c* p1, *p2;
	
	for (i = 0; i < N; i++)
	{
		cellx = particles[i].cellx + cx;
		celly = particles[i].celly + cy;
		cellz = particles[i].cellz + cz;
		p1 = &(particles[i]);

		for ( dx = cellx-1; dx < cellx+2; dx++)
			for ( dy = celly-1; dy < celly+2; dy++)
				for ( dz = cellz-1; dz < cellz+2; dz++)
				{
					p2 = celllist[(dx % cx)*cy*cz + (dy % cy)*cz + dz % cz];
					while (p2)
					{
					if (p2 > p1) findcollision(p1,p2);
					p2 = p2->next;
					}
				}   
	}
}



/**************************************************
**                FINDCOLLISIONCELL
** Find all collisions for particle p1 after a
** cellcrossing.
** This could find events that have already been
** scheduled, but they will be deleted as soon
** as one of the pair is processed.
**************************************************/
void MDCell::findcollisioncell(particle_c* p1, int type)   
{
  int dx,dy,dz, cellx, celly, cellz,ccx,ccy,ccz;
  cellx = p1->cellx; 
  celly = p1->celly;
  cellz = p1->cellz;
  particle_c* p2;
  
  int xmin = cellx-1;
  int ymin = celly-1;
  int zmin = cellz-1;
  int xmax = cellx+1;
  int ymax = celly+1;
  int zmax = cellz+1;
  switch (type)		
  {
    case -3:		//Negative z-direction
      zmax = zmin; break;
    case -2:		//Negative y-direction
      ymax = ymin; break;
    case -1:		//Negative x-direction
      xmax = xmin; break;
    case 1:			//Positive x-direction
      xmin = xmax; break;
    case 2:			//Positive y-direction
      ymin = ymax; break;
    case 3:			//Positive z-direction
      zmin = zmax; break;
  }
 
  for ( dx = xmin; dx <= xmax; dx++)
  {
    if (dx < 0) ccx = dx+cx; else if (dx >= cx) ccx = dx-cx; else ccx = dx;
    for ( dy = ymin; dy <= ymax; dy++)
    {
      if (dy < 0) ccy = dy+cy; else if (dy >= cy) ccy = dy-cy; else ccy = dy;
      for ( dz = zmin; dz <= zmax; dz++)
      {
        if (dz < 0) ccz = dz+cz; else if (dz >= cz) ccz = dz-cz; else ccz = dz;
        p2 = celllist[(ccx)*cy*cz + (ccy)*cz + ccz];
        while (p2)
        {
          if (p2 != p1) findcollision(p1,p2);
          p2 = p2->next;
        }
      }
    }
  }
}


/**************************************************
**                  COLLISION
** Process a single collision event
**************************************************/
void MDCell::collision(event_c* ev)
{
    particle_c* p1 = ev->p1;
    particle_c* p2 = ev->p2;

    double m1 = p1->mass, r1 = p1->r;
    double m2 = p2->mass, r2 = p2->r;
    
    update(p1);
    update(p2);
    
    double r = r1 + r2;
    double rinv = 1.0 / r;
    double dx = (p1->x - p2->x);			//Normalized distance vector
    double dy = (p1->y - p2->y);
    double dz = (p1->z - p2->z);
    if (p1->nearboxedge)
    {
        if (dx > hx) dx -= xsize; else if (dx < -hx) dx += xsize;  //periodic boundaries
        if (dy > hy) dy -= ysize; else if (dy < -hy) dy += ysize;
        if (dz > hz) dz -= zsize; else if (dz < -hz) dz += zsize;
    }
    dx *= rinv;  dy *= rinv;  dz *= rinv;

    double dvx = p1->vx - p2->vx;                               //relative velocity
    double dvy = p1->vy - p2->vy;
    double dvz = p1->vz - p2->vz;
    
    double b = dx * dvx + dy * dvy + dz * dvz;                  //dr.dv
    b *= 2.0 / (m1 + m2);
    double dv1 = b * m2, dv2 = b * m1;
  
    p1->vx -= dv1 * dx;         //Change velocities after collision
    p1->vy -= dv1 * dy;         //delta v = (-) dx2.dv2
    p1->vz -= dv1 * dz;
    p2->vx += dv2 * dx;
    p2->vy += dv2 * dy;
    p2->vz += dv2 * dz;
    
    dvtot += dv1*m1 * r;
    colcounter++;
  
    event_c* del = p1->cellcrossing;                        //Delete all old events for these particles
    while (del->nextp1 != del) removeevent(del->nextp1); //This includes 'ev' itself
    while (del->nextp2 != del) removeevent(del->nextp2);
    removeevent(del);
    findcrossing(p1);           //Find new cellcrossings
    findcollisions(p1, p2);

    del = p2->cellcrossing;
    while (del->nextp1 != del) removeevent(del->nextp1);
    while (del->nextp2 != del) removeevent(del->nextp2);
    removeevent(del);
    findcrossing(p2);
    findcollisions(p2, p1);


}


/**************************************************
**                  FINDCROSSING
** Find the next cellcrossing for a particle
**************************************************/
void MDCell::findcrossing(particle_c* part)
{
	double t, tmin;
	int type; //Type 1,2,3 refers to x,y,z direction, -1,-2,-3 to negative x,y,z direction 
	double vx = part->vx;
	double vy = part->vy;
	double vz = part->vz;
	
	if (vx < 0)
	{
		tmin = (part->cellx * cxsize - part->x )  / vx;
		type = -1;	
	}
	else
	{
		tmin = ((part->cellx + 1) * cxsize - part->x )  / vx;
		type = 1;
	}
	if (vy < 0)
	{
		t = (part->celly * cysize - part->y ) ;
		if (t > tmin * vy) {type = -2; tmin = t / vy;}
	}
	else
	{
		t = ((part->celly + 1) * cysize - part->y ) ;
		if (t < tmin * vy) {type = 2; tmin = t / vy;}
	}
	if (vz < 0)
	{
		t = (part->cellz * czsize - part->z ) ;
		if (t > tmin * vz) {type = -3; tmin = t / vz;}
	}
	else
	{
		t = ((part->cellz + 1) * czsize - part->z );
		if (t < tmin * vz) {type = 3; tmin = t / vz;}
	}
	createevent(part->t+tmin, part, part, type);
}

/**************************************************
**                  CELLCROSS
** Process a single cellcrossing event
**************************************************/
void MDCell::cellcross(event_c* ev)
{
	particle_c* part = ev->p1;
	int type = ev->eventtype;
	double pt= simtime-part->t;

	part->x += pt * part->vx;			//Update part
	part->y += pt * part->vy;
	part->z += pt * part->vz;
	part->t = simtime;

	if (part->prev) part->prev->next = part->next;    //Remove particle from celllist
	else            celllist[(part->cellx)*cy*cz + (part->celly)*cz + part->cellz] = part->next;
	if (part->next) part->next->prev = part->prev;

	switch (type)
	{
	case 1:     //+x
		part->cellx++; if (part->cellx == cx) {part->cellx = 0; part->x -= xsize; part->boxestraveledx++;}
		break;
	case 2:     //+y
		part->celly++; if (part->celly == cy) {part->celly = 0; part->y -= ysize; part->boxestraveledy++;}
		break;
	case 3:     //+z
		part->cellz++; if (part->cellz == cz) {part->cellz = 0; part->z -= zsize; part->boxestraveledz++;}
		break;
	case -1:    //-x
		if (part->cellx == 0) {part->cellx = cx-1; part->x += xsize; part->boxestraveledx--;} else part->cellx--;
		break;
	case -2:    //-y
		if (part->celly == 0) {part->celly = cy-1; part->y += ysize; part->boxestraveledy--;} else part->celly--;
		break;
	case -3:    //-z
		if (part->cellz == 0) {part->cellz = cz-1; part->z += zsize; part->boxestraveledz--;} else part->cellz--;
		break;
	}

	part->prev = nullptr;					//Add particle to celllist
	part->next = celllist[(part->cellx)*cy*cz + (part->celly)*cz + part->cellz];
	celllist[(part->cellx)*cy*cz + (part->celly)*cz + part->cellz] = part;
	if (part->next) part->next->prev = part;
	part->nearboxedge = (part->cellx == 0 || part->celly == 0 || part->cellz == 0 || part->cellx == cx - 1 || part->celly == cy - 1 || part->cellz == cz - 1);

	removeevent(ev);					//Note that the next event added after this HAS to be the new cellcrossing
	findcrossing(part);					//Find next crossing
	findcollisioncell(part, type);    			//Find collisions with particles in new neighbouring cells
}


/**************************************************
**                 INITEVENTPOOL
**************************************************/
void MDCell::initeventpool()
{
    eventlisttime = eventlisttimemultiplier / N;
    numeventlists = ceil(maxscheduletime / eventlisttime);
    maxscheduletime = numeventlists * eventlisttime;
    printf("number of event lists: %d\n", numeventlists);

    eventlists = (event_c**)calloc(numeventlists + 1, sizeof(event_c*));
    if (!eventlists)
    {
        printf("Failed to allocate memory for eventlists\n");
        exit(3);
    }

    totalevents = N * MAXEVENTS_C;
    eventlist = (event_c*)malloc(totalevents * sizeof(event_c));
    eventpool = (event_c**)malloc(totalevents * sizeof(event_c*));
    int i;
    event_c* e;
    for (i = 0; i < totalevents; i++)			//Clear eventpool
    {
        e = &(eventlist[totalevents - i - 1]);			//Fill in list of free events
        eventpool[i] = e;					
        eventpool[i]->left = nullptr;			
        eventpool[i]->right = nullptr;			
        e->nextp1 = e; e->prevp1 = e;			//Initialize circular lists. Only useful for the cell crossings.
        e->nextp2 = e; e->prevp2 = e;
        nempty++;						//nempty keeps track of the number of free events
    }
    root = eventpool[--nempty];				//Create root event
    root->eventtime = -99999999999.99;      //Root event is empty, but makes sure every other event has a parent
    root->eventtype = 99;					//This makes sure we don't have to keep checking this when adding/removing events
    root->parent = nullptr;

    event_c* writeevent = eventpool[--nempty]; //Setup write event
    writeevent->eventtime = 0;
    writeevent->eventtype = 100;
    root->right = writeevent;
    writeevent->parent = root;
    printf("Event tree initialized: %d events\n", totalevents - nempty);
}

/**************************************************
**                  ADDEVENTTOTREE
** Add event into binary tree
**************************************************/
void MDCell::addeventtotree(event_c* newevent)
{
    double time = newevent->eventtime;
    event_c* loc = root;
    int busy = 1;
    while (busy)						//Find location to add event into tree (loc)
    {
        if (time < loc->eventtime)				//Go left
        {
            if (loc->left) loc = loc->left;
            else
            {
                loc->left = newevent;
                busy = 0;
            }
        }
        else						//Go right
        {
            if (loc->right) loc = loc->right;
            else
            {
                loc->right = newevent;
                busy = 0;
            }
        }
    }
    newevent->parent = loc;
    newevent->left = nullptr;
    newevent->right = nullptr;  
}

/**************************************************
**                  ADDEVENT
** Add event to event calendar
**************************************************/
void MDCell::addevent(event_c* newevent)
{
    double dt = newevent->eventtime - reftime;
	
    if (dt < eventlisttime) //Event in near future: Put it in the tree
    {
      newevent->queue = currentlist;
      addeventtotree(newevent);
    }
    else                    //Put it in one of the event lists
    {
        double listoffset = dt/eventlisttime;
        int list_id;
        if (currentlist + listoffset > INT32_MAX) //Check if list_id is not too large
        {
          list_id = numeventlists;
        }
        else
        {
          list_id = currentlist + dt / eventlisttime;
          if (list_id >= numeventlists)
          {
              list_id -= numeventlists;
              if (list_id > currentlist - 1) list_id = numeventlists; //Overflow
          }
        }

        newevent->queue = list_id;
        newevent->right = eventlists[list_id]; //Add to linear list
        newevent->left = nullptr;
        if (newevent->right) newevent->right->left = newevent;
        eventlists[list_id] = newevent;
    }  
}
/**************************************************
**                  CREATEEVENT
** Create a new event at the provided time,
** involving particles p1, and p2, 
** and with the provided event type.
**************************************************/
void MDCell::createevent(double time, particle_c* p1, particle_c* p2, int type)
{
	event_c* newevent = eventpool[--nempty];		//Pick first unused event
	newevent->eventtime = time;
	newevent->p1 = p1;
	newevent->eventtype = type;
	
	if (type == 0)                        //Event is a collision
	{
		newevent->p2 = p2;
		event_c* cc = newevent->p1->cellcrossing;	//Use cellcrossing event with each particle
		newevent->nextp1 = cc->nextp1;      //... to link this event into their lists
		cc->nextp1 = newevent;		// cellcrossing -> this event -> other ...
		newevent->prevp1 = cc;
		newevent->nextp1->prevp1 = newevent;

		cc = newevent->p2->cellcrossing;
		newevent->nextp2 = cc->nextp2;
		cc->nextp2 = newevent;
		newevent->prevp2 = cc;
		newevent->nextp2->prevp2 = newevent;
	}


	addevent(newevent);
}

/**************************************************
**                     ADDNEXTEVENTLIST
** Sort all events from the first event list 
** into the binary tree.
**************************************************/
void MDCell::addnexteventlist()
{
    currentlist++;
    reftime += eventlisttime;
    if (currentlist == numeventlists) //End of array of event lists?
    {
        currentlist = 0;    //Start at the beginning again
		//Process overflow queue
		event_c* ev = eventlists[numeventlists];
        eventlists[numeventlists] = nullptr;
        listcounter2 = 0;
        while (ev)
        {
            event_c* nextev = ev->right;
            addevent(ev);
            ev = nextev;
            listcounter2 += 1;      //Count how many events there were in overflow
        }            
    }

    event_c* ev = eventlists[currentlist];
    while (ev)
    {
        event_c* nextev = ev->right;
        addeventtotree(ev);
        ev = nextev;
        listcounter1++;         //Count how many events there were in event list
    }
    eventlists[currentlist] = nullptr;
    mergecounter++;
}

/**************************************************
**                  REMOVEEVENT
** Remove an event from the event calendar
**************************************************/
void MDCell::removeevent(event_c* oldevent)
{

    if (oldevent->eventtype == 0)			//Update linked lists if it's  a particle event
    {
        oldevent->nextp1->prevp1 = oldevent->prevp1;
        oldevent->prevp1->nextp1 = oldevent->nextp1;
        oldevent->nextp2->prevp2 = oldevent->prevp2;
        oldevent->prevp2->nextp2 = oldevent->nextp2;
    }
    else if (oldevent->eventtype == 8)
    {
        oldevent->nextp1->prevp1 = oldevent->prevp1;
        oldevent->prevp1->nextp1 = oldevent->nextp1;
    }


    if (oldevent->queue != currentlist) //Not in the binary tree
    {
        if (oldevent->right) oldevent->right->left = oldevent->left;
        if (oldevent->left) oldevent->left->right = oldevent->right;
        else 
        {
            eventlists[oldevent->queue] = oldevent->right;
        }
        eventpool[nempty++] = oldevent;     //Put the removed event back in the event pool.
        return;
    }
    
    event_c* parent = oldevent->parent;
    event_c* node;					//This node will be attached to parent in the end
  
  
	if (oldevent->left == nullptr)			//Only one child: easy to delete
	{
		node = oldevent->right;			//Child2 is attached to parent
		if (node)
		{
			node->parent = parent;
		}
	}
	else if (oldevent->right == nullptr)		//Only one child again
	{
		node = oldevent->left;			//Child1 is attached to parent
		node->parent = parent;
	}
	else		  //Node to delete has 2 children
	{               //In this case: a) Find first node after oldevent     (This node will have no left)
					//              b) Remove this node from the tree     (Attach node->right to node->parent)
					//              c) Put this node in place of oldevent (Oldevent's children are adopted by node)
		node = oldevent->right;
		while (node->left) node = node->left;	//Find first node of right tree of descendants of oldevent
		event_c* pnode = node->parent;
		if (pnode != oldevent)			//node is not a child of oldevent
		{						//Both of oldevent's children should be connected to node
			pnode->left = node->right;		//Remove node from right tree
			if (node->right) node->right->parent = pnode;
			oldevent->left->parent = node;
			node->left = oldevent->left;
			oldevent->right->parent = node;
			node->right = oldevent->right;
		}
		else					//This means node == oldevent->right
		{						//Only left has to be attached to node
			oldevent->left->parent = node;
			node->left = oldevent->left;
		}
		node->parent = parent;
	}
	if (parent->left == oldevent) parent->left = node;
	else                          parent->right = node;
	eventpool[nempty++] = oldevent;     //Put the removed event back in the event pool.
}

/**************************************************
**                  OUTPUTSNAPSHOT
** Write a final snapshot to disk
**************************************************/
void MDCell::outputsnapshot()
{
    char filename[200];
    sprintf(filename, "snapshot_end.sph");
    FILE* file = fopen(filename, "w");
    int i;
    particle_c* p;
    fprintf(file, "%d\n%.12lf %.12lf %.12lf\n", (int)N, xsize, ysize, zsize);
    for (i = 0; i < N; i++)
    {
        p = &(particles[i]);
        double dt = simtime - p->t;
        p->x += p->vx * dt;
        p->y += p->vy * dt;
        p->z += p->vz * dt;
        p->t = simtime;


        fprintf(file, "%c %.12lf  %.12lf  %.12lf  %lf\n", 'a' + p->type, p->x + xsize * p->boxestraveledx, p->y + ysize * p->boxestraveledy, p->z + zsize * p->boxestraveledz, p->r);
    }
    fclose(file);

}


/**************************************************
**                    WRITE
** Writes a movie
**************************************************/
void MDCell::write()
{
    static int counter = 0;
    static int first = 1;
    static double lastsnapshottime = -999999999.9;
    static double dvtotlast = 0;
    static double timelast = 0;   
    int i;
    particle_c* p;
    FILE* file;
    
    double en = 0;
    for (i = 0; i < N; i++)
    {
        p = particles + i;
        en += p->mass * (p->vx * p->vx + p->vy * p->vy + p->vz * p->vz);
    }
    double temperature = 0.5 * en / (float)N / 1.5;

    double volume = xsize * ysize * zsize;
    double pressid = (double)N / volume;
    double pressnow = -(dvtot - dvtotlast) / (3.0 * volume * (simtime - timelast));
    pressnow += pressid;
    dvtotlast = dvtot;
    timelast = simtime;
    if (colcounter == 0) pressnow = 0;
    
    double listsize1 = (double)listcounter1 / mergecounter;     //Average number of events in the first event list
    int listsize2 = listcounter2;                               //Number of events in overflow list during last rescheduling (0 if not recently rescheduled)
    if (mergecounter == 0) listsize1 = 0;
    listcounter1 = listcounter2 = mergecounter = 0;

    // printf("Simtime: %lf, Collisions: %u, Press: %lf, T: %lf, Listsizes: (%lf, %d)\n", 
			// simtime, colcounter, pressnow, temperature, listsize1, listsize2);
    char filename[200];
    if (makesnapshots && simtime - lastsnapshottime > snapshotinterval - 0.001)
    {
        sprintf(filename, "../data/mov.n%d.v%.4lf.sph", N, xsize * ysize * zsize);
        if (first) { first = 0;  file = fopen(filename, "w"); }
        else                     file = fopen(filename, "a");
        fprintf(file, "%d\n%.12lf %.12lf %.12lf\n", (int)N, xsize, ysize, zsize);
        for (i = 0; i < N; i++)
        {
            p = &(particles[i]);
            update(p);

            fprintf(file, "%c %.12lf  %.12lf  %.12lf  %lf\n", 
				'a' + p->type, 
				p->x + xsize * p->boxestraveledx, 
				p->y + ysize * p->boxestraveledy, 
				p->z + zsize * p->boxestraveledz, p->r);
        }
        fclose(file);
        lastsnapshottime = simtime;
    }

    //Print some data to a file
    sprintf(filename, "../data/press.n%d.v%.4lf.sph", N, xsize * ysize * zsize);
    if (counter == 0) file = fopen(filename, "w");
    else              file = fopen(filename, "a");
    fprintf(file, "%lf %lf\n", simtime, pressnow);
    fclose(file);

    counter++;
  
	//Schedule next write event    
	createevent(simtime + writeinterval, nullptr, nullptr, 100);     //Add next write interval
}

/**************************************************
**                    BACKINBOX
** Apply periodic boundaries
** Just for initialization
**************************************************/
void MDCell::backinbox(particle_c* p)
{
    p->boxestraveledx += floor(p->x / xsize);
    p->boxestraveledy += floor(p->y / ysize);
    p->boxestraveledz += floor(p->z / zsize);
    p->x -= xsize * floor(p->x / xsize);
    p->y -= ysize * floor(p->y / ysize);
    p->z -= zsize * floor(p->z / zsize);
}


/**************************************************
**                    THERMOSTAT
** Simple thermostat
** Only used when "usethermostat" is non-zero
** Kicks a random selection of particles 
** periodically, giving them a new velocity from
** the Maxwell-Boltzmann distribution
**************************************************/
void MDCell::thermostat()
{
    int i, num;
    particle_c* p;
    event_c* del;
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
        del = p->cellcrossing;			//Delete all old events for this particle
        while (del->nextp1 != del) removeevent(del->nextp1);
        while (del->nextp2 != del) removeevent(del->nextp2);
        removeevent(del);
        findcrossing(p);           //Find new cellcrossings
        findcollisions(p, nullptr);
    }
    //Schedule next thermostat event
    createevent(simtime + thermostatinterval, nullptr, nullptr, 101);     //Add next write interval
}



