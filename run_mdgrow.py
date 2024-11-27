import os
import subprocess
# from tool import *
import shutil
import time
from Const import *
from Log import *
from utils import *
# from dp import *

class MDGrow_preparation():
    def __init__(self,num,phi,StepMax,samples=6,phi0=0.63):
        self.num = num
        self.phi = phi
        self.stepmax = StepMax
        self.samples = samples
        self.phi0 = phi0
        self.dirdata = mkdir("../data/")
        self.dir = mkdir("../data/N%d_phi=%.2le_StepMax=%02d/"%(num,phi,StepMax))
        self.dir_t = mkdir("../data/N%d_phi=%.2le_StepMax=%02d/-t_samples=%06d/"%(num,phi,StepMax,samples))
        
        #LOG
        logpath = self.dir + "my.log"
        self.logger = init_logger(path=logpath)
        self.logger.info("finish mkdir.")
    def shell(self,cmd):
        run_again_num = 0
        while 1:
            while_index = 1
            printlists = []
            p = subprocess.Popen(cmd,shell=True, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)  
            for printlist in iter(p.stdout.readline, b''):
                if "Segmentation fault" in printlist.decode("utf-8",  "ignore").strip():
                    self.logger.warning("Barriers need more memory.")
                    while_index = 0
                elif "./genrem: Text file busy" in printlist.decode("utf-8",  "ignore").strip():
                    self.logger.warning("./genrem: Text file busy.")
                    while_index = 0  #
                elif "./genrem: Permission denied" in printlist.decode("utf-8",  "ignore").strip():
                    self.logger.warning("./genrem: Permission denied.")
                    while_index = 0
                elif not "Junk in spin" in printlist.decode("utf-8",  "ignore").strip():
                    self.logger.info(printlist.decode("utf-8",  "ignore").strip())
                    if not printlist.strip() == "":
                        printlists.append(printlist.strip().decode('utf-8'))
            p.wait()
            if while_index:
                self.logger.info("finish shell.")
                break
            else:
                self.logger.warning("running the sample again.")
                run_again_num += 1
            if run_again_num >= 10:
                self.logger.warning("running again too much. Stopped.")
                break
        return printlists    
class MDGrow_dynamics(MDGrow_preparation):
    def __init__(self,num,phi,StepMax,samples=6,phi0=0.63):
        super().__init__(num,phi,StepMax,samples,phi0=0.63)
        

    
    def dynamics_preparation(self,):
        shell(Const.gppdynamics(self.num))
        try:
            shutil.copy("./mdgrow",self.dir)
        except:
            self.logger.warning("shutil.copy fails.")
    def dynamics_1(self,i,run=1,delete=0,prepare=0):
        start = time.perf_counter()
        #prepare:
        if prepare:
            self.dynamics_preparation()
            
        #running:   
        shellcont = self.dir + "mdgrow {phi} {samples} {replica} {STEP_MAX} {phi0}"\
            .format(phi=self.phi,samples=self.samples,replica=i,STEP_MAX=self.stepmax,phi0=self.phi0)
        shell(shellcont)
        #delete:

        finish = time.perf_counter()
        self.logger.info(f'Finished dynamics_1({self.num},{self.phi},{self.stepmax},{i}) in {round(finish-start,2)} seconds')   

    def dynamics_mp(self,begin=0,end=0,processes=10,run=1,delete=0):
        start = time.perf_counter()
        
        self.dynamics_preparation()
        filenums = np.linspace(begin,end - 1,end-begin)
        self.logger.debug(filenums)
        argss = [[int(filenum),run,delete] for filenum in filenums]
        MPpool(self.dynamics_1,argss,processes=processes)

        finish = time.perf_counter()
        self.logger.info(f'Finished dynamics_mp({begin},{end},{processes}) in {round(finish-start,2)} seconds')   

class MDGrow_DP(MDGrow_preparation):
    def __init__(self,num,phi,StepMax,samples=6,phi0=0.63):
        super().__init__(num,phi,StepMax,samples,phi0=0.63)
    
    def dynmaics_dp(self,):
        fnames = os.listdir(self.dir_t)
        data = read_data_from_txt(self.dir_t + fnames[0],[0,1,2,3,4,5,6,7,8])
        print_tw = data[0]
        MSD_0_tptw = np.zeros(len(print_tw),dtype="float64")
        MSD_0_tptw_half = np.zeros(len(print_tw),dtype="float64")
        MSD_ab = np.zeros(len(print_tw),dtype="float64")
        MSD_ab_half = np.zeros(len(print_tw),dtype="float64")
        delta_0_tptw = np.zeros(len(print_tw),dtype="float64")
        correlation_0_tptw = np.zeros(len(print_tw),dtype="float64")
        MSD_tw_tptw = np.zeros(len(print_tw),dtype="float64")
        delta_tw_tptw = np.zeros(len(print_tw),dtype="float64")
        correlation_tptw_self = np.zeros(len(print_tw),dtype="float64")
        correlation_tw_tptwPself = np.zeros(len(print_tw),dtype="float64")
        
        spls = len(fnames)
        for fname in fnames:
            data = read_data_from_txt(self.dir_t + fname,[0,1,2,3,4,5,6,7,8,9,10])
            try:
                MSD_0_tptw += np.array(data[1])
                MSD_ab += np.array(data[2])
                MSD_0_tptw_half += np.array(data[9])
                MSD_ab_half += np.array(data[10])
                delta_0_tptw += np.array(data[3])
                correlation_0_tptw += np.array(data[4])
                MSD_tw_tptw += np.array(data[5])
                delta_tw_tptw += np.array(data[6])
                correlation_tptw_self += np.array(data[7])
                correlation_tw_tptwPself += np.array(data[8])
            except:
                spls-=1
                print(fname)
        MSD_0_tptw = MSD_0_tptw/spls
        MSD_0_tptw_half = MSD_0_tptw_half/spls
        MSD_ab = MSD_ab/spls
        MSD_ab_half = MSD_ab_half/spls
        delta_0_tptw = delta_0_tptw/spls
        correlation_0_tptw = correlation_0_tptw/spls
        MSD_tw_tptw = MSD_tw_tptw/spls
        delta_tw_tptw = delta_tw_tptw/spls
        correlation_tptw_self = correlation_tptw_self/spls
        correlation_tw_tptwPself = correlation_tw_tptwPself/spls
        
        dump_data(self.dir + "-t_0ANDtw_N%d_phi=%.2le_samples=%d_replica=%d.dat"%(self.num,self.phi,self.samples,len(fnames)),[print_tw,MSD_0_tptw,MSD_ab,delta_0_tptw,correlation_0_tptw, MSD_0_tptw_half,MSD_ab_half]
        ,des=["#print_tw", "MSD_0_tptw", "MSD_ab","delta_0_tptw", "correlation_0_tptw", "MSD_0_tptw_half","MSD_ab_half"])
        dump_data(self.dir + "-t_twANDtptw_N%d_phi=%.2le_samples=%d_replica=%d.dat"%(self.num,self.phi,self.samples,len(fnames)),[print_tw,MSD_tw_tptw,delta_tw_tptw,correlation_tptw_self,correlation_tw_tptwPself]
        ,des=["#print_tw","MSD_tw_tptw","delta_tw_tptw","correlation_tptw_self","correlation_tw_tptwPself"])

            
class MDGrow(MDGrow_DP,MDGrow_dynamics):
    def __init__(self,num,phi,stepmax,samples,phi0=0.63):
        MDGrow_DP.__init__(self,num,phi,stepmax,samples,phi0)
        MDGrow_dynamics.__init__(self,num,phi,stepmax,samples,phi0)
            
        