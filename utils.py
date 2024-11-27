import os
import numpy as np
# import imageio
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from pylab import *
# from dos import *
import math
from scipy.stats import rv_discrete
from scipy.stats import beta
from scipy import integrate
import random
import multiprocessing as mp
import mpmath as mpm
import subprocess
# from mpmath import *

class bouchaud_ptau:
    """
    correlation(tw,t) is to calculate bouchaud_hyp2f1 correlation func, which is exact.
    """
    def __init__(self,p,taulist):
        self.p = p
        self.tl = taulist

#     def inversel(self)
    def correlation(self,tw, t):
        self.tw = tw
        self.t = t
#         return mpm.quad(self.invertl_pE_times_ptau,[1,inf])
        part_exp = [self.invertl_pE_times_exp(tau) for tau in self.tl]
        print("finish tw=%.2le c=%lf"%(tw,np.dot(np.array(part_exp),np.array(self.p))))
        # return np.dot(np.array(part_exp),np.array(self.p))
        return mpm.fdot(part_exp,self.p)
        
    def pE2(self,E):
        tau = self.tau
#         return 1/mpm.hyp2f1(1, x, 1+x, -1.0/E)/(E*tau+1)
        mid = [E*t/(E*t+1) for t in self.tl]
        # return tau/np.dot(np.array(mid),np.array(self.p))/(E*tau+1)
        return tau/mpm.fdot(mid,self.p)/(E*tau+1)
    def pE(self,E):
        return self.pE2(E)
#     def invertl_pE(self,tau):
#         self.tau = tau
#         return invertlaplace(self.pE,self.tw)
    def invertl_pE_times_exp(self,tau):
        self.tau = tau
        return mpm.invertlaplace(self.pE,self.tw)*mpm.exp(-self.t/tau)

# def Funcbar(x,y,bins):
#     return [],[]
# def Funclogbar(x,y,bins):
#     return [],[]
# def Funcbar_log(x,y,bins):
#     return [],[]
# def Funcsumpdf(x,y,bins):
#     return [],[]
def MP(func,argss):
    start = time.perf_counter()
    Processes = []
    for args in argss:
        p = mp.Process(target=func,args=args)
        p.start()
        Processes.append(p)
    for process in Processes:
        process.join()
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start,2)} seconds')   
def MPpool(func,argss,processes=10):
    start = time.perf_counter()

    pool = mp.Pool(processes = processes)
    # Processes = []
    for args in argss:
        pool.apply_async(func,args=args)
    pool.close()
    pool.join()

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start,2)} seconds')

def MPpools(func_argss_list,processes=10):
    """
    func_argss_list = [func_argss1,func_argss2,……]
    func_argss = [func,argss]
    argss = [args1,args2,……]
    args = [para1,para2,……]

    func_argss_list=[[func,[[]]],
                    []]
    """
    start = time.perf_counter()

    pool = mp.Pool(processes = processes)
    # Processes = []
    for func_argss in func_argss_list:
        func = func_argss[0]
        argss = func_argss[1]
        for args in argss:
            pool.apply_async(func,args=args)
    pool.close()
    pool.join()

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start,2)} seconds')

def judge(s,j):
    if not j:
        s = ""
    return s

# def cmd(cmd):
#     p0 = subprocess.Popen(cmd,stdshell=True, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)  
#     for printlist in iter(p.stdout.readline, b''):
#         print(printlist.decode("utf-8",  "ignore").strip(),flush=True)
#         if not printlist.strip() == "":
#             printlists.append(printlist.strip().decode('utf-8'))
#     p0.wait()
#     print("finish cmd")
#     return printlists

def shell(cmd):
    printlists = []
    p = subprocess.Popen(cmd,shell=True, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)  
    for printlist in iter(p.stdout.readline, b''):
        if not "Junk in spin" in printlist.decode("utf-8",  "ignore").strip():
            print(printlist.decode("utf-8",  "ignore").strip())
            if not printlist.strip() == "":
                printlists.append(printlist.strip().decode('utf-8'))
    p.wait()
    return printlists

def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)

    if not isExists:
        os.makedirs(path) 
        # return True

    return path

def read_data_from_txt(path,indexs=[0,1],row=[0,0]):#input null lists [[],[],……]
    if indexs:
        lists = [[] for index in indexs]
        if row[1] == 0:
            row[1] = 1e15
        j=0
        with open(path,'r') as fp:
            linedata = fp.readline()
            if "#" in linedata[:1] or "//" in linedata[:1]:
                linedata = fp.readline() 
            while 1:
                if linedata.strip()=="":
                    break
                if j >= row[1]:
                    break
                if j >= row[0]:
                    data = linedata.split()
                    for i in range(len(lists)):
                        index = indexs[i]
                        lists[i].append(float(data[index]))
                linedata = fp.readline()
                
    else:
        lists = []
        print("index is null read_data_from_txt.")
    # print(lists)
    # return tuple(lists)
    return lists

def rd(path,indexs=[0,1],row=[0,0]):#input null lists [[],[],……]
    if indexs:
        lists = [[] for index in indexs]
        if row[1] == 0:
            row[1] = 1e15
        j=0
        with open(path,'r') as fp:
            linedata = fp.readline()
            if "#" in linedata[:1] or "//" in linedata[:1]:
                linedata = fp.readline()     
            while 1:
                if linedata.strip()=="":
                    break
                if j >= row[1]:
                    break
                if j >= row[0]:
                    data = linedata.split()
                    for i in range(len(lists)):
                        index = indexs[i]
                        lists[i].append(data[index])
                linedata = fp.readline()
                
    else:
        lists = []
        print("index is null read_data_from_txt.")
    # print(lists)
    # return tuple(lists)
    return lists

# def rd(path,indexs=[0,1],row=[0,0]):#input null lists [[],[],……]
#     if indexs:
#         if row[1] == 0:
#             row[1] = len(data)
#         with open(path,'r') as fp:
#             data = fp.read().splitlines()
#             lists = [[float(data[i][index]) for i in range(row[0],row[1])] for index in indexs]
#     else:
#         lists = []
#         print("index is null read_data_from_txt.")
#     # print(lists)
#     # return tuple(lists)
#     return lists

def dump_data(path,lists,fmt="%le",des=""):  #()
    try:
        check = lists.any()
    except:
        check = lists
    if not check:
        print("lists is null when dumping data.") 
    else: 
        with open(path,'w+',encoding='utf-8') as fp:
            if isinstance(des,str):
                fp.write(des+"\n")
            if isinstance(des,list):
                for ele in des:
                    fp.write(ele+" ")
                fp.write(" \n")
            for i in range(len(lists[0])):
                for li in lists:
                    fp.write(fmt%li[i]+" ")
                fp.write("\n")


def doslist_del_0headtail(dos,bins):
    n_tail = 0
    n_head = 0
    for k in dos [::-1]:
        if k == 0:
            n_tail += 1
        else:
            break
    for k in dos:
        if k == 0:
            n_head += 1
        else:
            break
    if n_tail:
        return dos[n_head:(-n_tail)],bins[n_head:(-n_tail)]
    else:
        return dos[n_head:],bins[n_head:]


def doslist_mod0pdf(dos,bins):  
    dos,bins = doslist_del_0headtail(dos,bins)
    print(dos)
    print(bins)
    dosout = []
    binsout = [bins[0]]

    for i in range(1,len(dos)):
        if dos[i] == 0 and dos[i-1] > 0:# 1->0
            wsum = bins[i+1] - bins[i-1]
            psum = dos[i-1]*(bins[i] - bins[i-1])
            j = i - 1
        elif dos[i]==0 and dos[i-1]==0: # 0->0
            wsum += bins[i+1] - bins[i]
        elif dos[i]>0 and dos[i-1]==0: # 0->1
            dosout.append(psum/wsum)  #1000 1
            binsout.append(bins[i])
        else:
            dosout.append(dos[i-1])
            binsout.append(bins[i])
    dosout.append(dos[-1])
    binsout.append(bins[-1])
    lspout = [(binsout[i]+binsout[i+1])/2 for i in range(len(binsout) - 1)]
    widthout = [(-binsout[i]+binsout[i+1]) for i in range(len(binsout) - 1)]

    return lspout,dosout,binsout,widthout

def gen_doslist_mod0pdf(lists,bin_num,mm=[]):
    if mm:
        Min = mm[0]
        Max = mm[1]
    elif not mm:
        Min = min(lists)
        Max = max(lists)
    bins = np.linspace(Min,Max,bin_num + 1)
    doslist, b, c = plt.hist(lists,bins=bins,density=1)
    doslist, b = doslist_mod0pdf(doslist, b)
    plt.clf()
    lsplist = [(b[i]+b[i+1])/2 for i in range(len(b) - 1)]
    return lsplist, doslist

def gen_dosbinlist(lists,bin_num,mm=[]):
    if mm:
        Min = mm[0]
        Max = mm[1]
    elif not mm:
        Min = min(lists)
        Max = max(lists)
    bins = np.linspace(Min,Max,bin_num + 1)
    doslist, b, c = plt.hist(lists,bins=bins,density=1)
    plt.clf()
    lsplist = [(b[i]+b[i+1])/2 for i in range(len(b) - 1)]
    return  lsplist,doslist,b

def gen_doslist(lists,bin_num,mm=[]):
    if mm:
        Min = mm[0]
        Max = mm[1]
    elif not mm:
        Min = min(lists)
        Max = max(lists)
    bins = np.linspace(Min,Max,bin_num + 1)
    doslist, b, c = plt.hist(lists,bins=bins,density=1)
    plt.clf()
    lsplist = [(b[i]+b[i+1])/2 for i in range(len(b) - 1)]
    return lsplist, doslist

def gen_doslist_log(lists,bin_num):
    lists2 = [ele for ele in lists if ele>0]
    bins = np.logspace(np.log10(min(lists2)), np.log10(max(lists2)),bin_num + 1)
    doslist, b, c = plt.hist(lists2,bins=bins,density=1)
    plt.clf()
    del lists2 
    lsplist = [10**( (np.log10(b[i]) + np.log10(b[i+1]))/2 )  for i in range(len(b) - 1)]
    return lsplist, doslist

def gen_doslist_mixed(lists,bins_trct,bins_num_log):
    lists2 = [ele for ele in lists if ele>1]
    # lists2 = lists
    bins_line = np.linspace(2.0, bins_trct - 1, bins_trct - 2)
    bins_log = np.logspace(np.log10(bins_trct), np.log10(max(lists2)), bins_num_log + 1)
    bins = np.hstack([bins_line,bins_log])
    # print(bins)
    doslist, b, c = plt.hist(lists2,bins=bins,density=1)
    plt.clf()
    del lists2
    lsplist_line = bins_line
    lsplist_log = np.array([np.sqrt(bins_log[i]*bins_log[i+1]) for i in range(bins_num_log)])
    lsplist = np.hstack([lsplist_line,lsplist_log])
    widthlist_line = np.array([1.0 for i in range(bins_trct-2)])
    widthlist_log = np.array([bins_log[i+1]-bins_log[i] for i in range(bins_num_log)])
    widthlist = np.hstack([widthlist_line,widthlist_log])
    print(widthlist)
    # print(lsplist)
    print(lsplist[0],doslist[0])
    return lsplist, doslist, widthlist

def gen_doslist_mixed_linearFine(lists,bins_trct,bins_num_log,Print=1,mm=[]):
    lists2 = [ele for ele in lists if ele>0]
    # lists2 = lists
    if mm:
        Min = mm[0]
        Max = mm[1]
    elif not mm:
        Min = 1
        Max = max(lists2)
    bins_line = np.linspace(int(Min), bins_trct - 1, bins_trct - int(Min))
    bins_log = np.logspace(np.log10(bins_trct), np.log10(Max), bins_num_log + 1)
    bins = np.hstack([bins_line,bins_log])
    # print(bins)
    doslist, b, c = plt.hist(lists2,bins=bins,density=1)
    plt.clf()
    del lists2
    lsplist_line = bins_line
    lsplist_log = np.array([np.sqrt(bins_log[i]*bins_log[i+1]) for i in range(bins_num_log)])
    lsplist = np.hstack([lsplist_line,lsplist_log])
    widthlist_line = np.array([1.0 for i in range(bins_trct-int(Min))])
    widthlist_log = np.array([bins_log[i+1]-bins_log[i] for i in range(bins_num_log)])
    widthlist = np.hstack([widthlist_line,widthlist_log])
    if Print:
        print(widthlist)
        # print(lsplist)
        print(lsplist[0],doslist[0])
    return lsplist, doslist, widthlist

def gen_doslist2d_mixed_linearFine(lists1,lists2,bins_trct,bins_num_log,Print=1,mm=[]):
    # lists2 = [ele for ele in lists if ele>0]
    # lists2 = lists
    if mm:
        Min = mm[0]
        Max = mm[1]
    elif not mm:
        Min = 1
        Max = max(lists2)
    bins_line = np.linspace(int(Min), bins_trct - 1, bins_trct - int(Min))
    bins_log = np.logspace(np.log10(bins_trct), np.log10(Max), bins_num_log + 1)
    bins = np.hstack([bins_line,bins_log])
    # print(bins)
    doslist, b, c = plt.hist(lists2,bins=bins,density=1)
    prob_tau12,x,y,m = hist2d(lists1, lists2, bins=(bins,bins), norm=LogNorm(),label="pdf2d_1by1",density=1)

    plt.clf()
    del lists2
    del lists1
    lsplist_line = bins_line
    lsplist_log = np.array([np.sqrt(bins_log[i]*bins_log[i+1]) for i in range(bins_num_log)])
    lsplist = np.hstack([lsplist_line,lsplist_log])
    widthlist_line = np.array([1.0 for i in range(bins_trct-int(Min))])
    widthlist_log = np.array([bins_log[i+1]-bins_log[i] for i in range(bins_num_log)])
    widthlist = np.hstack([widthlist_line,widthlist_log])
    if Print:
        print(widthlist)
        # print(lsplist)
        print(lsplist[0],doslist[0])
    return lsplist, prob_tau12, widthlist

def plot(xys,labels,xlabel="",ylabel="",yerrs=[],func=str):# xys is xy tuple here; use "varname" package to be more automatical
    if not xlabel+ylabel:
        title = ""
        img_path = ""
    else:
        title = ylabel +"-"+ xlabel
        img_path = ylabel +"-"+ xlabel
    for label in labels:
        label.replace(" ","_")
        title = title + "_" + label
        img_path = img_path + "_" + label
    if title[0] == "_":
        title = title[1:]
    if img_path[0] == "_":
        img_path = img_path[1:]
    img_path = func(img_path)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    for xy in xys:
        idx = xys.index(xy)
        plt.plot(xy[0],xy[1],label=labels[idx])
    plt.grid(alpha=0.3)
    plt.legend()
    if yerrs:
        for yerr in yerrs:
            plt.errorbar(xy[0],xy[1],yerr=yerr,fmt='o',ecolor='r',color='b',elinewidth=1,capsize=1,ms=2)
    plt.savefig(img_path[0])
    plt.savefig(img_path[1])
    plt.clf()


def plot_ylog(xys,labels,xlabel="",ylabel="",yerrs=[],func=str):
    if not xlabel+ylabel:
        title = ""
        img_path = "" + "_ylog"
    else:
        title = ylabel +"-"+ xlabel
        img_path = ylabel +"-"+ xlabel + "_ylog"
    for label in labels:
        label.replace(" ","_")
        title = title + "_" + label
        img_path = img_path + "_" + label
    if title[0] == "_":
        title = title[1:]
    if img_path[0] == "_":
        img_path = img_path[1:]
    img_path = func(img_path)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale("log")
    plt.title(title)
    for xy in xys:
        idx = xys.index(xy)
        plt.plot(xy[0],xy[1],label=labels[idx])
    plt.grid(alpha=0.3)
    plt.legend()
    if yerrs:
        for yerr in yerrs:
            plt.errorbar(xy[0],xy[1],yerr=yerr,fmt='o',ecolor='r',color='b',elinewidth=1,capsize=1,ms=2)
    plt.savefig(img_path[0])
    plt.savefig(img_path[1])
    plt.clf()

def plot_xlog(xys,labels,xlabel="",ylabel="",yerrs=[],func=str):
    if not xlabel+ylabel:
        title = ""
        img_path = "" + "_xlog"
    else:
        title = ylabel +"-"+ xlabel
        img_path = ylabel +"-"+ xlabel + "_xlog"
    for label in labels:
        label.replace(" ","_")
        title = title + "_" + label
        img_path = img_path + "_" + label
    if title[0] == "_":
        title = title[1:]
    if img_path[0] == "_":
        img_path = img_path[1:]
    img_path = func(img_path)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale("log")
    plt.title(title)
    for xy in xys:
        idx = xys.index(xy)
        plt.plot(xy[0],xy[1],label=labels[idx])
    plt.grid(alpha=0.3)
    plt.legend()
    if yerrs:
        for yerr in yerrs:
            plt.errorbar(xy[0],xy[1],yerr=yerr,fmt='o',ecolor='r',color='b',elinewidth=1,capsize=1,ms=2)
    
    plt.savefig(img_path[0])
    plt.savefig(img_path[1])
    plt.clf()

def plot_xylog(xys,labels,xlabel="",ylabel="",yerrs=[],func=str):
    if not xlabel+ylabel:
        title = ""
        img_path = "" + "_xylog"
    else:
        title = ylabel +"-"+ xlabel
        img_path = ylabel +"-"+ xlabel + "_xylog"
    for label in labels:
        label.replace(" ","_")
    title = title + "_" + labels[0]
    img_path = img_path + "_" + labels[0]
    if title[0] == "_":
        title = title[1:]
    if img_path[0] == "_":
        img_path = img_path[1:]
    img_path = func(img_path)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale("log")
    plt.yscale("log")
    plt.title(title)
    # for xy in xys:
    #     idx = xys.index(xy)
    #     plt.plot(xy[0],xy[1],label=labels[idx])
    for i in range(len(xys)):
        plt.plot(xys[i][0],xys[i][1],label=labels[i])

    plt.grid(alpha=0.3)
    plt.legend()
    if yerrs:
        for yerr in yerrs:
            plt.errorbar(xy[0],xy[1],yerr=yerr,fmt='o',ecolor='r',color='b',elinewidth=1,capsize=1,ms=2)
    plt.savefig(img_path[0])
    plt.savefig(img_path[1])
    plt.clf()

def scatter(xys,labels,xlabel="",ylabel="",func=str):
    if not xlabel+ylabel:
        title = ""
        img_path = ""
    else:
        title = ylabel +"-"+ xlabel
        img_path = ylabel +"-"+ xlabel
    for label in labels:
        label.replace(" ","_")
        title = title + "_" + label
        img_path = img_path + "_" + label
    if title[0] == "_":
        title = title[1:]
    if img_path[0] == "_":
        img_path = img_path[1:]
    img_path = func(img_path)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    for xy in xys:
        idx = xys.index(xy)
        plt.scatter(xy[0],xy[1],label=labels[idx],s=8)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(img_path[0])
    plt.savefig(img_path[1])
    plt.clf()

def scatter_xylog(xys,labels,xlabel="",ylabel="",z=[],func=str):
    if not xlabel+ylabel:
        title = ""
        img_path = "" + "_xylog"
    else:
        title = ylabel +"-"+ xlabel
        img_path = ylabel +"-"+ xlabel + "_xylog"
    for label in labels:
        label.replace(" ","_")
        title = title + "_" + label
        img_path = img_path + "_" + label
    if title[0] == "_":
        title = title[1:]
    if img_path[0] == "_":
        img_path = img_path[1:]
    img_path = func(img_path)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale("log")
    plt.yscale("log")
    plt.title(title)
    try:
        check = z.any()
    except:
        check = z
    if not check:
        for xy in xys:
            idx = xys.index(xy)
            plt.scatter(xy[0],xy[1],label=labels[idx],s=8)
    else:
        for xy in xys:
            idx = xys.index(xy)
            plt.scatter(xy[0],xy[1],label=labels[idx],c=z,s=8)
        colorbar()
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(img_path[0])
    plt.savefig(img_path[1])
    plt.clf()

def scatter_xylog2(xys,labels,xlabel="",ylabel="",z=[],zlabel="pdf",normrange=[],func=str):
    if not xlabel+ylabel:
        title = ""
        img_path = "" + "_xylog"
    else:
        title = ylabel +"-"+ xlabel
        img_path = ylabel +"-"+ xlabel + "_xylog"
    for label in labels:
        label.replace(" ","_")
        title = title + "_" + label
        img_path = img_path + "_" + label
    if title[0] == "_":
        title = title[1:]
    if img_path[0] == "_":
        img_path = img_path[1:]
    img_path = func(img_path)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale("log")
    plt.yscale("log")
    plt.title(title)
    try:
        check = z.any()
    except:
        check = z
    if not check:
        for xy in xys:
            idx = xys.index(xy)
            plt.scatter(xy[0],xy[1],label=labels[idx],marker=",",s=16)
    else:
        for xy in xys:
            x = xy[0]
            y = xy[1]
            idx = xys.index(xy)
            f=plt.figure()
            ax0 = plt.subplot2grid((1,1),(0,0),colspan=1)
            ax0.set_xscale("log")
            ax0.set_yscale("log")
            ax0.set_xlabel("tau1")
            ax0.set_ylabel("tau2")
            if normrange:
                norm = mpl.colors.LogNorm(normrange[0],normrange[1])
                sc = ax0.scatter(x,y,c=z,norm=norm,label=labels[idx],marker=",",s=64)
            else:
                sc = ax0.scatter(x,y,c=z,label=labels[idx],marker=",",s=64)
            cbar = f.colorbar(sc)
            cbar.set_label(zlabel,fontsize=15)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(img_path[0])
    plt.savefig(img_path[1])
    plt.clf()

def plot_hist(lists,labels,xlabel="",ylabel="",rdn=0,func=str):# xys is xy tuple here; use "varname" package to be more automatical
    if not xlabel+ylabel:
        title = ""
        img_path = ""
    else:
        title = ylabel +"-"+ xlabel
        img_path = ylabel +"-"+ xlabel
    for label in labels:
        label.replace(" ","_")
        title = title + "_" + label
        img_path = img_path + "_" + label
    if title[0] == "_":
        title = title[1:]
    if img_path[0] == "_":
        img_path = img_path[1:]
    data = [rdn,img_path]
    img_path = func(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    for l in lists:
        idx = lists.index(l)
        plt.hist(l,label=labels[idx])
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(img_path[0])
    plt.savefig(img_path[1])
    plt.clf()


def plot_hist_ylog(lists,labels,xlabel="",ylabel="",rdn=0,func=str):
    if not xlabel+ylabel:
        title = ""
        img_path = "" + "_ylog"
    else:
        title = ylabel +"-"+ xlabel
        img_path = ylabel +"-"+ xlabel + "_ylog"
    for label in labels:
        label.replace(" ","_")
        title = title + "_" + label
        img_path = img_path + "_" + label
    if title[0] == "_":
        title = title[1:]
    if img_path[0] == "_":
        img_path = img_path[1:]
    data = [rdn,img_path]
    img_path = func(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale("log")
    plt.title(title)
    for l in lists:
        idx = lists.index(l)
        plt.hist(l,label=labels[idx])
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(img_path[0])
    plt.savefig(img_path[1])
    plt.clf()

def plot_hist2d(x,y,label,xlabel="",ylabel="",func=str,bins=100):
    if not xlabel+ylabel:
        title = ""
        img_path = "" 
    else:
        title = ylabel +"-"+ xlabel
        img_path = ylabel +"-"+ xlabel
    label.replace(" ","_")
    title = title + "_" + label
    img_path = img_path + "_" + label
    if title[0] == "_":
        title = title[1:]
    if img_path[0] == "_":
        img_path = img_path[1:]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    img_path = func(img_path)
    xbins = np.linspace(min(x),max(x),bins)
    ybins = np.linspace(min(y),max(y),bins)
    prob,x,y,m = hist2d(x, y, bins=(xbins,ybins), norm=LogNorm(),label=label,density=1)

    colorbar()
    plt.legend()
    plt.savefig(img_path[0])
    plt.savefig(img_path[1])
    plt.clf()

    return prob,x,y,m

def plot_hist2d_ylog(x,y,label,xlabel="",ylabel="",func=str,bins=100):
    if not xlabel+ylabel:
        title = ""
        img_path = "" 
    else:
        title = ylabel +"-"+ xlabel
        img_path = ylabel +"-"+ xlabel
    label.replace(" ","_")
    title = title + "_" + label
    img_path = img_path + "_" + label
    if title[0] == "_":
        title = title[1:]
    if img_path[0] == "_":
        img_path = img_path[1:]
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    img_path = func(img_path)
    xbins = np.linspace(min(x),max(x),bins)
    ybins = np.logspace(np.log10(min(y)),np.log10(max(y)),bins)
    prob,x,y,m = hist2d(x, y, bins=(xbins,ybins), norm=LogNorm(),label=label,density=1)

    colorbar()
    plt.legend()
    plt.savefig(img_path[0])
    plt.savefig(img_path[1])
    plt.clf()

    return prob,x,y,m

def plot_hist2d_xylog(x,y,label,xlabel="",ylabel="",func=str,bins=100):
    if not xlabel+ylabel:
        title = ""
        img_path = "" + "_xylog"
    else:
        title = ylabel +"-"+ xlabel
        img_path = ylabel +"-"+ xlabel + "_xylog"
    label.replace(" ","_")
    title = title + "_" + label
    img_path = img_path + "_" + label
    if title[0] == "_":
        title = title[1:]
    if img_path[0] == "_":
        img_path = img_path[1:]
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    img_path = func(img_path)
    xbins = np.logspace(np.log10(min(x)),np.log10(max(x)),bins)
    ybins = np.logspace(np.log10(min(y)),np.log10(max(y)),bins)
    prob,x,y,m = hist2d(x, y, bins=(xbins,ybins), norm=LogNorm(),label=label,density=1)

    colorbar()
    plt.legend()
    plt.savefig(img_path[0])
    plt.savefig(img_path[1])
    plt.clf()

    return prob,x,y,m

def list_d(l,number):
    return (np.array(l)/number).tolist()

def get_var_lists(lists,CI=2):
    """
    return a var list
    """
    res = []
    for l in lists:
        res.append(np.var(l)/np.sqrt(len(l)-1)*CI)
    return res
class REM:
    def __init__(self,num,temperature,TIME_MAX,betac,bins=200,minh=1e-5) -> None:
        self.num = num
        self.temp = temperature
        self.temperature = temperature
        self.timemax = TIME_MAX
        self.betac = betac
        self.alpha = temperature * betac
        self.bins = bins
        self.minh = minh
        strminh = "minh={}/".format("%.2le"%minh)
        self.dir = "../data/N={dd}_T={cc}_{ee}/{strminh}".format(cc="%.2f"%self.temp,dd=self.num,ee="%.0le"%self.timemax,strminh=strminh)
        self.dirObs = "../data/N={dd}_T={cc}_{ee}/{strminh}obs-t/".format(cc="%.2f"%self.temp,dd=self.num,ee="%.0le"%self.timemax,strminh=strminh)
        self.dirCorr = "../data/N={dd}_T={cc}_{ee}/{strminh}correlation-t/".format(cc="%.2f"%self.temp,dd=self.num,ee="%.0le"%self.timemax,strminh=strminh)
        self.dirCorrRate = "../data/N={dd}_T={cc}_{ee}/{strminh}correlation-tdtw_rate=".format(cc="%.2f"%self.temp,dd=self.num,ee="%.0le"%self.timemax,strminh=strminh)
        self.dirCorrConfRate = "../data/N={dd}_T={cc}_{ee}/{strminh}correlation-tdtw_conf_rate=".format(cc="%.2f"%self.temp,dd=self.num,ee="%.0le"%self.timemax,strminh=strminh)
        self.dirCorrRateTau0 = "../data/N={dd}_T={cc}_{ee}/{strminh}correlation-logtdlogtw_".format(cc="%.2f"%self.temp,dd=self.num,ee="%.0le"%self.timemax,strminh=strminh)
        self.dirCorrConfRateTau0 = "../data/N={dd}_T={cc}_{ee}/{strminh}correlation-logtdlogtw_conf_".format(cc="%.2f"%self.temp,dd=self.num,ee="%.0le"%self.timemax,strminh=strminh)
        self.dirTauConf = "../data/N={dd}_T={cc}_{ee}/{strminh}tau-t_conf/".format(cc="%.2f"%self.temp,dd=self.num,ee="%.0le"%self.timemax,strminh=strminh)
        self.dirE = "../data/N={dd}_T={cc}_{ee}/{strminh}E-t/".format(cc="%.2f"%self.temp,dd=self.num,ee="%.0le"%self.timemax,strminh=strminh)
        self.dirRealBarrier = "../data/N={dd}_T={cc}_{ee}/{strminh}barrier/".format(cc="%.2f"%self.temp,dd=self.num,ee="%.0le"%self.timemax,strminh=strminh)
        self.dirbarrier = "../data/N={dd}_{strminh}".format(cc="%.2f"%self.temp,dd=self.num,ee="%.0le"%self.timemax,strminh=strminh)
        self.dirBarrier = "../data/N={dd}_{strminh}lowest_barriers/".format(cc="%.2f"%self.temp,dd=self.num,ee="%.0le"%self.timemax,strminh=strminh,temp="%.2lf"%temperature)
        self.dirBarrierMinDistance = "../data/N={dd}_{strminh}barriers_minDistancePairs/".format(cc="%.2f"%self.temp,dd=self.num,ee="%.0le"%self.timemax,strminh=strminh,temp="%.2lf"%temperature)
        self.dirBarrier = "../data/N={dd}_{strminh}lowest_barriers/T={temp}/".format(cc="%.2f"%self.temp,dd=self.num,ee="%.0le"%self.timemax,strminh=strminh,temp="%.2lf"%temperature)
        self.dirTipsBarriers = "../data/N={dd}_{strminh}all_tips_barriers/".format(cc="%.2f"%self.temp,dd=self.num,ee="%.0le"%self.timemax,strminh=strminh,temp="%.2lf"%temperature)
        self.dirActivation = "../data/N={dd}_T={cc}_{ee}/{strminh}activation-t/".format(cc="%.2f"%self.temp,dd=self.num,ee="%.0le"%self.timemax,strminh=strminh)
        self.dirActivationSimple = "../data/N={dd}_T={cc}_{ee}/{strminh}activation-t_simple/".format(cc="%.2f"%self.temp,dd=self.num,ee="%.0le"%self.timemax,strminh=strminh)
        self.dirActivationEb = "../data/N={dd}_T={cc}_{ee}/{strminh}activation-t_Eb/".format(cc="%.2f"%self.temp,dd=self.num,ee="%.0le"%self.timemax,strminh=strminh)
        self.dirNextBasin = "../data/N={dd}_T={cc}_{ee}/{strminh}nextBasin/".format(cc="%.2f"%self.temp,dd=self.num,ee="%.0le"%self.timemax,strminh=strminh)
        mkdir(self.dir +"plot/")
        mkdir(self.dir +"plot/pdf")
        mkdir(self.dirbarrier +"plot/")
        mkdir(self.dirbarrier +"plot/pdf")

#corr-tdtw:
    def read_correlation_tdtw(self,rate,fnameNum=0):
        dirCorrRate = self.dirCorrRate + "{}/".format("%.2lf"%rate)
        fnames = os.listdir(dirCorrRate)
        if fnameNum:
            fnames = fnames[:fnameNum]
        #fnames = fnames[:-24]
        lenfnames = len(fnames)
        self.timelist = read_data_from_txt(dirCorrRate + fnames[0], [0])[0]
        self.corrlist = np.array([0.0 for time in self.timelist])
        for fname in fnames:
            corr1spl = read_data_from_txt(dirCorrRate + fname, [1])[0]
            if not corr1spl == []:
                try:
                    self.corrlist = self.corrlist + np.array(corr1spl)
                except:
                    print(fname)
            else:
                lenfnames = lenfnames - 1
        self.corrlist = self.corrlist/lenfnames
        outputName = 'correlation-tdtw_rem{num}_T={temp}_timemax={tm}_rate={rate}_splnum={spl}.t'.format(
                num = self.num, temp="%.2lf"%self.temp,
                tm="%.0le"%self.timemax, rate=rate,spl="%.1le"%len(fnames))
        # mkdir(dirtdtw)
        # print(dirtdtw + outputName)
        # self.corrlist = self.corrlist.tolist()
        dump_data(self.dir + outputName, [self.timelist,self.corrlist])
    def plot_correlation_tdtw(self,rate,fnameNum=0):
        self.read_correlation_tdtw(rate,fnameNum)
        # mkdir(self.dirCorr + "plot/")
        mkdir(self.dir + "plot/")
        time = self.timelist
        correlation = self.corrlist
        plot([(time,correlation)]
            ,[f'Correlation-tdtw rate={rate} splnum={fnameNum}']
            ,func=self.__imgpath)
        plot_xlog([(time,correlation)]
            ,[f'Correlation-tdtw rate={rate} splnum={fnameNum}']
            ,func=self.__imgpath)
        plot_xylog([(time,correlation)]
            ,[f'Correlation-tdtw rate={rate} splnum={fnameNum}']
            ,func=self.__imgpath)

#corr-logtdlogtw:
    def read_correlation_logtdlogtw(self,rate,tau0,fnameNum=0):
        dirCorrRateTau0 = self.dirCorrRateTau0 + "rate={rate}_tau0={tau0}/".format(rate="%.2lf"%rate,tau0="%.2lf"%tau0)
        fnames = os.listdir(dirCorrRateTau0)
        if fnameNum:
            fnames = fnames[:fnameNum]
        lenfnames = len(fnames)
        #fnames = fnames[:-24]
        self.timelist = read_data_from_txt(dirCorrRateTau0 + fnames[0], [0])[0]
        self.corrlist = np.array([0.0 for time in self.timelist])
        for fname in fnames:
            corr1spl = read_data_from_txt(dirCorrRateTau0 + fname, [1])[0]
            if not corr1spl == []:
                self.corrlist = self.corrlist + np.array(corr1spl)
            else:
                lenfnames = lenfnames - 1
        self.corrlist = self.corrlist/lenfnames
        # dirtdtw = self.dirCorr + "corr-tdtw/"
        outputName = 'correlation-logtdlogtw_rem{num}_T={temp}_timemax={tm}_rate={rate}_tau0={tau0}_splnum={spl}.t'.format(
                num = self.num, temp="%.2lf"%self.temp, tau0="%.2lf"%tau0,
                tm="%.0le"%self.timemax, rate="%.2lf"%rate,spl="%.1le"%len(fnames))
        # mkdir(dirtdtw)
        # print(dirtdtw + outputName)
        # self.corrlist = self.corrlist.tolist()
        dump_data(self.dir + outputName, [self.timelist,self.corrlist])
    def plot_correlation_logtdlogtw(self,rate,tau0,fnameNum=0):
        self.read_correlation_logtdlogtw(rate,tau0,fnameNum)
        # mkdir(self.dirCorr + "plot/")
        mkdir(self.dir + "plot/")
        time = self.timelist
        correlation = self.corrlist
        plot([(time,correlation)]
            ,[f'Correlation-t rate={rate} tau0={tau0} splnum={fnameNum}']
            ,func=self.__imgpath)
        plot_xlog([(time,correlation)]
            ,[f'Correlation-t rate={rate} tau0={tau0} splnum={fnameNum}']
            ,func=self.__imgpath)
        plot_xylog([(time,correlation)]
            ,[f'Correlation-t rate={rate} tau0={tau0} splnum={fnameNum}']
            ,func=self.__imgpath)

#corr-tdtw_conf:
    def read_correlation_tdtw_conf(self,rate,fnameNum=0):
        dirCorrConfRate = self.dirCorrConfRate + "{}/".format("%.2lf"%rate)
        fnames = os.listdir(dirCorrConfRate)
        if fnameNum:
            fnames = fnames[:fnameNum]
        lenfnames = len(fnames)
        #fnames = fnames[:-24]
        self.timelist = read_data_from_txt(dirCorrConfRate + fnames[0], [0])[0]
        self.corrlist = np.array([0.0 for time in self.timelist])
        for fname in fnames:
            corr1spl = read_data_from_txt(dirCorrConfRate + fname, [1])[0]
            if not corr1spl == []:
                self.corrlist = self.corrlist + np.array(corr1spl)
            else:
                lenfnames = lenfnames - 1
        self.corrlist = self.corrlist/lenfnames
        # dirtdtw = self.dirCorr + "corr-tdtw/"
        outputName = 'correlation-tdtw_conf_rem{num}_T={temp}_timemax={tm}_rate={rate}_splnum={spl}.t'.format(
                num = self.num, temp="%.2lf"%self.temp,
                tm="%.0le"%self.timemax, rate=rate,spl="%.1le"%len(fnames))
        # mkdir(dirtdtw)
        # print(dirtdtw + outputName)
        # self.corrlist = self.corrlist.tolist()
        dump_data(self.dir + outputName, [self.timelist,self.corrlist])
    def plot_correlation_tdtw_conf(self,rate,fnameNum=0):
        self.read_correlation_tdtw_conf(rate,fnameNum)
        # mkdir(self.dirCorr + "plot/")
        mkdir(self.dir + "plot/")
        time = self.timelist
        correlation = self.corrlist
        plot([(time,correlation)]
            ,[f'Correlation-t_conf rate={rate}']
            ,func=self.__imgpath)
        plot_xlog([(time,correlation)]
            ,[f'Correlation-t_conf rate={rate}']
            ,func=self.__imgpath)
        plot_xylog([(time,correlation)]
            ,[f'Correlation-t_conf rate={rate}']
            ,func=self.__imgpath)

#corr-logtdlogtw_conf:
    def read_correlation_logtdlogtw_conf(self,rate,tau0,fnameNum=0):
        dirCorrConfRateTau0 = self.dirCorrConfRateTau0 + "rate={rate}_tau0={tau0}/".format(rate="%.2lf"%rate,tau0="%.2lf"%tau0)
        fnames = os.listdir(dirCorrConfRateTau0)
        if fnameNum:
            fnames = fnames[:fnameNum]
        lenfnames = len(fnames)
        #fnames = fnames[:-24]
        self.timelist = read_data_from_txt(dirCorrConfRateTau0 + fnames[0], [0])[0]
        self.corrlist = np.array([0.0 for time in self.timelist])
        for fname in fnames:
            corr1spl = read_data_from_txt(dirCorrConfRateTau0 + fname, [1])[0]
            if not corr1spl == []:
                self.corrlist = self.corrlist + np.array(corr1spl)
            else:
                lenfnames = lenfnames - 1
        self.corrlist = self.corrlist/lenfnames
        # dirtdtw = self.dirCorr + "corr-tdtw/"
        outputName = 'correlation-logtdlogtw_conf_rem{num}_T={temp}_timemax={tm}_rate={rate}_tau0={tau0}_splnum={spl}.t'.format(
                num = self.num, temp="%.2lf"%self.temp, tau0="%.2lf"%tau0,
                tm="%.0le"%self.timemax, rate="%.2lf"%rate,spl="%.1le"%len(fnames))
        # mkdir(dirtdtw)
        # print(dirtdtw + outputName)
        # self.corrlist = self.corrlist.tolist()
        dump_data(self.dir + outputName, [self.timelist,self.corrlist])
    def plot_correlation_logtdlogtw_conf(self,rate,tau0,fnameNum=0):
        self.read_correlation_logtdlogtw_conf(rate,tau0,fnameNum)
        # mkdir(self.dirCorr + "plot/")
        mkdir(self.dir + "plot/")
        time = self.timelist
        correlation = self.corrlist
        plot([(time,correlation)]
            ,[f'Correlation-t_conf rate={rate} tau0={tau0} splnum={fnameNum}']
            ,func=self.__imgpath)
        plot_xlog([(time,correlation)]
            ,[f'Correlation-t_conf rate={rate} tau0={tau0} splnum={fnameNum}']
            ,func=self.__imgpath)
        plot_xylog([(time,correlation)]
            ,[f'Correlation-t_conf rate={rate} tau0={tau0} splnum={fnameNum}']
            ,func=self.__imgpath)

#tau-t_conf:
    def read_tau_t_conf(self,fnameNum=0):
        fnames = os.listdir(self.dirTauConf)
        if fnameNum:
            fnames = fnames[:fnameNum]
        lenfnames = len(fnames)
        self.timelist = read_data_from_txt(self.dirTauConf + fnames[0], [0])[0]
        self.taulist = np.array([0.0 for time in self.timelist])
        for fname in fnames:
            tau1spl = read_data_from_txt(self.dirTauConf + fname, [1])[0]
            if not tau1spl == []:
                self.taulist = self.taulist + np.array(abs(np.array(tau1spl)))
            else:
                lenfnames = lenfnames - 1
        self.taulist = self.taulist/lenfnames
        #     print(fname+" error.",file = sys.stderr)
        # dirtdtw = self.dirCorr + "corr-tdtw/"
        outputName = 'tau-t_conf_rem{num}_T={temp}_timemax={tm}_splnum={spl}.t'.format(
                num = self.num, temp="%.2lf"%self.temp,
                tm="%.0le"%self.timemax,spl="%.1le"%len(fnames))
        # mkdir(dirtdtw)
        # print(dirtdtw + outputName)
        # self.corrlist = self.corrlist.tolist()
        dump_data(self.dir + outputName, [self.timelist,self.taulist])
    def plot_tau_t_conf(self,fnameNum=0):
        self.read_tau_t_conf(fnameNum)
        # mkdir(self.dirCorr + "plot/")
        mkdir(self.dir + "plot/")
        time = self.timelist
        TrappingTime = self.taulist
        plot([(time,TrappingTime)]
            ,[f'tau-t_conf']
            ,func=self.__imgpath)
        plot_xlog([(time,TrappingTime)]
            ,[f'tau-t_conf']
            ,func=self.__imgpath)
        plot_xylog([(time,TrappingTime)]
            ,[f'tau-t_conf']
            ,func=self.__imgpath)

#tau-t_conf:
    def read_tau_t(self,fnameNum=0):
        fnames = os.listdir(self.dirTauConf)
        if fnameNum:
            fnames = fnames[:fnameNum]
        lenfnames = len(fnames)
        self.timelist = read_data_from_txt(self.dirTauConf + fnames[0], [0])[0]
        self.taulist = np.array([0.0 for time in self.timelist])
        for fname in fnames:
            tau1spl = read_data_from_txt(self.dirTauConf + fname, [1])[0]
            if not tau1spl == []:
                self.taulist = self.taulist + np.array(abs(np.array(tau1spl)))
            else:
                lenfnames = lenfnames - 1
        self.taulist = self.taulist/lenfnames
        #     print(fname+" error.",file = sys.stderr)
        # dirtdtw = self.dirCorr + "corr-tdtw/"
        outputName = 'tau-t_rem{num}_T={temp}_timemax={tm}_splnum={spl}.t'.format(
                num = self.num, temp="%.2lf"%self.temp,
                tm="%.0le"%self.timemax,spl="%.1le"%len(fnames))
        # mkdir(dirtdtw)
        # print(dirtdtw + outputName)
        # self.corrlist = self.corrlist.tolist()
        dump_data(self.dir + outputName, [self.timelist,self.taulist])
    def plot_tau_t(self,fnameNum=0):
        self.read_tau_t(fnameNum)
        # mkdir(self.dirCorr + "plot/")
        mkdir(self.dir + "plot/")
        time = self.timelist
        TrappingTime = self.taulist
        plot([(time,TrappingTime)]
            ,[f'tau-t']
            ,func=self.__imgpath)
        plot_xlog([(time,TrappingTime)]
            ,[f'tau-t']
            ,func=self.__imgpath)
        plot_xylog([(time,TrappingTime)]
            ,[f'tau-t']
            ,func=self.__imgpath)

#barrier minDistancePairs:
    def read_barriers_minDistancePairs(self,fnameNum=0):
        fnames = os.listdir(self.dirBarrierMinDistance)
        if fnameNum:
            fnames = fnames[:fnameNum]
        print(fnames)
        #fnames = fnames[:100]
        self.distance = []
        self.eb1list = []
        self.eb2list = []
        self.eblist = []

        for fname in fnames:
            media = read_data_from_txt(self.dirBarrierMinDistance + fname, [0,1,2])   #distance ,eb1, eb2
            self.distance.extend(media[0])
            self.eb1list.extend(media[1])
            self.eb2list.extend(media[2])
            self.eblist.extend(media[1])
            self.eblist.extend(media[2])
        self.splnum = len(fnames)
    def plot_barriers_minDistancePairs(self,fnameNum=0):
        self.read_barriers_minDistancePairs(fnameNum)
        scatter([(self.distance,self.eb1list)],["rem{}".format(int(self.num))],xlabel="distance",ylabel="barrier",func=self.__imgpath_barrier)
        scatter_xylog([(self.distance,self.eb1list)],["rem{}".format(int(self.num))],xlabel="distance",ylabel="barrier",func=self.__imgpath_barrier)
        plot_hist2d_xylog(self.distance,self.eb1list,"rem{}".format(int(self.num)),xlabel="distance",ylabel="barrier",func=self.__imgpath_barrier)
        plot_hist2d(self.distance,self.eb1list,"rem{}".format(int(self.num)),xlabel="distance",ylabel="barrier",func=self.__imgpath_barrier)
  

#lowerst barriers:
    def read_lowest_barriers(self,fnameNum=0):
        fnames = os.listdir(self.dirBarrier)
        if fnameNum:
            fnames = fnames[:fnameNum]
        print(fnames)
        #fnames = fnames[:100]
        self.datalist = [[] for i in range(4)]
        for fname in fnames:
            media = read_data_from_txt(self.dirBarrier + fname, [0,1,2,3])   #barrier, eff_barrier, iner_en, entropy,log(1.0*conf_num)
            for i in range(4):
                self.datalist[i].extend(media[i])
        self.splnum = len(fnames)
    def get_dos_lowest_barriers(self):
        self.lsps_barrier = [[] for i in range(7)]
        self.doss_barrier = [[] for i in range(7)]
        Name = ["" for i in range(7)]
        for i in range(4):
            self.lsps_barrier[i], self.doss_barrier[i] = gen_doslist(self.datalist[i], int(self.bins/2))
        self.lsps_barrier[4], self.doss_barrier[4] = self.treeArrhenius(self.lsps_barrier[0], self.doss_barrier[0])
        if self.__txtpath("tau-Eb_Eblowsen-tau0_prob_ylog").split("/")[-1] in os.listdir(self.dir):
            self.lsps_barrier[5], self.doss_barrier[5] = self.pdfTau_from_dynamical_EbTau(self.lsps_barrier[0], self.doss_barrier[0])
        if self.__txtpath("tau-Eb_Fit_Eblowsen-tau0_prob_ylog").split("/")[-1] in os.listdir(self.dir):
            self.lsps_barrier[6], self.doss_barrier[6] = self.pdfTau_from_dynamical_EbTauFit(self.lsps_barrier[0], self.doss_barrier[0])
        # self.taulist_fromEb = np.exp(np.array(self.datalist[0])/self.temperature)
        # self.gendos_printf_tau_genfunc(self.taulist_fromEb, len(self.taulist_fromEb), 50, "treeTaulistFromEb_thenPdf",self.bins)
        Name[0] = 'pdf-barrier_rem{num}_minh={minh}_splnum={spl}_bins={bins}.t'.format(
                num = self.num, temp="%.2lf"%self.temp,bins=int(self.bins/2),
                tm="%.0le"%self.timemax,spl="%.1le"%self.splnum,minh="%.2le"%self.minh)
        Name[1] = 'pdf-effBarrier_rem{num}_minh={minh}_splnum={spl}.t'.format(
                num = self.num, temp="%.2lf"%self.temp,
                tm="%.0le"%self.timemax,spl="%.1le"%self.splnum,minh="%.2le"%self.minh)
        Name[2] = 'pdf-inerEn_rem{num}_minh={minh}_splnum={spl}.t'.format(
                num = self.num, temp="%.2lf"%self.temp,
                tm="%.0le"%self.timemax,spl="%.1le"%self.splnum,minh="%.2le"%self.minh)
        Name[3] = 'pdf-entropy_rem{num}_minh={minh}_splnum={spl}.t'.format(
                num = self.num, temp="%.2lf"%self.temp,
                tm="%.0le"%self.timemax,spl="%.1le"%self.splnum,minh="%.2le"%self.minh)
        Name[4] = 'pdf-treeTau_rem{num}_minh={minh}_splnum={spl}_bins={bins}.t'.format(
                num = self.num, temp="%.2lf"%self.temp,bins=int(self.bins/2),
                tm="%.0le"%self.timemax,spl="%.1le"%self.splnum,minh="%.2le"%self.minh)
        Name[5] = 'pdf-treeTau_fromDynamicsTauEb_rem{num}_minh={minh}_splnum={spl}_bins={bins}.t'.format(
                num = self.num, temp="%.2lf"%self.temp,bins=int(self.bins/2),
                tm="%.0le"%self.timemax,spl="%.1le"%self.splnum,minh="%.2le"%self.minh)
        Name[6] = 'pdf-treeTau_fromDynamicsTauEbFit_rem{num}_minh={minh}_splnum={spl}_bins={bins}.t'.format(
                num = self.num, temp="%.2lf"%self.temp,bins=int(self.bins/2),
                tm="%.0le"%self.timemax,spl="%.1le"%self.splnum,minh="%.2le"%self.minh)
        for i in range(7):
            dump_data(self.dirbarrier + Name[i], [self.lsps_barrier[i],self.doss_barrier[i]])
    def plot_dos_lowest_barriers(self,fnameNum=0):
        self.read_lowest_barriers(fnameNum)
        self.get_dos_lowest_barriers()
        labels = ["barrier", "effBarrier", "inerEn", "entropy","treeTrappingTime","TreeTauFromDyTauEb","TreeTauFromDyTauEbFit"]
        for i in range(7):
            plot([(self.lsps_barrier[i], self.doss_barrier[i])]
                ,[labels[i]],
                func = self.__imgpath_barrier)
            plot_xlog([(self.lsps_barrier[i], self.doss_barrier[i])]
                ,[labels[i]],
                func = self.__imgpath_barrier)
            plot_ylog([(self.lsps_barrier[i], self.doss_barrier[i])]
                ,[labels[i]],
                func = self.__imgpath_barrier)
            plot_xylog([(self.lsps_barrier[i], self.doss_barrier[i])]
                ,[labels[i]],
                func = self.__imgpath_barrier)
    def plot_dos_low_eb_only(self,fnameNum=0):
        self.read_lowest_barriers(fnameNum)
        self.get_dos_lowest_barriers()
        self.__plot_ylog_eb_fitfunc("eb_mixed_fitfunc_Tsplnum={spl}_rem{a}_T={b}_timemax={c}.t".format(a=self.num
            ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,spl="%.1le"%self.splnum)
            ,"pdf"
            ,self.lsps_barrier[0], self.doss_barrier[0]
            ,"rem{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature))

    def read_dynamical_EbTau_relation(self):
        fname = self.__txtpath("tau-Eb_{}".format("Eblowsen-tau0_prob_ylog"))
        media = read_data_from_txt(fname, [0,1])
        eb, tau = media[0], np.array(media[1])
        ebset = list(set(eb))  #[1,2,2,3,3,3]
        idxlists = [[idx for idx,val in enumerate(eb) if val == ele] for ele in ebset] #[[0], [1, 2], [3, 4, 5]]
        taulist = [int(np.sum(np.array(tau[idxlist]))/len(idxlist)) for idxlist in idxlists]
        return media[0], media[1]  #Eb, tau
    def pdfTau_from_dynamical_EbTau(self,lsp,pdf):
        self.arreb, self.arrtau = self.read_dynamical_EbTau_relation()
        lsp_pdf = np.array([self.pdfTau_from_dynamical_EbTau_1ele(lsp[i],pdf[i]) for i in range(len(lsp))]).T
        return lsp_pdf[0],lsp_pdf[1]
        # dosout = [dos[i]*self.temp/lspout[i] for i in range(len(dos))]
    def read_dynamical_EbTau_relationFit(self):
        fname = self.__txtpath("tau-Eb_Fit_{}".format("Eblowsen-tau0_prob_ylog"))
        media = read_data_from_txt(fname, [0,1])
        eb, tau = media[0], np.array(media[1])
        ebset = list(set(eb))  #[1,2,2,3,3,3]
        idxlists = [[idx for idx,val in enumerate(eb) if val == ele] for ele in ebset] #[[0], [1, 2], [3, 4, 5]]
        taulist = [int(np.sum(np.array(tau[idxlist]))/len(idxlist)) for idxlist in idxlists]
        return media[0], media[1]  #Eb, tau
    def pdfTau_from_dynamical_EbTauFit(self,lsp,pdf):
        self.arreb, self.arrtau = self.read_dynamical_EbTau_relationFit()
        lsp_pdf = np.array([self.pdfTau_from_dynamical_EbTau_1ele(lsp[i],pdf[i]) for i in range(len(lsp))]).T
        return lsp_pdf[0],lsp_pdf[1]
        # dosout = [dos[i]*self.temp/lspout[i] for i in range(len(dos))]
    def pdfTau_from_dynamical_EbTau_1ele(self,eb,f):
        idx = 0
        ptau = 0.0
        tau = 0.0
        if eb > self.arreb[-1]:
            tau = self.arrtau[-1]
            der = (self.arreb[-1]-self.arreb[-2])/(self.arrtau[-1]-self.arrtau[-2])
            ptau = der * f
            idx = 1
        else:
            for i in range(len(self.arreb) - 1):
                if self.arreb[i]<eb<=self.arreb[i+1]:
                    tau = self.arrtau[i]  #tau(Eb)
                    der = (self.arreb[i+1]-self.arreb[i])/(self.arrtau[i+1]-self.arrtau[i]) # dEb/dtau
                    ptau = der * f    #p(tau) = dEb/dtau * f(Eb)
                    break
            idx = 2
        if idx==0:
            print(eb)
        return [tau, ptau]
#barrier Etip:
    def read_barrier(self,fnameNum=0):
        dirbarrier = self.dirbarrier + "barrier/"
        fnames = os.listdir(dirbarrier)
        fnames = [fnameB for fnameB in fnames if fnameB[-3:]=="bar"]
        self.rdn = random.randint(0,len(fnames))
        if fnameNum == 1:
            fnames = [fnames[self.rdn]]
        elif fnameNum:
            fnames = fnames[:fnameNum]
        self.etiplist = []
        for fname in fnames:
            with open(dirbarrier + fname,"r",encoding="utf-8") as fp:
                fp.readline()
                while 1:
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    self.etiplist.append(float(linedata.split()[2])/self.num)
        
        self.splnum = len(self.etiplist)
    def read_barrier2(self,fnameNum=0,eth=0.6):
        fnames = os.listdir(self.dirBarrier)
        self.rdn = random.randint(0,len(fnames))
        if fnameNum == 1:
            fnames = [fnames[self.rdn]]
        elif fnameNum:
            fnames = fnames[:fnameNum]
        self.eslist = []
        self.etiplist = []
        self.inerenlist = []
        self.eblist = []

        for fname in fnames:
            media = read_data_from_txt(self.dirBarrier + fname, [6,7,2])#eb 0, eff_barrier 1, iner_en 2, entropy 3,log(1.0*conf_num) 4,cur->up->en - iner_en 5
                                                                            #, cur->en 6, cur->up->en 7
            self.etiplist.extend(media[0])
            self.eslist.extend(media[1])
            # self.inerenlist.extend(media[2])
        self.eblist = [self.eslist[i]-self.etiplist[i] for i in range(len(self.eslist))]
        self.effetiplist = [self.eff_etip(etip,eth) for etip in self.etiplist]
        self.effeslist = [self.effetiplist[i]+self.eblist[i] for i in range(len(self.eslist))]
        self.splnum = len(self.etiplist)  
    def Boltzmann_Esmodified(self,elist,temp):
        weightlist = np.exp(np.array(elist)*(-1)/temp)
        resultlist = [weight/np.sum(weightlist) for weight in weightlist]
        return resultlist
    def eff_etip(self,etip,value):
        result = etip
        if etip < -value*self.num:
            result = -value*self.num
        return result
    def plot_barrier_pdf_etip(self,temp,fnameNum=0):
        self.read_barrier2(fnameNum)
        print(self.etiplist[:100])
        # weightlist = self.Boltzmann_Esmodified(self.etiplist,temp)
        weightlist = self.Boltzmann_Esmodified(self.eslist,temp)
        etip_output, weight_output = Funcsumpdf((np.array(self.etiplist)/self.num).tolist(),weightlist,self.bins)
        output_name = 'pdf-Etip_BoltzmannEsMod_rem{num}_T={temp}_minh={minh}_splnum={spl}_bins={bins}.t'.format(
                num = self.num, temp="%.2lf"%temp,
                spl="%.1le"%self.splnum,minh="%.2le"%self.minh,bins=self.bins)
        dump_data(self.dirbarrier + output_name, [etip_output, weight_output])
        plot([(etip_output, weight_output)]
                ,["pdf-Etip_BoltzmannEsMod T={T} rem{num}".format(T="%.2lf"%temp,num=self.num)],
                func = self.__imgpath_barrier)

        eb_output, weight_output = Funcsumpdf(self.eblist,weightlist,self.bins)
        output_name = 'pdf-Eb_BoltzmannEsMod_rem{num}_T={temp}_minh={minh}_splnum={spl}_bins={bins}.t'.format(
                num = self.num, temp="%.2lf"%temp,
                spl="%.1le"%self.splnum,minh="%.2le"%self.minh,bins=self.bins)
        dump_data(self.dirbarrier + output_name, [eb_output, weight_output])
        plot_ylog([(eb_output, weight_output)]
                ,["pdf-Eb_BoltzmannEsMod T={T} rem{num}".format(T="%.2lf"%temp,num=self.num)],
                func = self.__imgpath_barrier)
    def plot_barrier_pdf_etip2(self,temp,fnameNum=0,eth=0.6):
        self.read_barrier2(fnameNum,eth)
        print(self.etiplist[:100])
        print(self.effetiplist[:100])

        # weightlist = self.Boltzmann_Esmodified(self.etiplist,temp)
        weightlist = self.Boltzmann_Esmodified(self.effeslist,temp)
        etip_output, weight_output = Funcsumpdf((np.array(self.etiplist)/self.num).tolist(),weightlist,self.bins)
        output_name = 'pdf-Etip_BoltzmannEsMod_effetip_rem{num}_T={temp}_eth={eth}_minh={minh}_splnum={spl}_bins={bins}.t'.format(
                num = self.num, temp="%.2lf"%temp,eth="%.2lf"%eth,
                spl="%.1le"%self.splnum,minh="%.2le"%self.minh,bins=self.bins)
        dump_data(self.dirbarrier + output_name, [etip_output, weight_output])
        plot([(etip_output, weight_output)]
                ,["pdf-Etip_BoltzmannEsMod_effetip T={T} rem{num} eth={eth}".format(T="%.2lf"%temp,num=self.num,eth="%.2lf"%eth)],
                func = self.__imgpath_barrier)

        eb_output, weight_output = Funcsumpdf(self.eblist,weightlist,self.bins)
        output_name = 'pdf-Eb_BoltzmannEsMod_effetip_rem{num}_T={temp}_eth={eth}_minh={minh}_splnum={spl}_bins={bins}.t'.format(
                num = self.num, temp="%.2lf"%temp,eth="%.2lf"%eth,
                spl="%.1le"%self.splnum,minh="%.2le"%self.minh,bins=self.bins)
        dump_data(self.dirbarrier + output_name, [eb_output, weight_output])
        plot_ylog([(eb_output, weight_output)]
                ,["pdf-Eb_BoltzmannEsMod_effetip T={T} rem{num} eth={eth}".format(T="%.2lf"%temp,num=self.num,eth="%.2lf"%eth)],
                func = self.__imgpath_barrier)
    def get_tipsbarriers_Boltzmann_modified_effeb(self,temp,fnameNum=0):
        fnames = os.listdir(self.dirTipsBarriers)
        self.rdn = random.randint(0,len(fnames))
        if fnameNum == 1:
            fnames = [fnames[self.rdn]]
        elif fnameNum:
            fnames = fnames[:fnameNum]
        self.eslist = []
        self.etiplist = []
        self.inerenlist = []
        self.eblist = []

        for fname in fnames:
            media = read_data_from_txt(self.dirTipsBarriers + fname, [0,1,2])#eb 0, eff_barrier 1, iner_en 2, entropy 3,log(1.0*conf_num) 4,cur->up->en - iner_en 5
                                                                            #, cur->en 6, cur->up->en 7
            self.etiplist.extend(media[0])
            self.eblist.extend(media[1])
            # self.inerenlist.extend(media[2])
        self.eslist = [self.eblist[i]+self.etiplist[i] for i in range(len(self.eblist))]
        self.splnum = len(self.etiplist)
    def plot_tipsbarriers_effeb_pdf_etipeb(self,temp,fnameNum=0):
        self.get_tipsbarriers_Boltzmann_modified_effeb(fnameNum)
        print(self.etiplist[:100])
        # weightlist = self.Boltzmann_Esmodified(self.etiplist,temp)
        weightlist = self.Boltzmann_Esmodified(self.eslist,temp)
        etip_output, weight_output = Funcsumpdf((np.array(self.etiplist)/self.num).tolist(),weightlist,self.bins)
        output_name = 'pdf-Etip_tipsbarriersAll_BoltzmannEsMod_rem{num}_T={temp}_minh={minh}_splnum={spl}_bins={bins}.t'.format(
                num = self.num, temp="%.2lf"%temp,
                spl="%.1le"%self.splnum,minh="%.2le"%self.minh,bins=self.bins)
        dump_data(self.dirbarrier + output_name, [etip_output, weight_output])
        plot([(etip_output, weight_output)]
                ,["pdf-Etip_tipsbarriersAll_BoltzmannEsMod T={T} rem{num}".format(T="%.2lf"%temp,num=self.num)],
                func = self.__imgpath_barrier)

        eb_output, weight_output = Funcsumpdf(self.eblist,weightlist,self.bins)
        output_name = 'pdf-Eb_tipsbarriersAll_BoltzmannEsMod_rem{num}_T={temp}_minh={minh}_splnum={spl}_bins={bins}.t'.format(
                num = self.num, temp="%.2lf"%temp,
                spl="%.1le"%self.splnum,minh="%.2le"%self.minh,bins=self.bins)
        dump_data(self.dirbarrier + output_name, [eb_output, weight_output])
        plot_ylog([(eb_output, weight_output)]
                ,["pdf-Eb_tipsbarriersAll_BoltzmannEsMod T={T} rem{num}".format(T="%.2lf"%temp,num=self.num)],
                func = self.__imgpath_barrier)

    def get_minmax_of_tree(self,fnameNum=0):
        fnames = os.listdir(self.dirBarrier)
        self.rdn = random.randint(0,len(fnames))
        if fnameNum == 1:
            fnames = [fnames[self.rdn]]
        elif fnameNum:
            fnames = fnames[:fnameNum]
        Minetip = 9999999999999
        Maxetip = -9999999999999
        Mineb = 9999999999999
        Maxeb = -9999999999999
        self.splnum = 0

        for fname in fnames:
            media = read_data_from_txt(self.dirBarrier + fname, [6,7,2])#eb 0, eff_barrier 1, iner_en 2, entropy 3,log(1.0*conf_num) 4,cur->up->en - iner_en 5
                                                                            #, cur->en 6, cur->up->en 7
            self.splnum += len(media[0])
            eblist = [media[1][i]-media[0][i] for i in range(len(media[0]))]
            etiplist = [media[0][i]/self.num for i in range(len(media[0]))]
            eslist = media[1]
            etiplist.append(Minetip)
            Minetip = min(etiplist)
            etiplist = etiplist[:-1]
            etiplist.append(Maxetip)
            Maxetip = max(etiplist) 
            eblist.append(Mineb)
            Mineb = min(eblist)
            eblist = eblist[:-1]
            eblist.append(Maxeb)
            Maxeb = max(eblist) 

            # self.inerenlist.extend(media[2])    
        return Minetip,Maxetip,Mineb,Maxeb
    # def get_tipsbarriers_Boltzmann_modified_effeb_1by1(self,temp,fnameNum=0):
    #     fnames = os.listdir(self.dirTipsBarriers)
    #     self.rdn = random.randint(0,len(fnames))
    #     if fnameNum == 1:
    #         fnames = [fnames[self.rdn]]
    #     elif fnameNum:
    #         fnames = fnames[:fnameNum]
    #     self.etip_weight = np.array([0.0 for i in range(self.bins)])
    #     self.eb_weight = np.array([0.0 for i in range(self.bins)])

    #     Minetip,Maxetip,Mineb,Maxeb = self.get_minmax_of_tree(fnameNum)
    #     for fname in fnames:
    #         media = read_data_from_txt(self.dirTipsBarriers + fname, [0,1,2])#etip 0, eff_eb 1, BoltzmannMod 2
    #         etiplist = media[0]
    #         eblist = media[1]
    #         weightlist = media[2]
    #         etip_output, etip_weight = Funcsumpdf((np.array(etiplist)/self.num).tolist(),weightlist,self.bins,mm=[Minetip,Maxetip])
    #         eb_output, eb_weight = Funcsumpdf(eblist,weightlist,self.bins,mm=[Mineb,Maxeb])
    #         self.etip_weight = self.etip_weight + np.array(etip_weight)/len(fnames)
    #         self.eb_weight = self.eb_weight + np.array(eb_weight)/len(fnames)
    #     self.etip_output = etip_output
    #     self.eb_output = eb_output
    # def plot_tipsbarriers_effeb_pdf_etipeb_1by1(self,temp,fnameNum=0):
    #     self.get_tipsbarriers_Boltzmann_modified_effeb_1by1(temp,fnameNum)
    #     print(self.eb_output)
    #     output_name = 'pdf-Etip_effeb_1by1sum_BoltzmanneffEsMod_rem{num}_T={temp}_minh={minh}_splnum={spl}_bins={bins}.t'.format(
    #             num = self.num, temp="%.2lf"%temp,
    #             spl="%.1le"%self.splnum,minh="%.2le"%self.minh,bins=self.bins)
    #     dump_data(self.dirbarrier + output_name, [self.etip_output,  self.etip_weight])
    #     plot([(self.etip_output,  self.etip_weight)]
    #             ,["pdf-Etip_BoltzmanneffEsMod T={T} rem{num} effeb_1by1sum".format(T="%.2lf"%temp,num=self.num)],
    #             func = self.__imgpath_barrier)

    #     output_name = 'pdf-Eb_effeb_1by1sum_BoltzmanneffEsMod_rem{num}_T={temp}_minh={minh}_splnum={spl}_bins={bins}.t'.format(
    #             num = self.num, temp="%.2lf"%temp,
    #             spl="%.1le"%self.splnum,minh="%.2le"%self.minh,bins=self.bins)
    #     dump_data(self.dirbarrier + output_name, [self.eb_output, self.eb_weight])
    #     plot_ylog([(self.eb_output, self.eb_weight)]
    #             ,["pdf-Eb_BoltzmanneffEsMod T={T} rem{num} effeb_1by1sum".format(T="%.2lf"%temp,num=self.num)],
    #             func = self.__imgpath_barrier)
    def Boltzmann_EbEtipmodified(self,eblist,etip,temp):
        weightlist = np.exp(np.array(eblist)*(-1)/temp)
        resultlist = [np.exp(-(2.0*eblist[i]+etip)/temp)/np.sum(weightlist) for i in range(len(weightlist))]
        return resultlist
    def Boltzmann_EbEtipmodified2(self,eblist,etip,temp):
        weightlist = np.exp((np.array(eblist)+etip)*(-1)/temp)
        resultlist = [weight/np.sum(weightlist) for weight in weightlist]
        return resultlist
    def get_minmaxEb_of_tipsbarriers(self,fnameNum=0):
        fnames = os.listdir(self.dirTipsBarriers)
        self.rdn = random.randint(0,len(fnames))
        if fnameNum == 1:
            fnames = [fnames[self.rdn]]
        elif fnameNum:
            fnames = fnames[:fnameNum]
        Mineb = 9999999999999
        Maxeb = -9999999999999
        self.splnum = 0

        for fname in fnames:
            eblists = []
            with open(self.dirTipsBarriers + fname,"r",encoding="utf-8") as fp:
                while 1:
                    linedata = fp.readline()
                    if linedata.strip() == "":
                        break
                    data = linedata.split()
                    eblist = [float(eb) for eb in data[1:]]
                    eblists.extend(eblist)
            self.splnum += len(eblists)
            eblists.append(Mineb)
            Mineb = min(eblists)
            eblists = eblists[:-1]
            eblists.append(Maxeb)
            Maxeb = max(eblists) 
        return Mineb,Maxeb
    def get_tipsbarriers_Boltzmann_modified_1by1(self,temp,fnameNum=0):
        fnames = os.listdir(self.dirTipsBarriers)
        self.rdn = random.randint(0,len(fnames))
        if fnameNum == 1:
            fnames = [fnames[self.rdn]]
        elif fnameNum:
            fnames = fnames[:fnameNum]
        self.eb_weight = np.array([0.0 for i in range(self.bins)])
        eblists = []
        weightlists = []
        Mineb,Maxeb = self.get_minmaxEb_of_tipsbarriers(fnameNum)
        for fname in fnames:
            eblists = []
            weightlists = []
            with open(self.dirTipsBarriers + fname,"r",encoding="utf-8") as fp:
                while 1:
                    linedata = fp.readline()
                    if linedata.strip() == "":
                        break
                    data = linedata.split()
                    etip = float(data[0])
                    eblist = [float(eb) for eb in data[1:]]
                    eblists.extend(eblist)
                    weightlist = self.Boltzmann_EbEtipmodified(eblist,etip,temp)
                    weightlists.extend(weightlist)
            weightlists = np.array(weightlists)/np.sum(weightlists)
            eb_output, eb_weight = Funcsumpdf(eblists,weightlists,self.bins,mm=[Mineb,Maxeb])
            self.eb_weight = self.eb_weight + np.array(eb_weight)/len(fnames)
        self.eb_output = eb_output
    def plot_barrier_pdf_etipeb_allEb_1by1(self,temp,fnameNum=0):
        self.get_tipsbarriers_Boltzmann_modified_1by1(temp,fnameNum)
        print(self.eb_output)

        output_name = 'pdf-Eb_tipsAllbarriers_1by1sum_BoltzmannEbEtipMod_rem{num}_T={temp}_minh={minh}_splnum={spl}_bins={bins}.t'.format(
                num = self.num, temp="%.2lf"%temp,
                spl="%.1le"%self.splnum,minh="%.2le"%self.minh,bins=self.bins)
        dump_data(self.dirbarrier + output_name, [self.eb_output, self.eb_weight])
        plot_ylog([(self.eb_output, self.eb_weight)]
                ,["pdf-Eb_tipsAllbarriers_BoltzmannEbEtipMod T={T} rem{num} 1by1sum".format(T="%.2lf"%temp,num=self.num)],
                func = self.__imgpath_barrier)
    

    def get_tipsbarriers_Boltzmann_modified(self,temp,fnameNum=0):
        fnames = os.listdir(self.dirTipsBarriers)
        self.rdn = random.randint(0,len(fnames))
        if fnameNum == 1:
            fnames = [fnames[self.rdn]]
        elif fnameNum:
            fnames = fnames[:fnameNum]
        self.eb_weight = np.array([0.0 for i in range(self.bins)])
        self.eblists = []
        self.weightlists = []
        for fname in fnames:
            with open(self.dirTipsBarriers + fname,"r",encoding="utf-8") as fp:
                while 1:
                    linedata = fp.readline()
                    if linedata.strip() == "":
                        break
                    data = linedata.split()
                    etip = float(data[0])
                    eblist = [float(eb) for eb in data[1:]]
                    self.eblists.extend(eblist)
                    # weightlist = self.Boltzmann_EbEtipmodified(eblist,etip,temp)
                    weightlist = np.exp((2*np.array(eblist)+etip)*(-1)/temp)
                    self.weightlists.extend(weightlist)
        self.splnum = len(self.weightlists)
    def plot_barrier_pdf_etipeb_allEb(self,temp,fnameNum=0):
        self.get_tipsbarriers_Boltzmann_modified(temp,fnameNum)
        print(self.weightlists[:20])
        self.weightlists = np.array(self.weightlists)/np.sum(self.weightlists)
        eb_output, eb_weight = Funcsumpdf(self.eblists,self.weightlists,self.bins)
        output_name = 'pdf-Eb_tipsAllbarriers_BoltzmannEbEtipMod_rem{num}_T={temp}_minh={minh}_splnum={spl}_bins={bins}.t'.format(
                num = self.num, temp="%.2lf"%temp,
                spl="%.1le"%self.splnum,minh="%.2le"%self.minh,bins=self.bins)
        dump_data(self.dirbarrier + output_name, [eb_output, eb_weight])
        plot_ylog([(eb_output, eb_weight)]
                ,["pdf-Eb_tipsAllbarriers_BoltzmannEbEtipMod T={T} rem{num}".format(T="%.2lf"%temp,num=self.num)],
                func = self.__imgpath_barrier)

    def get_barrier_Boltzmann_modified_1by1(self,temp,fnameNum=0):
        fnames = os.listdir(self.dirBarrier)
        self.rdn = random.randint(0,len(fnames))
        if fnameNum == 1:
            fnames = [fnames[self.rdn]]
        elif fnameNum:
            fnames = fnames[:fnameNum]
        self.etip_weight = np.array([0.0 for i in range(self.bins)])
        self.eb_weight = np.array([0.0 for i in range(self.bins)])

        Minetip,Maxetip,Mineb,Maxeb = self.get_minmax_of_tree(fnameNum)

        for fname in fnames:
            media = read_data_from_txt(self.dirBarrier + fname, [6,7,2])#eb 0, eff_barrier 1, iner_en 2, entropy 3,log(1.0*conf_num) 4,cur->up->en - iner_en 5
                                                                            #, cur->en 6, cur->up->en 7
            eblist = [media[1][i]-media[0][i] for i in range(len(media[0]))]
            etiplist = media[0]
            eslist = media[1]
            weightlist = self.Boltzmann_Esmodified(eslist,temp)
            etip_output, etip_weight = Funcsumpdf((np.array(etiplist)/self.num).tolist(),weightlist,self.bins,mm=[Minetip,Maxetip])
            eb_output, eb_weight = Funcsumpdf(eblist,weightlist,self.bins,mm=[Mineb,Maxeb])
            self.etip_weight = self.etip_weight + np.array(etip_weight)/len(fnames)
            self.eb_weight = self.eb_weight + np.array(eb_weight)/len(fnames)
        self.etip_output = etip_output
        self.eb_output = eb_output
    def plot_barrier_pdf_etipeb_1by1(self,temp,fnameNum=0):
        self.get_barrier_Boltzmann_modified_1by1(temp,fnameNum)
        print(self.eb_output)
        output_name = 'pdf-Etip_1by1sum_BoltzmannEsMod_rem{num}_T={temp}_minh={minh}_splnum={spl}_bins={bins}.t'.format(
                num = self.num, temp="%.2lf"%temp,
                spl="%.1le"%self.splnum,minh="%.2le"%self.minh,bins=self.bins)
        dump_data(self.dirbarrier + output_name, [self.etip_output,  self.etip_weight])
        plot([(self.etip_output,  self.etip_weight)]
                ,["pdf-Etip_BoltzmannEsMod T={T} rem{num} 1by1sum".format(T="%.2lf"%temp,num=self.num)],
                func = self.__imgpath_barrier)

        output_name = 'pdf-Eb_1by1sum_BoltzmannEsMod_rem{num}_T={temp}_minh={minh}_splnum={spl}_bins={bins}.t'.format(
                num = self.num, temp="%.2lf"%temp,
                spl="%.1le"%self.splnum,minh="%.2le"%self.minh,bins=self.bins)
        dump_data(self.dirbarrier + output_name, [self.eb_output, self.eb_weight])
        plot_ylog([(self.eb_output, self.eb_weight)]
                ,["pdf-Eb_BoltzmannEsMod T={T} rem{num} 1by1sum".format(T="%.2lf"%temp,num=self.num)],
                func = self.__imgpath_barrier)
    def plot_barrier_pdf_lowest_es(self,fnameNum=0):
        self.read_barrier2(fnameNum)
        lsp, dos = gen_doslist(np.array(self.eslist)/self.num, int(self.bins))
        output_name = 'pdf-Es_TreeLowest_rem{num}_minh={minh}_splnum={spl}_bins={bins}.t'.format(
                num = self.num,spl="%.1le"%self.splnum,minh="%.2le"%self.minh,bins=self.bins)
        dump_data(self.dirbarrier + output_name, [lsp, dos])
        plot([(lsp, dos)]
            ,["TreeLowest rem{num} splnum={spl} bins={bins}".format(num=self.num,spl="%.1le"%self.splnum,bins=self.bins)]
            ,xlabel="Es",ylabel="pdf"
            ,func = self.__imgpath_barrier)
    def plot_barrier_pdf2d_lowestEs_Etip(self,fnameNum=0):
        self.read_barrier2(fnameNum)
        plot_hist2d(np.array(self.etiplist)/self.num,np.array(self.eslist)/self.num
            ,"pdf splnum={spl} bins={bins}".format(spl="%.1le"%self.splnum,bins=self.bins)
            ,xlabel="Etip",ylabel="Es"
            ,func=self.__imgpath_barrier,bins=self.bins)
    def plot_barrier_pdf2d_lowestEb_Es(self,fnameNum=0):
        self.read_barrier2(fnameNum)
        plot_hist2d(np.array(self.eblist)/self.num,np.array(self.eslist)/self.num
            ,"pdf splnum={spl} bins={bins}".format(spl="%.1le"%self.splnum,bins=self.bins)
            ,xlabel="Eb",ylabel="Es"
            ,func=self.__imgpath_barrier,bins=self.bins)
    def plot_barrier_pdf2d_lowestEb_Etip(self,fnameNum=0):
        self.read_barrier2(fnameNum)
        plot_hist2d(np.array(self.eblist)/self.num,np.array(self.etiplist)/self.num
            ,"pdf splnum={spl} bins={bins}".format(spl="%.1le"%self.splnum,bins=self.bins)
            ,xlabel="Eb",ylabel="Etip"
            ,func=self.__imgpath_barrier,bins=self.bins)
    




#E-t:

#obs-t: 
    def read_averageObs_t(self,fnameNum=0):
        fnames = os.listdir(self.dirObs)
        if fnameNum:
            fnames = fnames[:fnameNum]
        lenfnames = len(fnames)
        self.obslists = [[] for i in range(8)] 
        self.obslists[0] = read_data_from_txt(self.dirObs + fnames[0], [0])[0] #timelist
        for i in range(1,8):
            self.obslists[i] = np.array([0.0 for j in range(len(self.obslists[0]))])
        for fname in fnames:
            media = read_data_from_txt(self.dirObs + fname, [2,3,4,5,6,7,8])   #time, tip, tau, tau0, tipen, sen, lowsen, freeen, entropy            
            if not media[0] == []:
                for i in range(1,8):
                    self.obslists[i] = self.obslists[i] + np.array(media[i-1])
            else:
                lenfnames = lenfnames - 1
        for i in range(1,8):
            self.obslists[i] = self.obslists[i]/lenfnames
        output_name = "averageObs-t_rem{num}_T={temp}_timemax={tm}_splnum={spl}.t".format(
                num = self.num, temp="%.2lf"%self.temp,
                tm="%.0le"%self.timemax,spl="%.1le"%len(fnames))
        dump_data(self.dir + output_name, self.obslists)
    def plot_averageObs_t(self,fnameNum=0):
        self.read_averageObs_t(fnameNum)
        labels = ["tau-t", "tau0-t", "tipEn-t", "saddleEn-t", "lowestEn-t","freeEnergy-t","entropy-t"]
        # self.obslists[0] = np.array(self.obslists[0])
        # print(self.obslists[0])
        # print(self.obslists[1])
        # plot([(self.obslists[0], self.obslists[1])]
        #         ,[labels[0]],
        #         func = self.__imgpath)
        for i in range(1,8):
            plot([(self.obslists[0], self.obslists[i])]
                ,[labels[i-1]],
                func = self.__imgpath)
            plot_xlog([(self.obslists[0], self.obslists[i])]
                ,[labels[i-1]],
                func = self.__imgpath)
            plot_xylog([(self.obslists[0], self.obslists[i])]
                ,[labels[i-1]],
                func = self.__imgpath)
    
    def read_E_t(self,fnameNum=0):
        fnames = os.listdir(self.dirE)
        if fnameNum:
            fnames = fnames[:fnameNum]
        lenfnames = len(fnames)
        self.obslists = [[] for i in range(2)] 
        self.obslists[0] = read_data_from_txt(self.dirE + fnames[0], [0])[0] #timelist
        self.obslists[1] = np.array([0.0 for j in range(len(self.obslists[0]))])
        for fname in fnames:
            media = read_data_from_txt(self.dirE + fname, [1])   #time, tip, tau, tau0, tipen, sen, lowsen, freeen, entropy            
            if not media[0] == []:
                self.obslists[1] = self.obslists[1] + np.array(media[0])
            else:
                lenfnames = lenfnames - 1
        
        self.obslists[1] = self.obslists[1]/lenfnames
        output_name = "E-t_rem{num}_T={temp}_timemax={tm}_splnum={spl}.t".format(
                num = self.num, temp="%.2lf"%self.temp,
                tm="%.0le"%self.timemax,spl="%.1le"%len(fnames))
        dump_data(self.dir + output_name, self.obslists)
    def plot_E_t(self,fnameNum=0):
        self.read_E_t(fnameNum)
        labels = ["E-t"]
        # self.obslists[0] = np.array(self.obslists[0])
        # print(self.obslists[0])
        # print(self.obslists[1])
        # plot([(self.obslists[0], self.obslists[1])]
        #         ,[labels[0]],
        #         func = self.__imgpath)
        plot([(self.obslists[0], self.obslists[1])]
            ,[labels[0]],
            func = self.__imgpath)
        plot_xlog([(self.obslists[0], self.obslists[1])]
            ,[labels[0]],
            func = self.__imgpath)


#pdf-tau0 dynamics of activation:
    def read_activation_taulists(self,time_th,fnameNum=0,time_start=1):
        fnames = os.listdir(self.dirActivation)
        if fnameNum:
            fnames = fnames[:int(fnameNum)]
        self.fnum = len(fnames)
        self.taulist = []
        self.tau0list = []
        self.tauslist = []
        # timelist = read_data_from_txt(self.dirActivation + fnames[0], [0])[0] #timelist
        # for i in range(len(timelist)):
        #     if timelist[i] > time_th:
        #         break
        # ith = i
        self.splsum = 0
        for fname in fnames:
            media = read_data_from_txt(self.dirActivation + fname, [0,8,9,10])   #0 i,1 last_sen,2 last_basin_tipen,3 sen
                                                                                #,4 last_sdec,5 last_basin_tip,6 sdec
                                                                                #,7 last_lowest_sdec
                                                                                #,8 basin_time,9 hangout_time,10 trapping_time  
                                                                                #,11 iner_en,12 now_eff_free_en,13 last_lowest_sen
                                                                                #,14 confnum             
            media_time = media[0]
            for i in range(len(media_time)):
                if media_time[i] > time_start:
                    break
            ist = i
            for i in range(ist,len(media_time)):
                if media_time[i] > time_th:
                    break
            ith = i
            self.splsum += ith - ist
            media_tau0 = media[1][ist:ith]
            media_taus = media[2][ist:ith]
            media_tau = media[3][ist:ith]
            self.taulist.extend(media_tau)
            self.tau0list.extend(media_tau0)
            self.tauslist.extend(media_taus)
        # output_name = "averageObs-t_rem{num}_T={temp}_timemax={tm}_splnum={spl}.t".format(
        #         num = self.num, temp="%.2lf"%self.temp,
        #         tm="%.0le"%self.timemax,spl="%.1le"%len(fnames))
        # dump_data(self.dir + output_name, self.obslists)
    def plot_activation_taulists(self,time_th,fnameNum=0,time_start=1,bins_trct=100):
        self.read_activation_taulists(time_th,fnameNum,time_start)
        self.gendos_printf_tau_genfunc(self.taulist, self.splsum, bins_trct, "mc_taulist_beforeThermal_time={s}-{e}".format(s="%.1le"%time_start,e="%.1le"%time_th),self.bins)
        self.gendos_printf_tau_genfunc(self.tau0list, self.splsum, bins_trct, "mc_tau0list_beforeThermal_time={s}-{e}".format(s="%.1le"%time_start,e="%.1le"%time_th),self.bins)
        self.gendos_printf_tau_genfunc(self.tauslist, self.splsum, 20, "mc_tauslist_beforeThermal_timeh={s}-{e}".format(s="%.1le"%time_start,e="%.1le"%time_th),20)

#pdf activation_eb:
    def read_activation_barrier(self,time_th,fnameNum=0,time_start=1):
        fnames = os.listdir(self.dirActivation)
        if fnameNum:
            fnames = fnames[:int(fnameNum)]
        self.fnum = len(fnames)
        self.barrier = []
        self.barrier_lowsen = []
        self.barrier_iner_lowsen = []
        self.barrier_free_lowsen = []

        self.splsum = 0
        for fname in fnames:
            media = read_data_from_txt(self.dirActivation + fname, [0,8,9,10,11,12,13,2,3])   #0 i,1 last_sen,2 last_basin_tipen,3 sen
                                                                                #,4 last_sdec,5 last_basin_tip,6 sdec
                                                                                #,7 last_lowest_sdec
                                                                                #,8 basin_time,9 hangout_time,10 trapping_time  
                                                                                #,11 iner_en,12 now_eff_free_en,13 last_lowest_sen
                                                                                #,14 confnum             
            media_time = media[0]
            for i in range(len(media_time)):
                if media_time[i] > time_start:
                    break
            ist = i
            for i in range(ist,len(media_time)):
                if media_time[i] > time_th:
                    break
            ith = i
            self.splsum += ith - ist

            media_barrier = [media[8][i] - media[7][i] for i in range(ist, ith)]
            media_barrier_lowsen = [media[6][i] - media[7][i] for i in range(ist, ith)]
            media_barrier_IL = [media[6][i] - media[4][i] for i in range(ist, ith)]
            media_barrier_FL = [media[6][i] - media[5][i] for i in range(ist, ith)]
            # self.barrier.extend(media_barrier)
            self.barrier_lowsen.extend(media_barrier_lowsen)
            # self.barrier_iner_lowsen.extend(media_barrier_IL)
            # self.barrier_free_lowsen.extend(media_barrier_FL)
    def plot_activation_barrier(self,time_th,fnameNum=0,time_start=1):
        self.read_activation_barrier(time_th,fnameNum,time_start)
        # self.genpdf_printf_Eb(self.barrier,"standard_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),self.bins)
        self.genpdf_printf_Eb(self.barrier_lowsen,"lowsen_standard_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),int(self.bins/5))
        # self.genpdf_printf_Eb(self.barrier_iner_lowsen,"iner_lowsen_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),int(self.bins/5))
        # self.genpdf_printf_Eb(self.barrier_free_lowsen,"free_lowsen_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),int(self.bins/5))
#pdf-E activation:
    def read_activation_E(self,time_th,fnameNum=0,time_start=1):
        fnames = os.listdir(self.dirActivation)
        if fnameNum:
            fnames = fnames[:int(fnameNum)]
        self.fnum = len(fnames)
        self.Etiplist = []
        self.Eslist = []
        self.splsum = 0
        for fname in fnames:
            media = read_data_from_txt(self.dirActivation + fname, [0,2,13])   #0 i,1 last_sen,2 last_basin_tipen,3 sen
                                                                                #,4 last_sdec,5 last_basin_tip,6 sdec
                                                                                #,7 last_lowest_sdec
                                                                                #,8 basin_time,9 hangout_time,10 trapping_time  
                                                                                #,11 iner_en,12 now_eff_free_en,13 last_lowest_sen
                                                                                #,14 confnum             
            media_time = media[0]
            for i in range(len(media_time)):
                if media_time[i] > time_start:
                    break
            ist = i
            for i in range(ist,len(media_time)):
                if media_time[i] > time_th:
                    break
            ith = i
            self.splsum += ith - ist

            self.Etiplist.extend(media[1][ist:ith])
            self.Eslist.extend(media[2][ist:ith])
        self.Eblist = [self.Eslist[i] - self.Etiplist[i] for i in range(len(self.Etiplist))]
        self.splnum = len(self.Etiplist)
    def plot_activation_pdf_Etip_pdf_Es(self,time_th,fnameNum=0,time_start=1):
        self.read_activation_E(time_th,fnameNum,time_start)
        print(self.Etiplist[:100])
        # self.genpdf_printf_Eb(self.barrier,"standard_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),self.bins)
        self.genpdf_printf_E(np.array(self.Etiplist)/self.num,"tip_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),int(self.bins))
        self.genpdf_printf_E(np.array(self.Eslist)/self.num,"saddle_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),int(self.bins))
        # self.genpdf_printf_Eb(self.barrier_iner_lowsen,"iner_lowsen_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),int(self.bins/5))
        # self.genpdf_printf_Eb(self.barrier_free_lowsen,"free_lowsen_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),int(self.bins/5))
    def plot_activation_pdf2d_Etip_Es(self,time_th,fnameNum=0,time_start=1):
        self.read_activation_E(time_th,fnameNum,time_start)
        plot_hist2d(np.array(self.Etiplist)/self.num,np.array(self.Eslist)/self.num
            ,"pdf(dynamics) splnum={spl} bins={bins}".format(spl="%.1le"%self.splnum,bins=self.bins)
            ,xlabel="Etip",ylabel="Es"
            ,func=self.__imgpath,bins=self.bins)
    def plot_activation_pdf2d_Eb_Es(self,time_th,fnameNum=0,time_start=1):
        self.read_activation_E(time_th,fnameNum,time_start)
        plot_hist2d(np.array(self.Eblist)/self.num,np.array(self.Eslist)/self.num
            ,"pdf(dynamics) splnum={spl} bins={bins}".format(spl="%.1le"%self.splnum,bins=self.bins)
            ,xlabel="Eb",ylabel="Es"
            ,func=self.__imgpath,bins=self.bins)
    def plot_activation_pdf2d_Eb_Etip(self,time_th,fnameNum=0,time_start=1):
        self.read_activation_E(time_th,fnameNum,time_start)
        plot_hist2d(np.array(self.Eblist)/self.num,np.array(self.Etiplist)/self.num
            ,"pdf(dynamics) splnum={spl} bins={bins}".format(spl="%.1le"%self.splnum,bins=self.bins)
            ,xlabel="Eb",ylabel="Etip"
            ,func=self.__imgpath,bins=self.bins)
    def plot_activation_pdf_E(self,time_th,fnameNum=0,time_start=1):
        self.read_activation_E(time_th,fnameNum,time_start)
        print(self.Etiplist[:100])
        # self.genpdf_printf_Eb(self.barrier,"standard_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),self.bins)
        self.genpdf_printf_E(np.array(self.Etiplist)/self.num,"tip_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),int(self.bins))
        self.genpdf_printf_E(np.array(self.Eslist)/self.num,"saddle_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),int(self.bins))
        plot_hist2d(np.array(self.Etiplist)/self.num,np.array(self.Eslist)/self.num
            ,"pdf(dynamics) spls={spl} bins={bins} r={r}".format(spl="%.1le"%self.splnum,bins=self.bins,r="%.0le"%self.fnum)
            ,xlabel="Etip",ylabel="Es"
            ,func=self.__imgpath,bins=self.bins)
        plot_hist2d(np.array(self.Eblist)/self.num,np.array(self.Eslist)/self.num
            ,"pdf(dynamics) spls={spl} bins={bins} r={r}".format(spl="%.1le"%self.splnum,bins=self.bins,r="%.0le"%self.fnum)
            ,xlabel="Eb",ylabel="Es"
            ,func=self.__imgpath,bins=self.bins)

#pdf activation:
    def read_activation(self,time_th,fnameNum=0,time_start=1):
        fnames = os.listdir(self.dirActivation)
        if fnameNum:
            fnames = fnames[:int(fnameNum)]
        self.fnum = len(fnames)
        self.taulist = []
        self.tau0list = []
        self.tauslist = []
        self.barrier = []
        self.barrier_lowsen = []
        self.barrier_iner_lowsen = []
        self.barrier_free_lowsen = []

        self.splsum = 0
        for fname in fnames:
            media = read_data_from_txt(self.dirActivation + fname, [0,8,9,10,11,12,13,2,3])   #0 i,1 last_sen,2 last_basin_tipen,3 sen
                                                                                #,4 last_sdec,5 last_basin_tip,6 sdec
                                                                                #,7 last_lowest_sdec
                                                                                #,8 basin_time,9 hangout_time,10 trapping_time  
                                                                                #,11 iner_en,12 now_eff_free_en,13 last_lowest_sen
                                                                                #,14 confnum             
            media_time = media[0]
            for i in range(len(media_time)):
                if media_time[i] > time_start:
                    break
            ist = i
            for i in range(ist,len(media_time)):
                if media_time[i] > time_th:
                    break
            ith = i
            self.splsum += ith - ist
            media_tau0 = media[1][ist:ith]
            media_taus = media[2][ist:ith]
            media_tau = media[3][ist:ith]
            media_barrier = [media[8][i] - media[7][i] for i in range(ist, ith)]
            media_barrier_lowsen = [media[6][i] - media[7][i] for i in range(ist, ith)]
            media_barrier_IL = [media[6][i] - media[4][i] for i in range(ist, ith)]
            media_barrier_FL = [media[6][i] - media[5][i] for i in range(ist, ith)]
            self.barrier.extend(media_barrier)
            self.barrier_lowsen.extend(media_barrier_lowsen)
            self.barrier_iner_lowsen.extend(media_barrier_IL)
            self.barrier_free_lowsen.extend(media_barrier_FL)
            self.taulist.extend(media_tau)
            self.tau0list.extend(media_tau0)
            self.tauslist.extend(media_taus)
    def plot_activation(self,time_th,fnameNum=0,time_start=1,bins_trct=100):
        self.read_activation(time_th,fnameNum,time_start)
        self.gendos_printf_tau_genfunc(self.taulist, self.splsum, bins_trct, "mc_taulist_beforeThermal_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),self.bins)
        self.gendos_printf_tau_genfunc(self.tau0list, self.splsum, bins_trct, "mc_tau0list_beforeThermal_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),self.bins)
        self.gendos_printf_tau_genfunc(self.tauslist, self.splsum, 20, "mc_tauslist_beforeThermal_timeh={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),20)
        self.genpdf_printf_Eb(self.barrier,"standard_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),self.bins)
        self.genpdf_printf_Eb(self.barrier_lowsen,"lowsem_standard_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),int(self.bins/5))
        self.genpdf_printf_Eb(self.barrier_iner_lowsen,"iner_lowsen_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),int(self.bins/5))
        self.genpdf_printf_Eb(self.barrier_free_lowsen,"free_lowsen_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),int(self.bins/5))
        # self.genpdf_printf_tau_fromDyEb_DyEbTau(self.barrier,"standard_time={s}-{e}".format(s="%.1le"%time_start,e="%.1le"%time_th), self.bins)

#pdf activation_simple:
    def read_activation_simple(self,time_th,fnameNum=0,time_start=1):
        fnames = os.listdir(self.dirActivationSimple)
        if fnameNum:
            fnames = fnames[:int(fnameNum)]
        self.fnum = len(fnames)
        self.tau0list = []
        self.taulist = []
        self.barrier_lowsen = []

        self.splsum = 0
        for fname in fnames:
            media = read_data_from_txt(self.dirActivationSimple + fname, [0,1,2,3])   #0 i, 1 tau0, 2 low_barrier, 3 tau
            media_time = media[0]
            for i in range(len(media_time)):
                if media_time[i] > time_start:
                    break
            ist = i
            for i in range(ist,len(media_time)):
                if media_time[i] > time_th:
                    break
            ith = i
            self.splsum += ith - ist
            media_tau0 = media[1][ist:ith]
            media_barrier_lowsen = media[2][ist:ith]
            media_tau = media[3][ist:ith]
            self.barrier_lowsen.extend(media_barrier_lowsen)
            self.tau0list.extend(media_tau0)
            self.taulist.extend(media_tau)
    def plot_activation_simple(self,time_th,fnameNum=0,time_start=1,bins_trct=100):
        self.read_activation_simple(time_th,fnameNum,time_start)
        self.gendos_printf_tau_genfunc(self.tau0list, self.splsum, bins_trct, "mc_tau0list_beforeThermal_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),self.bins)
        self.gendos_printf_tau_genfunc(self.taulist, self.splsum, bins_trct, "mc_taulist_beforeThermal_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),self.bins)
        self.genpdf_printf_Eb(self.barrier_lowsen,"lowsen_standard_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),int(self.bins/5))
        self.__hist2d_taulog("Eblowsen"
                        ,"tau0"
                        ,"prob density_ylog"
                        ,self.barrier_lowsen
                        ,self.tau0list
                        ,"rem{a}_T={b}_t={s}-{e}_spl={spl}".format(a=self.num,b="%.2f"%self.temperature,e="%.1le"%time_th,s="%.1le"%time_start,spl="%.1le"%len(self.tau0list))
                        ,"Eblowsen-tau0_prob_ylog"
                        ,self.bins)#
        self.__hist2d_taulog("Eblowsen"
                        ,"tau"
                        ,"prob density_ylog"
                        ,self.barrier_lowsen
                        ,self.taulist
                        ,"rem{a}_T={b}_t={s}-{e}_spl={spl}".format(a=self.num,b="%.2f"%self.temperature,e="%.1le"%time_th,s="%.1le"%time_start,spl="%.1le"%len(self.tau0list))
                        ,"Eblowsen-tau_prob_ylog"
                        ,self.bins)#
        # self.genpdf_printf_tau_fromDyEb_DyEbTau(self.barrier,"standard_time={s}-{e}".format(s="%.1le"%time_start,e="%.1le"%time_th), self.bins)
    def plot_activation_tauEb_simple(self,time_th,fnameNum=0,time_start=1,bins_trct=100):
        self.read_activation_simple(time_th,fnameNum,time_start)
        self.gendos_printf_tau_genfunc(self.tau0list, self.splsum, bins_trct, "mc_tau0list_beforeThermal_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),self.bins)
        self.genpdf_printf_Eb(self.barrier_lowsen,"lowsen_standard_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),int(self.bins/5))
        # self.genpdf_printf_tau_fromDyEb_DyEbTau(self.barrier,"standard_time={s}-{e}".format(s="%.1le"%time_start,e="%.1le"%time_th), self.bins)
    def plot_arrhenius_simple(self,time_th,fnameNum=0,time_start=1,bins_trct=100):
        self.read_activation_simple(time_th,fnameNum,time_start)
        self.__hist2d_taulog2("Eblowsen"
                        ,"tau0"
                        ,"prob density_ylog"
                        ,self.barrier_lowsen
                        ,self.tau0list
                        ,"rem{a}_T={b}_t={s}-{e}_spl={spl}".format(a=self.num,b="%.2f"%self.temperature,e="%.1le"%time_th,s="%.1le"%time_start,spl="%.1le"%len(self.tau0list))
                        ,"Eblowsen-tau0_prob_ylog_2".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum)#_time={s}-{e}_replica={r}
                        ,self.bins)#
        self.__hist2d_taulog("Eblowsen"
                        ,"tau0"
                        ,"prob density_ylog"
                        ,self.barrier_lowsen
                        ,self.tau0list
                        ,"rem{a}_T={b}_t={s}-{e}_spl={spl}".format(a=self.num,b="%.2f"%self.temperature,e="%.1le"%time_th,s="%.1le"%time_start,spl="%.1le"%len(self.tau0list))
                        ,"Eblowsen-tau0_prob_ylog".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum)#_time={s}-{e}_replica={r}
                        ,self.bins)#
#pdf-tau activaiton_simple 1by1
    def get_minmax_tau_activation_simple(self,time_th,fnameNum=0,time_start=1):
        fnames = os.listdir(self.dirActivationSimple)
        if fnameNum:
            fnames = fnames[:int(fnameNum)]
        self.fnum = len(fnames)
        tau0min = 1e15
        tau0max = -1e15

        self.splsum = 0
        for fname in fnames:
            media = read_data_from_txt(self.dirActivationSimple + fname, [0,1,2,3])   #0 i, 1 tau0, 2 low_barrier, 3 tau
            media_time = media[0]
            for i in range(len(media_time)):
                if media_time[i] > time_start:
                    break
            ist = i
            for i in range(ist,len(media_time)):
                if media_time[i] > time_th:
                    break
            ith = i
            self.splsum += ith - ist
            media_tau0 = media[1][ist:ith]
            # media_barrier_lowsen = media[2][ist:ith]
            # media_tau = media[3][ist:ith]
            tau0min = min(min(media_tau0),tau0min)
            tau0max = max(max(media_tau0),tau0max)
        return [tau0min,tau0max]
    def read_activation_simple_taulists_1by1(self,time_th,fnameNum=0,time_start=1,bins_trct=100):
        fnames = os.listdir(self.dirActivationSimple)
        if fnameNum:
            fnames = fnames[:int(fnameNum)]
        self.fnum = len(fnames)
        self.tau0list = []
        self.pdftau0 = np.array([0.0 for i in range(self.bins + bins_trct - 1)])
        mm = self.get_minmax_tau_activation_simple(time_th,fnameNum,time_start)
        self.splnum = 0
        i=0
        for fname in fnames:
            media = read_data_from_txt(self.dirActivationSimple + fname, [0,1,2,3])   #0 i, 1 tau0, 2 low_barrier, 3 tau
            media_time = media[0]
            for i in range(len(media_time)):
                if media_time[i] > time_start:
                    break
            ist = i
            for i in range(ist,len(media_time)):
                if media_time[i] > time_th:
                    break
            ith = i
            self.splsum += ith - ist
            media_tau0 = media[1][ist:ith]
            # media_barrier_lowsen = media[2][ist:ith]
            # media_tau = media[3][ist:ith]
            lsptau0, pdftau0, widthtau0 = gen_doslist_mixed_linearFine(media_tau0, bins_trct, self.bins,Print=0, mm=mm)
            self.pdftau0 = self.pdftau0 + np.array(pdftau0)/self.fnum
            self.splnum += len(media_tau0)
            print(i,end="-",flush=True)
        self.lsptau0 = lsptau0
        self.widthtau0 = widthtau0
    def plot_activation_simple_taulists_1by1(self,time_th,fnameNum=0,time_start=1,bins_trct=100):
        self.read_activation_simple_taulists_1by1(time_th,fnameNum,time_start)
        fname_tau0 = self.dir + "pdf-tau_1by1_Tsplnum={spl}_mc_tau0list_beforeThermal_time={s}-{e}_replica={r}_rem{a}_T={b}_timemax={c}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,spl="%.1le"%self.splnum,s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum)
        dump_data(fname_tau0,[self.lsptau0,self.pdftau0,self.widthtau0])
        self.__plot_xylog_tau_genfunc("tau_1by1_fitfunc_Tsplnum={spl}_mc_tau0list_beforeThermal_time={s}-{e}_replica={r}_rem{a}_T={b}_timemax={c}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,spl="%.1le"%self.splnum,s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum)
            ,"pdf"
            ,self.lsptau0
            ,self.pdftau0
            ,"rem{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature))
#pdf activation 1by1:
    def get_minmax_activation_simple(self,time_th,fnameNum=0,time_start=1):
        fnames = os.listdir(self.dirActivationSimple)
        if fnameNum:
            fnames = fnames[:int(fnameNum)]
        self.fnum = len(fnames)
        tau0min = 1e15
        tau0max = -1e15
        taumin = 1e15
        taumax = -1e15
        ebmin = 1e15
        ebmax = -1e15

        self.splsum = 0
        for fname in fnames:
            media = read_data_from_txt(self.dirActivationSimple + fname, [0,1,2,3])   #0 i, 1 tau0, 2 low_barrier, 3 tau
            media_time = media[0]
            for i in range(len(media_time)):
                if media_time[i] > time_start:
                    break
            ist = i
            for i in range(ist,len(media_time)):
                if media_time[i] > time_th:
                    break
            ith = i
            self.splsum += ith - ist
            media_tau0 = media[1][ist:ith]
            media_barrier_lowsen = media[2][ist:ith]
            media_tau = media[3][ist:ith]
            tau0min = min(min(media_tau0),tau0min)
            tau0max = max(max(media_tau0),tau0max)
            taumin = min(min(media_tau),taumin)
            taumax = max(max(media_tau),taumax)
            ebmin = min(min(media_barrier_lowsen),ebmin)
            ebmax = max(max(media_barrier_lowsen),ebmax)
        return [[tau0min,tau0max], [taumin,taumax], [ebmin,ebmax]]
    def read_activation_simple_1by1(self,time_th,fnameNum=0,time_start=1,bins_trct=100):
        fnames = os.listdir(self.dirActivationSimple)
        if fnameNum:
            fnames = fnames[:int(fnameNum)]
        self.fnum = len(fnames)
        self.pdftau0 = np.array([0.0 for i in range(self.bins + bins_trct - 1)])
        self.pdftau = np.array([0.0 for i in range(self.bins + bins_trct - 1)])
        self.pdfeb = np.array([0.0 for i in range(self.bins)])

        mms = self.get_minmax_activation_simple(time_th,fnameNum,time_start)
        self.splnum = 0
        i=0
        for fname in fnames:
            media = read_data_from_txt(self.dirActivationSimple + fname, [0,1,2,3])   #0 i, 1 tau0, 2 low_barrier, 3 tau
            media_time = media[0]
            for i in range(len(media_time)):
                if media_time[i] > time_start:
                    break
            ist = i
            for i in range(ist,len(media_time)):
                if media_time[i] > time_th:
                    break
            ith = i
            self.splsum += ith - ist
            media_tau0 = media[1][ist:ith]
            media_barrier_lowsen = media[2][ist:ith]
            media_tau = media[3][ist:ith]
            lsptau0, pdftau0, widthtau0 = gen_doslist_mixed_linearFine(media_tau0, bins_trct, self.bins,Print=0, mm=mms[0])
            lsptau, pdftau, widthtau = gen_doslist_mixed_linearFine(media_tau, bins_trct, int(self.bins/3),Print=0, mm=mms[1])
            lspeb, pdfeb= gen_doslist(media_barrier_lowsen, self.bins, mm=mms[2])

            self.pdftau0 = self.pdftau0 + np.array(pdftau0)/self.fnum
            self.pdftau = self.pdftau + np.array(pdftau)/self.fnum
            self.pdfeb = self.pdfeb + np.array(pdfeb)/self.fnum

            self.splnum += len(media_tau0)
            print(i,end="-",flush=True)

        self.lsptau0 = lsptau0
        self.widthtau0 = widthtau0
        self.lsptau = lsptau
        self.widthtau = widthtau
        self.lspeb = lspeb
    def plot_activation_simple_1by1(self,time_th,fnameNum=0,time_start=1,bins_trct=100):
        self.read_activation_simple_1by1(time_th,fnameNum,time_start)
        fname_tau0 = self.dir + "pdf-tau_1by1_Tsplnum={spl}_mc_tau0list_beforeThermal_time={s}-{e}_replica={r}_rem{a}_T={b}_timemax={c}_bins={bins}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,spl="%.1le"%self.splnum,s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum,bins="%.1le"%self.bins)
        dump_data(fname_tau0,[self.lsptau0,self.pdftau0,self.widthtau0])
        self.__plot_xylog_tau_genfunc("tau_1by1_fitfunc_Tsplnum={spl}_mc_tau0list_beforeThermal_time={s}-{e}_replica={r}_rem{a}_T={b}_timemax={c}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,spl="%.1le"%self.splnum,s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum)
            ,"pdf"
            ,self.lsptau0
            ,self.pdftau0
            ,"rem{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature))
        fname_tau = self.dir + "pdf-tau_1by1_Tsplnum={spl}_mc_taulist_beforeThermal_time={s}-{e}_replica={r}_rem{a}_T={b}_timemax={c}_bins={bins}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,spl="%.1le"%self.splnum,s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum,bins="%.1le"%self.bins)
        dump_data(fname_tau,[self.lsptau,self.pdftau,self.widthtau])
        self.__plot_xylog_tau_genfunc("tau_1by1_fitfunc_Tsplnum={spl}_mc_taulist_beforeThermal_time={s}-{e}_replica={r}_rem{a}_T={b}_timemax={c}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,spl="%.1le"%self.splnum,s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum)
            ,"pdf"
            ,self.lsptau
            ,self.pdftau
            ,"rem{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature))
        fname_eb = self.dir + "pdf-eb_1by1_Tsplnum={spl}_mc_taulist_beforeThermal_time={s}-{e}_replica={r}_rem{a}_T={b}_timemax={c}_bins={bins}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,spl="%.1le"%self.splnum,s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum,bins="%.1le"%(self.bins/3))
        dump_data(fname_eb,[self.lspeb,self.pdfeb])
        plot_ylog([(self.lspeb,self.pdfeb)]
                ,["pdf-eb_1by1_Tsplnum={spl}_mc_taulist_beforeThermal_time={s}-{e}_replica={r}_rem{a}_T={b}_timemax={c}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,spl="%.1le"%self.splnum,s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum)],
                func = self.__imgpath)

#pdf activation_Eb:
    def read_activation_Eb(self,time_th,fnameNum=0,time_start=1):
        fnames = os.listdir(self.dirActivationEb)
        if fnameNum:
            fnames = fnames[:int(fnameNum)]
        self.fnum = len(fnames)
        self.barrier_lowsen = []

        self.splsum = 0
        for fname in fnames:
            media = read_data_from_txt(self.dirActivationEb + fname, [0,1])   #0 i, 1 low_barrier
            media_time = media[0]
            for i in range(len(media_time)):
                if media_time[i] > time_start:
                    break
            ist = i
            for i in range(ist,len(media_time)):
                if media_time[i] > time_th:
                    break
            ith = i
            self.splsum += ith - ist
            media_barrier_lowsen = media[1][ist:ith]
            self.barrier_lowsen.extend(media_barrier_lowsen)
    def plot_activation_Eb(self,time_th,fnameNum=0,time_start=1,bins_trct=100):
        self.read_activation_Eb(time_th,fnameNum,time_start)
        self.genpdf_printf_Eb(self.barrier_lowsen,"lowsen_standard_besidesEndTime_time={s}-{e}_replica={r}".format(s="%.1le"%time_start,e="%.1le"%time_th,r="%.0le"%self.fnum),int(self.bins/5))

                
#basin connectivity from activation-t:
    def read_activation_for_basin_connect(self,time_th,time_start=1):
        fnamesAct = os.listdir(self.dirActivation)
        fnamesB = os.listdir(self.dirRealBarrier)
        fnamesB = [fnameB for fnameB in fnamesB if fnameB[-3:]=="bar"]
        self.rdn = random.randint(0,len(fnamesAct))
        fnameAct = fnamesAct[self.rdn]
        fnameB = fnamesB[self.rdn]
        self.fnum = 1

        self.splsum = 0
        declist = []
        with open(self.dirRealBarrier + fnameB,"r",encoding="utf-8") as fp:
            fp.readline()
            while 1:
                linedata = fp.readline()
                if linedata.strip()=="":
                    break
                declist.append(self.conf_to_dec(str(linedata.split()[1])))
        self.barrier = [[] for i in range(len(declist))]
        self.barrier_iner = [[] for i in range(len(declist))]
        self.barrier_free = [[] for i in range(len(declist))]
        self.index = [[] for i in range(len(declist))]
        self.index_iner = [[] for i in range(len(declist))]
        self.index_free = [[] for i in range(len(declist))]
        
        media = read_data_from_txt(self.dirActivation + fnameAct, [0,8,9,10,11,12,13,2,3,5])   #0 i,1 last_sen,2 last_basin_tipen,3 sen
                                                                            #,4 last_sdec,5 last_basin_tip,6 sdec
                                                                            #,7 last_lowest_sdec
                                                                            #,8 basin_time,9 hangout_time,10 trapping_time  
                                                                            #,11 iner_en,12 now_eff_free_en,13 last_lowest_sen
                                                                            #,14 confnum             
        media_time = media[0]
        for i in range(len(media_time)):
            if media_time[i] > time_start:
                break
        ist = i
        for i in range(ist,len(media_time)):
            if media_time[i] > time_th:
                break
        ith = i
        self.splsum += ith - ist
        media_barrier = [media[8][i] - media[7][i] for i in range(ist, ith)]
        media_barrier_IL = [media[8][i] - media[4][i] for i in range(ist, ith)]
        media_barrier_FL = [media[8][i] - media[5][i] for i in range(ist, ith)]
        media_etip = media[7][ist:ith]
        media_tip = media[9][ist:ith]

        self.etip = [[media_etip[j+1] for j in range(len(media_etip)-1) if int(media_tip[j])==declist[i]] for i in range(len(declist))]
        self.barrier = [[media_barrier[j] for j in range(len(media_barrier)) if int(media_tip[j])==declist[i]] for i in range(len(declist))]
        self.barrier_iner = [[media_barrier_IL[j] for j in range(len(media_barrier_IL)) if int(media_tip[j])==declist[i]] for i in range(len(declist))]
        self.barrier_free = [[media_barrier_FL[j] for j in range(len(media_barrier_FL)) if int(media_tip[j])==declist[i]] for i in range(len(declist))]
        dbg = [1 for j in range(len(media_tip)) if int(media_tip[j])==declist[0]]
        print(declist)
        print(len(dbg))
        # print(self.barrier)
        # print(fnameAct,fnameB)
        # print(media_tip)
        # print(declist[0])
        # print(ist,ith,self.splsum)
        # self.index = [[i for] for i in range(len(declist))]
        # self.index_iner = [[] for i in range(len(declist))]
        # self.index_free = [[] for i in range(len(declist))]

        # self.barrier.extend(media_barrier)
        # self.barrier_iner.extend(media_barrier_IL)
        # self.barrier_free.extend(media_barrier_FL)
        return [self.barrier,self.barrier_iner,self.barrier_free,self.etip]
    def plot_hist_basin_connect(self,time_th,time_start=1):
        lists = self.read_activation_for_basin_connect(time_th,time_start)
        mkdir(self.dir + "plot/" +"basinConnect{}".format(self.rdn))
        mkdir(self.dir + "plot/" +"basinConnect{}/pdf".format(self.rdn))
        labels = ["barrier","barrier_inerEn","barrier_freeEn","etip"]
        
        for i in range(len(lists)):
            for j in range(len(self.barrier)):
                plot_hist([lists[i][j]],[labels[i]+str(j)],rdn=self.rdn,func=self.__imgpath_basin_connect)
                plot_hist_ylog([lists[i][j]],[labels[i]+str(j)],rdn=self.rdn,func=self.__imgpath_basin_connect)
    def conf_to_dec(self,conf):
        dec = 0
        num = len(conf)
        for i in range(len(conf)):
            if conf[num-1 - i] == "+":
                dec += 2**i
        return int(dec)

    def read_activation_for_basin_connect_of_big_lowbarriers(self,time_th,time_start=1,value=0.8,fnameNum=0):
        fnames = os.listdir(self.dirActivation)
        value = self.num * value
        if fnameNum:
            fnames = fnames[:int(fnameNum)]
        self.barrier = []
        self.confnum = []
        self.low_barrier = []
        for fname in fnames:
            media = read_data_from_txt(self.dirActivation + fname, [0,2,3,5,13,14])   #0 i,1 last_sen,2 last_basin_tipen,3 sen
                                                                                #,4 last_sdec,5 last_basin_tip,6 sdec
                                                                                #,7 last_lowest_sdec
                                                                                #,8 basin_time,9 hangout_time,10 trapping_time  
                                                                                #,11 iner_en,12 now_eff_free_en,13 last_lowest_sen
                                                                                #,14 confnum             
            media_time = media[0]
            for i in range(len(media_time)):
                if media_time[i] > time_start:
                    break
            ist = i
            for i in range(ist,len(media_time)):
                if media_time[i] > time_th:
                    break
            ith = i
            # self.splsum += ith - ist
            media_barrier = [media[2][i] - media[1][i] for i in range(ist, ith)]
            media_low_barrier = [media[4][i] - media[1][i] for i in range(ist, ith)]
            media_tip = media[3][ist:ith]
            media_confnum = media[-1][ist:ith]
            media_needed_idx = [i for i in range(len(media_low_barrier)) if media_low_barrier[i] >= value]
            media_needed_barrier = np.array(media_barrier)[media_needed_idx].tolist()
            media_needed_lowbarrier = np.array(media_low_barrier)[media_needed_idx].tolist()
            media_needed_tip = np.array(media_tip)[media_needed_idx].tolist()
            media_tip_set = [media_tip[i] for i in range(len(media_low_barrier)) if media_low_barrier[i] >= value]
            media_tip_set = list(set(media_tip_set))
            barrier = [[media_needed_barrier[j] for j in range(len(media_needed_barrier)) if int(media_needed_tip[j])==int(tip)] for tip in media_tip_set ]
            confnum = [media_confnum[media_barrier.index(barrier[i][0])] for i in range(len(barrier))]
            low_barrier = [media_low_barrier[media_barrier.index(barrier[i][0])] for i in range(len(barrier))]
            self.barrier.extend(barrier)
            self.confnum.extend(confnum)
            self.low_barrier.extend(low_barrier)
        print(len(self.barrier))
    def plot_hist_basin_connect_of_big_lowbarriers(self,time_th,time_start=1,value=0.8,fnameNum=0):
        self.read_activation_for_basin_connect_of_big_lowbarriers(time_th,time_start,value,fnameNum)
        mkdir(self.dir + "plot/" +"basinConnect_big_lowbarriers")
        mkdir(self.dir + "plot/" +"basinConnect_big_lowbarriers/pdf")
        label = ["barrier_loweb={eb}_confnum={cn}".format(eb="%.4lf"%self.low_barrier[i]
                ,cn=self.confnum[i]) for i in range(len(self.low_barrier))]
        
        for i in range(len(label)):
            plot_hist([self.barrier[i]],[label[i]],func=self.__imgpath_basin_connect_of_big_lowbarriers)
            plot_hist_ylog([self.barrier[i]],[label[i]],func=self.__imgpath_basin_connect_of_big_lowbarriers)


#pdf-Eb-tau0: 
    def read_arrhenius(self,time_th,fnameNum=0,time_start=1):
        fnames = os.listdir(self.dirActivation)
        if fnameNum:
            fnames = fnames[:int(fnameNum)]
        self.fnum = len(fnames)
        self.barrier = []
        self.barrier_lowsen = []
        self.barrier_iner_lowsen = []
        self.barrier_free_lowsen = []
        self.tau0list = []

        self.splsum = 0
        for fname in fnames:
            media = read_data_from_txt(self.dirActivation + fname, [0,2,3,8,11,12,13])   #0 i,1 last_sen,2 last_basin_tipen,3 sen
                                                                                #,4 last_sdec,5 last_basin_tip,6 sdec
                                                                                #,7 last_lowest_sdec
                                                                                #,8 basin_time,9 hangout_time,10 trapping_time  
                                                                                #,11 iner_en,12 now_eff_free_en,13 last_lowest_sen
                                                                                #,14 confnum          
            media_time = media[0]
            for i in range(len(media_time)):
                if media_time[i] > time_start:
                    break
            ist = i
            for i in range(ist,len(media_time)):
                if media_time[i] > time_th:
                    break
            ith = i
            self.splsum += ith - ist
            media_barrier = [media[2][i] - media[1][i] for i in range(ist, ith)]
            media_barrier_lowsen = [media[6][i] - media[1][i] for i in range(ist, ith)]
            media_barrier_IL = [media[6][i] - media[4][i] for i in range(ist, ith)]
            media_barrier_FL = [media[6][i] - media[5][i] for i in range(ist, ith)]
            media_tau0 = media[3][ist:ith]
            self.barrier.extend(media_barrier)
            self.barrier_lowsen.extend(media_barrier_lowsen)
            self.barrier_iner_lowsen.extend(media_barrier_IL)
            self.barrier_free_lowsen.extend(media_barrier_FL)
            self.tau0list.extend(media_tau0)
    def plot_arrhenius(self,time_th,fnameNum=0,time_start=1):
        self.read_arrhenius(time_th,fnameNum,time_start)
        self.__hist2d_taulog("Eb1_t={s}-{e}_spl={spl}".format(e="%.1le"%time_th,s="%.1le"%time_start,spl="%.1le"%len(self.barrier))
                        ,"tau0"
                        ,"prob density_ylog"
                        ,self.barrier
                        ,self.tau0list
                        ,"rem{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature)
                        ,"Eb1-tau0_prob_ylog"
                        ,self.bins)#
        self.__hist2d_taulog("Eb_lowsen_t={s}-{e}_spl={spl}".format(e="%.1le"%time_th,s="%.1le"%time_start,spl="%.1le"%len(self.barrier))
                        ,"tau0"
                        ,"prob density_ylog"
                        ,self.barrier_lowsen
                        ,self.tau0list
                        ,"rem{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature)
                        ,"Eb_lowsen-tau0_prob_ylog"
                        ,self.bins)#
        self.__hist2d_taulog("EbIL_t={s}-{e}_spl={spl}".format(e="%.1le"%time_th,s="%.1le"%time_start,spl="%.1le"%len(self.barrier))
                        ,"tau0"
                        ,"prob density_ylog"
                        ,self.barrier_iner_lowsen
                        ,self.tau0list
                        ,"rem{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature)
                        ,"EbIL-tau0_prob_ylog"
                        ,self.bins)#
        self.__hist2d_taulog("EbFL_t={s}-{e}_spl={spl}".format(e="%.1le"%time_th,s="%.1le"%time_start,spl="%.1le"%len(self.barrier))
                        ,"tau0"
                        ,"prob density_ylog"
                        ,self.barrier_free_lowsen
                        ,self.tau0list
                        ,"rem{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature)
                        ,"EbFL-tau0_prob_ylog"
                        ,self.bins)#
    
    
    def read_obslists_nextBasin(self,samplenum,specific_time,fnameNum=0):
        dirNextBasin = self.dirNextBasin + "splNum={spl}_t={t}/".format(spl="%.0e"%samplenum,t="%.0e"%specific_time)
        fnames = os.listdir(dirNextBasin)
        if fnameNum:
            fnames = fnames[:int(fnameNum)]
        self.replica = len(fnames)
        self.barrier = []
        self.barrier_lowsen = []
        self.tau0list = []
        self.tauslist = []
        self.taulist = []
        for fname in fnames:
            media = read_data_from_txt(dirNextBasin + fname, [1,2,6,7,8,11])   # last_sen, last_basin_tipen, sen,
            # print(fname)
            self.barrier.extend([media[1][i] - media[0][i] for i in range(len(media[0]))])#  last_sdec, last_basin_tip, sdec, 
            self.barrier_lowsen.extend([media[5][i] - media[0][i] for i in range(len(media[0]))])
            self.tau0list.extend(media[2])                      #  basin_time, hangout_time, trapping_time
            self.tauslist.extend(media[3])                      #  iner_en, now_eff_free_en, last_lowest_sen
            self.taulist.extend(media[4])
        self.splsum = int(samplenum) * len(fnames)
    def plot_gen_obslists_nextBasin(self,samplenum,specific_time,fnameNum=0, bins_trct=100):
        self.read_obslists_nextBasin(samplenum,specific_time,fnameNum)
        self.gendos_printf_tau_genfunc(self.taulist, self.splsum, bins_trct, "MCnextBasin_taulist_time={t}_replica={r}".format(t="%.0e"%specific_time,r="%.0e"%self.replica),self.bins)
        self.gendos_printf_tau_genfunc(self.tau0list, self.splsum, bins_trct, "MCnextBasin_tau0list_time={t}_replica={r}".format(t="%.0e"%specific_time,r="%.0e"%self.replica),self.bins)
        self.gendos_printf_tau_genfunc(self.tauslist, self.splsum, 20, "MCnextBasin_tauslist_time={t}_replica={r}".format(t="%.0e"%specific_time,r="%.0e"%self.replica),20)
        self.genpdf_printf_Eb(self.barrier,"Eb_sen_MCnextBasin_time={t}_replica={r}".format(t="%.0e"%specific_time,r="%.0e"%self.replica),int(self.bins/3))
        self.genpdf_printf_Eb(self.barrier_lowsen,"Eb_lowsen_MCnextBasin_time={t}_replica={r}".format(t="%.0e"%specific_time,r="%.0e"%self.replica),int(self.bins/3))
    def read_averageObs_nextBasin(self,samplenum,fnameNum=0):
        self.timelist = []
        dirnames = os.listdir(self.dirNextBasin)
        realdirnames = []
        for name in dirnames:
            t = float(name.split("_")[1][2:])
            if not int(t):
                t = 1.0
            if float(name.split("_")[0][7:]) - samplenum < 0.1:
                self.timelist.append(t)
                realdirnames.append(name)
        print(realdirnames)
        self.ebAvrg = np.array([0.0 for i in range(len(realdirnames))])
        self.etipAvrg = np.array([0.0 for i in range(len(realdirnames))])
        self.tau0Avrg = np.array([0.0 for i in range(len(realdirnames))])
        self.ebErr = np.array([0.0 for i in range(len(realdirnames))])
        self.etipErr = np.array([0.0 for i in range(len(realdirnames))])
        self.tau0Err = np.array([0.0 for i in range(len(realdirnames))])
        
        for i in range(len(realdirnames)):
            dirNextBasin = self.dirNextBasin + realdirnames[i] +"/"
            fnames = os.listdir(dirNextBasin)
            num = 0
            if fnameNum:
                fnames = fnames[:int(fnameNum)]
            for fname in fnames:
                media = read_data_from_txt(dirNextBasin + fname, [1,2,6,7,8,11]) 
                self.ebAvrg[i] += np.sum(np.array([media[5][j] - media[0][j] for j in range(len(media[0]))]))
                self.etipAvrg[i] += np.sum(np.array(media[0]))
                self.tau0Avrg[i] += np.sum(np.array(media[2]))
                num += len(media[0])
            self.ebAvrg[i] = self.ebAvrg[i]/num
            self.etipAvrg[i] = self.etipAvrg[i]/num
            self.tau0Avrg[i] = self.tau0Avrg[i]/num
            for fname in fnames:
                media = read_data_from_txt(dirNextBasin + fname, [1,2,6,7,8,11]) 
                self.ebErr[i] += np.sum(np.array([(media[5][j] - media[0][j] - self.ebAvrg[i])**2/(num-1)/num for j in range(len(media[0]))]))
                self.etipErr[i] += np.sum(np.array([(media[0][j]-self.etipAvrg[i])**2/(num-1)/num for j in range(len(media[0]))]))
                self.tau0Err[i] += np.sum(np.array([(media[2][j]-self.tau0Avrg[i])**2/(num-1)/num for j in range(len(media[0]))]))
            self.ebErr[i] = np.sqrt(self.ebErr[i])*2
            self.etipErr[i] = np.sqrt(self.etipErr[i])*2
            self.tau0Err[i] = np.sqrt(self.tau0Err[i])*2

            self.replica = len(fnames)
        return [self.ebAvrg,self.etipAvrg,self.tau0Avrg]
    def plot_averageObs_nextBasin(self,samplenum,fnameNum=0):
        obslist = self.read_averageObs_nextBasin(samplenum,fnameNum)
        errlist = [self.ebErr,self.etipErr,self.tau0Err]
        fname = self.__txtpath("obs-t_nextBasin_1splnum={s}_replica={r}".format(s="%.0le"%samplenum,r="%.0le"%self.replica))
        dump_data(fname,obslist)
        des = ["eb","etip","tau0"]
        for i in range(len(obslist)):
            plot_xlog([(self.timelist,obslist[i])]
            ,[des[i]]
            ,yerrs=[errlist[i]]
            ,func=self.__imgpath)




    def __imgpath_barrier(self,datatype):
        path = self.dirbarrier +"plot/"+ "{datatype}_rem{num}_minh={minh}".format(
                datatype = datatype,
                num = self.num, temp="%.2lf"%self.temp,
                tm="%.0le"%self.timemax,minh="%.2le"%self.minh)
        pathpdf = self.dirbarrier +"plot/pdf/"+ "{datatype}_rem{num}_minh={minh}".format(
                datatype = datatype,
                num = self.num, temp="%.2lf"%self.temp,
                tm="%.0le"%self.timemax,minh="%.2le"%self.minh)
        png = path + ".png"
        pdf = pathpdf + ".pdf"
        return [png, pdf]
    def __imgpath(self,datatype):
        path = self.dir +"plot/"+ "{datatype}_rem{num}_T={temp}_timemax={tm}".format(
                datatype = datatype,
                num = self.num, temp="%.2lf"%self.temp,
                tm="%.0le"%self.timemax)
        pathpdf = self.dir +"plot/pdf/"+ "{datatype}_rem{num}_T={temp}_timemax={tm}".format(
                datatype = datatype,
                num = self.num, temp="%.2lf"%self.temp,
                tm="%.0le"%self.timemax)
        png = path + ".png"
        pdf = pathpdf + ".pdf"
        return [png, pdf]
    def __imgpath_basin_connect(self,datatypes):
        rnd = datatypes[0]
        datatype = datatypes[1]
        path = self.dir +"plot/basinConnect{}/".format(rnd) + "{datatype}_rem{num}_T={temp}_timemax={tm}".format(
                datatype = datatype,
                num = self.num, temp="%.2lf"%self.temp,
                tm="%.0le"%self.timemax)
        pathpdf = self.dir +"plot/basinConnect{}/pdf/".format(rnd) + "{datatype}_rem{num}_T={temp}_timemax={tm}".format(
                datatype = datatype,
                num = self.num, temp="%.2lf"%self.temp,
                tm="%.0le"%self.timemax)
        png = path + ".png"
        pdf = pathpdf + ".pdf"
        return [png, pdf]
    def __imgpath_basin_connect_of_big_lowbarriers(self,datatypes):
        rnd = datatypes[0]
        datatype = datatypes[1]
        path = self.dir +"plot/basinConnect_big_lowbarriers/" + "{datatype}_rem{num}_T={temp}_timemax={tm}".format(
                datatype = datatype,
                num = self.num, temp="%.2lf"%self.temp,
                tm="%.0le"%self.timemax)
        pathpdf = self.dir +"plot/basinConnect_big_lowbarriers/pdf/" + "{datatype}_rem{num}_T={temp}_timemax={tm}".format(
                datatype = datatype,
                num = self.num, temp="%.2lf"%self.temp,
                tm="%.0le"%self.timemax)
        png = path + ".png"
        pdf = pathpdf + ".pdf"
        return [png, pdf]
    def __txtpath(self,datatype):
        path = self.dir + "{datatype}_rem{num}_T={temp}_timemax={tm}".format(
                datatype = datatype,
                num = self.num, temp="%.2lf"%self.temp,
                tm="%.0le"%self.timemax)
        txt = path + ".t"
        return txt
    def treeArrhenius(self,lsp,dos):
        tau0 = 1
        lspout = [tau0*np.exp(ele/self.temp) for ele in lsp]
        dosout = [dos[i]*self.temp/lspout[i] for i in range(len(dos))]
        return lspout, dosout
    
    def gendos_printf_tau_genfunc(self, lists, splNum, bins_trct, dirname, bins=100):
        # lists = [ele for ele in lists if ele > 0]
        # if not lists == []:
        lsp, dos, width = gen_doslist_mixed_linearFine(lists, bins_trct, bins)
        del lists
        fname = self.dir + "pdf-tau_mixed_Tsplnum={spl}_{dir}_rem{a}_T={b}_timemax={c}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,spl="%.1le"%splNum,dir=dirname)
        dump_data(fname, [lsp, dos, width])
        self.__plot_xylog_tau_genfunc("tau_mixed_fitfunc_Tsplnum={spl}_{dir}_rem{a}_T={b}_timemax={c}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,spl="%.1le"%splNum,dir=dirname)
            ,"pdf"
            ,lsp
            ,dos
            ,"rem{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature))
        return lsp, dos, width
    def genpdf_printf_Eb(self, lists, des, bins=100):
        # lists = [ele for ele in lists if ele > 0]
        # if not lists == []:
        lsp, dos = gen_doslist(lists, bins)
        splNum = len(lists)
        del lists
        fname = self.dir + "pdf-Eb_{dir}_splnum={spl}_rem{a}_T={b}_timemax={c}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,spl="%.1le"%splNum,dir=des)
        dump_data(fname, [lsp, dos])
        plot_ylog([(lsp, dos)]
                ,["pdf-Eb_{dir}_splnum={spl}_rem{a}_T={b}_timemax={c}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,spl="%.1le"%splNum,dir=des)],
                func = self.__imgpath)
        return lsp, dos
    def genpdf_printf_E(self, lists, des, bins=20):
        # lists = [ele for ele in lists if ele > 0]
        # if not lists == []:
        lsp, dos = gen_doslist(lists, bins)
        splNum = len(lists)
        del lists
        fname = self.dir + "pdf-E_{dir}_splnum={spl}_rem{a}_T={b}_timemax={c}_bins={bins}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,spl="%.1le"%splNum,dir=des,bins=self.bins)
        dump_data(fname, [lsp, dos])
        plot([(lsp, dos)]
                ,["pdf-E_{dir}_splnum={spl}_rem{a}_T={b}_timemax={c}_bins={bins}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,spl="%.1le"%splNum,dir=des,bins=self.bins)],
                func = self.__imgpath)
        return lsp, dos
    def genpdf_printf_tau_fromDyEb_DyEbTau(self, lists, des, bins=100):
        lsp, dos = gen_doslist(lists, bins)
        lsptau, dostau = self.pdfTau_from_dynamical_EbTau(lsp, dos)
        splNum = len(lists)
        fname = self.dir + "pdf-tau_fromDyEb_DyEbTau_{dir}_splnum={spl}_rem{a}_T={b}_timemax={c}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,spl="%.1le"%splNum,dir=des)
        dump_data(fname, [lsptau, dostau])
        plot_xylog([(lsptau, dostau)]
                ,["pdf-tau_{dir}_fromDyEb_DyEbTau_splnum={spl}_rem{a}_T={b}_timemax={c}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,spl="%.1le"%splNum,dir=des)],
                func = self.__imgpath)

    def __plot_xylog_tau_genfunc(self,xlabel,ylabel,x,y,label):
        title = ylabel +"-"+ xlabel +"_xylog"
        img_path = ylabel +"-"+ xlabel +"_xylog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(title)
        plt.plot(x,y,label=label)
        x2 = linspace(10**(np.log10(max(x))*2/3), max(x),1000)
        y2 = x2**(-1.0*self.temperature*self.betac-1)*100
        self.a = 1.0*self.temperature*self.betac
        plt.plot(x2,y2,label="y=x^(-1-alpha)) alpha={}".format(1.0*self.temperature*self.betac),color="pink")
        # logx = np.log(x)
        # logy = np.log(y)
        para, pcov = curve_fit(self.__fit_tau_genfunc_no_a,np.array(x),np.array(y),maxfev = 10000000)
        fit_y = self.__fit_tau_genfunc_no_a(x, para[0], para[1]) #tau0, x, y
        # plt.plot(x,fit_y,label="Smirnov tau0={tau0} x={x} y={y}".format(tau0="%.2f"%para[0],x="%.2f"%para[1],y="%.2f"%para[2]),color="green")
        plt.plot(x,fit_y,label="Smirnov tau0={tau0} x={x} y={y}".format(tau0="%.2f"%para[0],x="%.3f"%self.a,y="%.2f"%para[1]),color="green")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path[0])
        plt.savefig(img_path[1])
        plt.clf()
        return fit_y   
    def __fit_tau_genfunc(self,tau,tau0,a,b):
        return ((1.0+a)/b)**(a/b) * tau0**a * b/math.gamma(a/b)  *  tau**(-a-1.0)  *  np.exp(-1.0*tau0**b*(1.0+a)/b * tau**(-b))
    def __fit_tau_genfunc_no_a(self,tau,tau0,b):
        a = 1.0*self.temperature*self.betac
        return ((1.0+a)/b)**(a/b) * tau0**a * b/math.gamma(a/b)  *  tau**(-a-1.0)  *  np.exp(-1.0*tau0**b*(1.0+a)/b * tau**(-b))
    def __plot_ylog_eb_fitfunc(self,xlabel,ylabel,x,y,label):
        title = ylabel +"-"+ xlabel +"_ylog"
        img_path = ylabel +"-"+ xlabel +"_ylog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale("log")
        plt.title(title)
        plt.plot(x,y,label=label)
        x2 = linspace(10**(np.log10(max(x))*2/3), max(x),1000)
        y2 = x2**(-1.0*self.temperature*self.betac-1)*100
        self.a = 1.0*self.temperature*self.betac
        pltcc.plot(x2,y2,label="y=exp(-Eb/Tc) Tc={}".format(1.0/self.betac),color="pink")
        # logx = np.log(x)
        # logy = np.log(y)
        para, pcov = curve_fit(self.__eb_fitfunc_1,np.array(x),np.array(y),maxfev = 10000000)
        fit_y = self.__eb_fitfunc_1(x, para[0], para[1], para[2]) #tau0, x, y
        # plt.plot(x,fit_y,label="Smirnov tau0={tau0} x={x} y={y}".format(tau0="%.2f"%para[0],x="%.2f"%para[1],y="%.2f"%para[2]),color="green")
        plt.plot(x,fit_y,label="a*exp(-eb^(b+1)/Tc/(eb^b+c)) a={a} b={b} c={c}".format(a="%.2f"%para[0],b="%.2f"%para[1],c="%.2f"%para[2]),color="green")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path[0])
        plt.savefig(img_path[1])
        plt.clf()
        return fit_y  
    def __eb_fitfunc_1(self,x,a,b,c):
        return a*np.exp(-x^(b+1)/Tc/(x^b+c))
    def __hist2d_taulog(self,xlabel,ylabel,title,x,y,label,img_path,bins):
        des = img_path
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.yscale("log")
        yy = [i for i in y if i>0]
        xx = [x[j] for j in range(len(x)) if y[j]>0]

        # xbins
        # bins=10
        # self.bins = 10
        xbins = np.linspace(0.0,max(xx),bins)
        # ybins = np.logspace(np.log10(min(yy)), np.log10(max(yy)),bins)
        bins_trct = self.num * 2
        bins_num_log = int(self.bins) 
        bins_line = np.linspace(1.0, bins_trct - 1, bins_trct - 1)
        bins_log = np.logspace(np.log10(bins_trct), np.log10(max(yy)), bins_num_log + 1)
        ybins = np.hstack([bins_line,bins_log])
        prob,X,Y,m = hist2d(xx, yy, bins=(xbins,ybins), norm=LogNorm(),label=label,density=1)
        print(prob.shape,len(X.tolist()),len(Y.tolist()))
        # print(prob,len(X.tolist()),len(Y.tolist()))
        print("x0=",X[0],len(X.tolist()),len(xbins.tolist()))
        lsplist_line = bins_line
        lsplist_log = np.array([np.sqrt(bins_log[i]*bins_log[i+1]) for i in range(bins_num_log)])

        xwidthlist = np.array([xbins[i+1]-xbins[i] for i in range(len(xbins) - 1)])
        ywidthlist_line = np.array([1.0 for i in range(bins_trct-1)])
        ywidthlist_log = np.array([bins_log[i+1]-bins_log[i] for i in range(bins_num_log)])
        ywidthlist = np.hstack([ywidthlist_line,ywidthlist_log])
        X = [(X[i]+X[i+1])/2 for i in range(len(X)-1)]
        Y = np.hstack([lsplist_line,lsplist_log])
        # xy = np.array([[x,y] for x in X for y in Y]).reshape(len(X),len(Y),2)
        xArrh = [0.0]
        yArrh = [1.0]
        err = [0.0]
        probMat = np.array(prob).reshape(len(X),len(Y))
        for i in range(len(X)):
            for j in range(len(Y)):
                probMat[i][j] = xwidthlist[i]*ywidthlist[j]*probMat[i][j]
 
        sumx = [np.sum(probMat[:i][:].flatten()) for i in range(len(X)+1)]
        sumy = [np.sum(probMat.T[:][:i].flatten()) for i in range(len(Y)+1)]
        # print(np.sum(probMat.T[:3][:].flatten()),np.sum(probMat.T[:3][:]))
        # print(np.sum(probMat[:][:3].flatten()),np.sum(probMat[:][:3]))
        # print("xxx",np.sum(probMat.T[:][:len(Y)].flatten()))
        # print(probMat)
        # print(probMat[:2][:])
        # print(probMat[:][:2])
        # print(sumx)
        # print(sumy)
        # print(sumx,sumy)
        # print(probMat[:][:1])
        # print(probMat.T[:1][:])
        
        for j in range(1,len(Y)):
            for i in range(len(X)):
                if sumy[j] - sumx[i] < 0:
                    idx = i if np.abs(sumy[j] - sumx[i]) < np.abs(sumy[j] - sumx[i-1]) else i-1
                    xArrh.append(X[idx])
                    yArrh.append(Y[j])
                    err.append(sumy[j] - sumx[idx])
                    break
        # logy = [log10(i) for i in y if i>0]
        # xx = [i for i in x]
        # index = [i for i,x in enumerate(y) if x<=0]
        # for i in index:
        #     xx.pop(i)
        xArrhFit = xArrh[int(0.2*bins):int(0.8*bins)]
        yArrhFit = yArrh[int(0.2*bins):int(0.8*bins)]
        para, pcov = curve_fit(self.Arrhenius_func,np.array(xArrhFit),np.array(yArrhFit),maxfev = 10000000)
        xArrhFit = np.linspace(0,max(xArrh),bins)
        yArrhFit = self.Arrhenius_func(xArrhFit, para[0]) #tau0, x, y

        xxbar, yy_output = Funcbar(xx, yy, bins)
        xxbar2, explogyy_output = Funclogbar(xx, yy, bins)
        yybar, xx_output = Funcbar_log(yy, xx, bins)
        dump_data(self.__txtpath("arithMean(tau)-Eb_{}".format(des)), [xxbar, yy_output])
        dump_data(self.__txtpath("geoMean(tau)-Eb_{}".format(des)), [xxbar2, explogyy_output])
        dump_data(self.__txtpath("tau-arithMean(Eb)_{}".format(des)), [xx_output,yybar])
        dump_data(self.__txtpath("tau-Eb_{}".format(des)), [xArrh,yArrh,err])
        dump_data(self.__txtpath("tau-Eb_Fit_{}".format(des)), [xArrhFit,yArrhFit,err])
        plt.plot(xxbar,yy_output,color="pink",label="<tau>(Eb)")
        plt.plot(xxbar2,explogyy_output,color="coral",label="<tau>(Eb) geoMean")
        plt.plot(xx_output,yybar,color="grey",label="tau(<Eb>)")
        plt.plot(xArrh,yArrh,label="tau(Eb)")
        plt.plot(xArrhFit,yArrhFit,label="tau(Eb) Fit")


        ratio1 = self.ratio_dynamicToStandard_Arrh(xxbar,yy_output)
        ratio2 = self.ratio_dynamicToStandard_Arrh(xxbar2,explogyy_output)
        ratio3 = self.ratio_dynamicToStandard_Arrh(xx_output,yybar)
        ratio4 = self.ratio_dynamicToStandard_Arrh(xArrh,yArrh)


        dump_data(self.__txtpath("Arrhenius_rate-Eb_arithMean(tau)-Eb_{}".format(des)),[xxbar,ratio1,])
        dump_data(self.__txtpath("Arrhenius_rate-Eb_geoMean(tau)-Eb_{}".format(des)),[xxbar2,ratio2,])
        dump_data(self.__txtpath("Arrhenius_rate-Eb_tau-arithMean(Eb)_{}".format(des)),[xx_output,ratio3,])
        dump_data(self.__txtpath("Arrhenius_rate-Eb_tau-Eb_{}".format(des)),[xArrh,ratio4,])

        del xxbar
        del yy_output
        del yybar
        del xx_output
        #theoretical result:
        tau = np.exp(xbins/self.temperature)
        plt.plot(xbins,tau,color="black",label="tau=exp(Eb/T)")
        #fir for all data:
        # fit_x = [ele for ele in xx if ele<3.5]
        # fit_y = [yy[i] for i in range(len(yy)) if xx[i]<3.5]
        # para, pcov = curve_fit(self.fit_func,np.array(fit_x),np.array(fit_y))
        # tau_fit = self.fit_func(xbins,para[0])
        # plt.plot(xbins,tau_fit,color="green",label="tau=exp(Eb/{bb})".format(bb="%.2f"%(1.0/para[0])))
        colorbar()
        plt.legend()
        plt.savefig(img_path[0])
        plt.savefig(img_path[1])
        plt.clf()
        del xx
        del yy
        # del fit_x
        # del fit_y
    def Arrhenius_func(self,x,const):
        return const*np.exp(x/self.temperature)

    def __hist2d_taulog2(self,xlabel,ylabel,title,x,y,label,img_path,bins):
        des = img_path
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.yscale("log")
        yy = [i for i in y if i>0]
        xx = [x[j] for j in range(len(x)) if y[j]>0]

        # xbins
        xbins = np.linspace(min(xx),max(xx),bins)
        ybins = np.logspace(np.log10(min(yy)), np.log10(max(yy)),bins)
        prob,X,Y,m = hist2d(xx, yy, bins=(xbins,ybins), norm=LogNorm(),label=label,density=1)
        print(prob.shape,len(X.tolist()),len(Y.tolist()))
        # print(prob,len(X.tolist()),len(Y.tolist()))

        # lsplist_line = bins_line
        # lsplist_log = np.array([np.sqrt(bins_log[i]*bins_log[i+1]) for i in range(bins_num_log)])
        X = [(X[i]+X[i+1])/2 for i in range(len(X)-1)]
        Y = [(Y[i]*Y[i+1])**(1/2) for i in range(len(Y)-1)]
        print(Y)
        # xy = np.array([[x,y] for x in X for y in Y]).reshape(len(X),len(Y),2)
        xArrh = [0.0]
        yArrh = [1.0]
        err = [0.0]
        probMat = np.array(prob).reshape(len(X),len(Y))
        sumx = [np.sum(probMat.T[:i][:]) for i in range(len(X)+1)]
        sumy = [np.sum(probMat[:][:i]) for i in range(len(Y)+1)]
        
        for j in range(1,len(Y)):
            for i in range(len(X)):
                if sumy[j] - sumx[i] < 0:
                    idx = i if np.abs(sumy[j] - sumx[i]) < np.abs(sumy[j] - sumx[i-1]) else i-1
                    xArrh.append(X[idx])
                    yArrh.append(Y[j])
                    err.append(sumy[j] - sumx[idx])
                    break
        # logy = [log10(i) for i in y if i>0]
        # xx = [i for i in x]
        # index = [i for i,x in enumerate(y) if x<=0]
        # for i in index:
        #     xx.pop(i)
        xArrhFit = xArrh[int(0.2*bins):int(0.8*bins)]
        yArrhFit = yArrh[int(0.2*bins):int(0.8*bins)]
        para, pcov = curve_fit(self.Arrhenius_func,np.array(xArrhFit),np.array(yArrhFit),maxfev = 10000000)
        xArrhFit = np.linspace(0,max(xArrh),bins)
        yArrhFit = self.Arrhenius_func(xArrhFit, para[0]) #tau0, x, y

        xxbar, yy_output = Funcbar(xx, yy, bins)
        xxbar2, explogyy_output = Funclogbar(xx, yy, bins)
        yybar, xx_output = Funcbar_log(yy, xx, bins)
        dump_data(self.__txtpath("arithMean(tau)-Eb_{}2".format(des)), [xxbar, yy_output])
        dump_data(self.__txtpath("geoMean(tau)-Eb_{}2".format(des)), [xxbar2, explogyy_output])
        dump_data(self.__txtpath("tau-arithMean(Eb)_{}2".format(des)), [xx_output,yybar])
        dump_data(self.__txtpath("tau-Eb_{}2".format(des)), [xArrh,yArrh,err])
        dump_data(self.__txtpath("tau-Eb_Fit_{}2".format(des)), [xArrhFit,yArrhFit,err])
        plt.plot(xxbar,yy_output,color="pink",label="<tau>(Eb)")
        plt.plot(xxbar2,explogyy_output,color="coral",label="<tau>(Eb) geoMean")
        plt.plot(xx_output,yybar,color="grey",label="tau(<Eb>)")
        plt.plot(xArrh,yArrh,label="tau(Eb)")
        plt.plot(xArrhFit,yArrhFit,label="tau(Eb) Fit")


        ratio1 = self.ratio_dynamicToStandard_Arrh(xxbar,yy_output)
        ratio2 = self.ratio_dynamicToStandard_Arrh(xxbar2,explogyy_output)
        ratio3 = self.ratio_dynamicToStandard_Arrh(xx_output,yybar)
        ratio4 = self.ratio_dynamicToStandard_Arrh(xArrh,yArrh)


        dump_data(self.__txtpath("Arrhenius_rate-Eb_arithMean(tau)-Eb_{}2".format(des)),[xxbar,ratio1,])
        dump_data(self.__txtpath("Arrhenius_rate-Eb_geoMean(tau)-Eb_{}2".format(des)),[xxbar2,ratio2,])
        dump_data(self.__txtpath("Arrhenius_rate-Eb_tau-arithMean(Eb)_{}2".format(des)),[xx_output,ratio3,])
        dump_data(self.__txtpath("Arrhenius_rate-Eb_tau-Eb_{}2".format(des)),[xArrh,ratio4,])
        

        del xxbar
        del yy_output
        del yybar
        del xx_output
        #theoretical result:
        tau = np.exp(xbins/self.temperature)
        plt.plot(xbins,tau,color="black",label="tau=exp(Eb/T)")
        #fir for all data:
        # fit_x = [ele for ele in xx if ele<3.5]
        # fit_y = [yy[i] for i in range(len(yy)) if xx[i]<3.5]
        # para, pcov = curve_fit(self.fit_func,np.array(fit_x),np.array(fit_y))
        # tau_fit = self.fit_func(xbins,para[0])
        # plt.plot(xbins,tau_fit,color="green",label="tau=exp(Eb/{bb})".format(bb="%.2f"%(1.0/para[0])))
        colorbar()
        plt.legend()
        plt.savefig(img_path[0])
        plt.savefig(img_path[1])
        plt.clf()
        del xx
        del yy
        # del fit_x
        # del fit_y
    def ratio_dynamicToStandard_Arrh(self,xx_output,yybar):
        # tau = np.exp(np.array(xx_output)/self.temperature)
        tau = [np.exp(xx/self.temperature) for xx in xx_output]
        ratio = [yybar[i]/tau[i] for i in range(len(xx_output))]
        return ratio



class HandleData:
    def __init__(self,model,num,temperature,betac,TIME_MAX,startnum,samplenum,samplenum_tree,bins,bins2,dir_index) -> None:
        self.model = model
        self.num = num
        self.temperature = temperature
        self.timemax = TIME_MAX
        self.start = startnum
        self.end = samplenum
        self.endtree = samplenum_tree
        self.bins = bins
        self.bins2 = bins2
        self.betac = betac
        self.dir = "../data/N={dd}_T={cc}_{ee}{ff}/".format(cc="%.2f"%self.temperature,dd=self.num,ee="%.0le"%self.timemax,ff=dir_index)
        self.treedir = "../data/N={dd}_T={cc}_{ee}{ff}/tree_data/".format(cc="%.2f"%self.temperature,dd=self.num,ee="%.0le"%self.timemax,ff=dir_index)
        self.dtime = 1e6
        self.dtime2 = 1e7
        self.taumin = 1e2
        self.taumax = 1e7
        self.alpha = temperature * betac

        self.e0lists = []
        self.eslists = []
        self.e1lists = []
        self.eb1lists = []
        self.eb2lists = []
        self.effeb1lists = []
        self.effeb2lists = []

        self.tau0lists = []
        self.tauslists = []
        self.taulists = []

        self.ageing_func_timelist = []     # to timemax/2
        self.ageing_func_rate = []
        self.tw = 1e4
        self.begint = 1e4

        self.treeblists = []
        self.treee0lists = []
        self.treeeslists = []
        
        self.e0barlist1 = [0 for i in range(int(TIME_MAX/self.dtime))]
        self.e0numlist1 = [0 for i in range(int(TIME_MAX/self.dtime))]
        self.e0timelist1 = [self.dtime*i for i in range(int(TIME_MAX/self.dtime))]

        self.e0barlist2 = [0 for i in range(int(TIME_MAX/self.dtime2))]
        self.e0numlist2 = [0 for i in range(int(TIME_MAX/self.dtime2))]
        self.e0timelist2 = [self.dtime2*i for i in range(int(TIME_MAX/self.dtime2))]

        self.timelists = []

        self.e1timelist = [i*(i+1) for i in range(self.__time_to_int(self.timemax))]  # con use for e0bar and tau0bar
        self.e1barlist = [0.0 for i in range(self.__time_to_int(self.timemax))]
        self.e1numlist = [0 for i in range(self.__time_to_int(self.timemax))]   # con use for e0bar and tau0bar
        self.e0barlist = [0.0 for i in range(self.__time_to_int(self.timemax))]
        self.tau0barlist = [0.0 for i in range(self.__time_to_int(self.timemax))]

        self.tau1timelist = [i*(i+1) for i in range(self.__time_to_int(self.timemax))]
        self.tau1barlist = [0.0 for i in range(self.__time_to_int(self.timemax))]
        self.tau1numlist = [0 for i in range(self.__time_to_int(self.timemax))]

        self.treeleb = []
        self.treeleffeb = []
        self.treeentropy = []

        self.timelist = [i*(i+1)/2 for i in range(1,self.__time_to_int2(self.timemax))]
        self.ebart = [0 for i in range(1,self.__time_to_int2(self.timemax))]
        self.e0bart = [0 for i in range(1,self.__time_to_int2(self.timemax))]
        self.dec_t = [0 for i in range(1,self.__time_to_int2(self.timemax))]
        self.esbart = [0 for i in range(1,self.__time_to_int2(self.timemax))]

        self.r1e1 = []
        self.r2e1 = []
        self.r3e1 = []
        # self.

        self.tau0lists_pair = []
        self.tau1lists_pair = []
        self.eb0lists_pair = []
        self.eb1lists_pair = []
    def read_correlation_tdtw(self):
        ageing_func_hit = np.array([0 for i in range(self.__time_to_int(self.timemax))])
        ageing_func_total = np.array([0 for i in range(self.__time_to_int(self.timemax))])
        self.tw = 0
        
        for i in range(self.start,self.end):
            time = 0
            jlast_last = 0
            act_str = "activation-t_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            with open(self.dir + act_str,'r') as fp:
                while 1:
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    timelast = time
                    data = linedata.split()
                    time = float(data[0])
                    jlast = self.__time_to_int(timelast)
                    jnow = self.__time_to_int(time)
                    jnow_middle = self.__time_to_int(2/3*time)
                    j_middlelist = [k for k in range(jlast,jnow_middle)]
                    jlist = [k for k in range(jlast,jnow)]
                    ageing_func_hit[j_middlelist] += 1
                    ageing_func_total[jlist] += 1
                    
        self.ageing_func_rate = [ageing_func_hit[i]/ageing_func_total[i] for i in range(self.__time_to_int(self.timemax/2)) if ageing_func_total[i]!=0]        
        self.ageing_func_timelist = [i*(i+1) for i in range(self.__time_to_int(self.timemax/2)) if ageing_func_total[i]!=0]
    def plot_correlation_tdtw(self):
        self.read_correlation_tdtw()
        fname = self.dir + "correlation-t/correlation-tdtw_{model}{a}_T={b}_timemax={c}.t".format(
            a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model)
        print(fname)
        self.printf(self.ageing_func_timelist
            , self.ageing_func_rate, fname)
        # self.__plot("tw"
        #     ,"C(tw,0.5*tw)"
        #     ,self.ageing_func_timelist
        #     ,self.ageing_func_rate
        #     ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        # self.__plot_xlog("tw"
        #     ,"C(tw,0.5*tw)"
        #     ,self.ageing_func_timelist
        #     ,self.ageing_func_rate
        #     ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        # self.__plot_xylog("tw"
        #     ,"C(tw,0.5*tw)"
        #     ,self.ageing_func_timelist
        #     ,self.ageing_func_rate
        #     ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))


    def read_data_dynamics(self):
        ageing_func_hit = np.array([0 for i in range(self.__time_to_int(self.timemax))])
        ageing_func_total = np.array([0 for i in range(self.__time_to_int(self.timemax))])
        ageing_func_hit_fixed = np.array([0 for i in range(self.__time_to_int(self.timemax))])
        ageing_func_total_fixed = np.array([0 for i in range(self.__time_to_int(self.timemax))])
        ageing_func_hit_beginfixed = np.array([0 for i in range(self.__time_to_int(self.timemax - self.begint))])
        ageing_func_total_beginfixed = np.array([0 for i in range(self.__time_to_int(self.timemax - self.begint))])

        for i in range(self.start,self.end):
            self.taulist = []
            self.eb1list = []
            time = 0
            jlast_last = 0
            act_str = "activation-t_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            with open(self.dir + act_str,'r') as fp:
                #fp.readline()
                while 1:
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    else:
                        # if(len(linedata.split())<2):
                        #     print(i)
                        timelast = time
                        try:
                            (time,e0,es,e1,tip0,saddle,tip1,effe0,effes,effe1,effs,tau0,taus, tau) = [t(s) for t,s in zip((int,float,float,float,int,int,int,float,float,float,int,int,int,int),linedata.split())]
                        except:
                            print(act_str)
                        jlast = self.__time_to_int(timelast)
                        jnow = self.__time_to_int(time)
                        jnow_middle = self.__time_to_int(2/3*time)
                        j_middlelist = [k for k in range(jlast,jnow_middle)]
                        jlist = [k for k in range(jlast,jnow)]
                        ageing_func_hit[j_middlelist] += 1
                        ageing_func_total[jlist] += 1

                        # if time>1e8 and time<9*1e8:
                        #     self.taulists_range.append(tau)
                        #     self.taulist_range.append(tau)

                        if time>self.tw:
                            jnow_middle_fixed = self.__time_to_int(time - self.tw)
                            j_middlelist_fixed = [k for k in range(jlast,jnow_middle_fixed)]
                            jlist_fixed = [k for k in range(jlast,jnow)]
                            ageing_func_hit_fixed[j_middlelist_fixed] += 1
                            ageing_func_total_fixed[jlist_fixed] += 1

                        if time>self.begint and self.begint>=timelast:
                            j_beginfixed = self.__time_to_int(time - self.begint)
                            jlist_beginfixed = [k for k in range(0,j_beginfixed)]
                            ageing_func_hit_beginfixed[jlist_beginfixed] += 1
                            ageing_func_total_beginfixed += 1


            #             for j in range(jlast,jnow):
            #                 self.e1barlist[j] += e1/self.num
            #                 self.e1numlist[j] += 1
            #                 self.e0barlist[j] += e0/self.num
            #                 self.tau0barlist[j] += float(tau)
            #             for j in range(jlast_last,jlast):
            #                 self.tau1barlist[j] += float(tau)
            #                 self.tau1numlist[j] += 1
            #             jlast_last = jlast

            #             # self.timelists.append(float(time))
            #             if -0.9<e0/self.num<=-0.7:
            #                 self.r1e1.append(e1/self.num)
            #             if -0.7<e0/self.num<=-0.5:
            #                 self.r2e1.append(e1/self.num)
            #             if -0.5<e0/self.num<=-0.3:
            #                 self.r3e1.append(e1/self.num)
            #             self.e0lists.append(e0/self.num)
            #             self.eslists.append(es/self.num)
            #             self.e1lists.append(e1/self.num)
            #             self.eb1lists.append((es-e0))
            #             self.eb1list.append((es-e0))
            #             self.eb2lists.append((es-e1))
            #             self.effeb1lists.append((effes-effe0))
            #             self.effeb2lists.append((effes-effe1))

            #             self.tau0lists.append(float(tau0))
            #             self.tauslists.append(float(taus))
            #             self.taulists.append(float(tau))
            #             self.taulist.append(float(tau))

            #             time_index = int(time/self.dtime)
            #             self.e0barlist1[time_index] += e1/self.num
            #             self.e0numlist1[time_index] += 1
            #             time_index2 = int(time/self.dtime2)
            #             self.e0barlist2[time_index2] += e1/self.num
            #             self.e0numlist2[time_index2] += 1
            # self.tau0lists_pair.extend(self.taulist[1:-1])
            # self.tau1lists_pair.extend(self.taulist[2:])
            # self.eb0lists_pair.extend(self.eb1list[1:-1])
            # self.eb1lists_pair.extend(self.eb1list[2:])

       

        self.ageing_func_rate = [ageing_func_hit[i]/ageing_func_total[i] for i in range(self.__time_to_int(self.timemax/2)) if ageing_func_total[i]!=0]        
        self.ageing_func_timelist = [i*(i+1) for i in range(self.__time_to_int(self.timemax/2)) if ageing_func_total[i]!=0]
        self.ageing_func_rate_fixed = [ageing_func_hit_fixed[i]/ageing_func_total_fixed[i] for i in range(self.__time_to_int(self.timemax/2)) if ageing_func_total_fixed[i]!=0]        
        self.ageing_func_timelist_fixed = [i*(i+1) for i in range(self.__time_to_int(self.timemax/2)) if ageing_func_total_fixed[i]!=0]
        self.ageing_func_rate_beginfixed = [ageing_func_hit_beginfixed[i]/ageing_func_total_beginfixed[i] for i in range(self.__time_to_int(self.timemax/2)) if ageing_func_total_beginfixed[i]!=0]        
        self.ageing_func_timelist_beginfixed = [i*(i+1)+self.begint for i in range(self.__time_to_int(self.timemax/2)) if ageing_func_total_beginfixed[i]!=0]

        self.e1barlist = [self.e1barlist[j]/self.e1numlist[j] for j in range(self.__time_to_int(self.timemax)) if self.e1numlist[j]!=0]
        self.e0barlist = [self.e0barlist[j]/self.e1numlist[j] for j in range(self.__time_to_int(self.timemax)) if self.e1numlist[j]!=0]
        self.tau0barlist = [self.tau0barlist[j]/self.e1numlist[j] for j in range(self.__time_to_int(self.timemax)) if self.e1numlist[j]!=0]
        self.tau1barlist = [self.tau1barlist[j]/self.tau1numlist[j] for j in range(self.__time_to_int(self.timemax)) if self.tau1numlist[j]!=0]
        #time:
        self.e1timelist = [self.e1timelist[j] for j in range(self.__time_to_int(self.timemax)) if self.e1numlist[j]!=0]
        self.tau1timelist = [self.tau1timelist[j] for j in range(self.__time_to_int(self.timemax)) if self.tau1numlist[j]!=0]

        self.e0barlist1 = [self.e0barlist1[i]/self.e0numlist1[i] for i in range(int(self.timemax/self.dtime)) if self.e0numlist1[i]>0]
        self.e0barlist2 = [self.e0barlist2[i]/self.e0numlist2[i] for i in range(int(self.timemax/self.dtime2)) if self.e0numlist2[i]>0]
        self.e0timelist1 = [self.dtime*i for i in range(int(self.timemax/self.dtime)) if self.e0numlist1[i]>0]
        self.e0timelist2 = [self.dtime2*i for i in range(int(self.timemax/self.dtime2)) if self.e0numlist2[i]>0]
        print("finish: read and construct data from dynamics.")


    def read_rem_tau_gen_func_differentSpl(self,dirname):
        self.taulists_genfunc1e0 = []
        self.taulists_genfunc1e1 = []
        self.taulists_genfunc1e2 = []
        self.taulists_genfunc1e3 = []
        self.taulists_genfunc1e4 = []
        self.taulists_genfunc1e5 = []
        self.tau0lists_genfunc1e0 = []
        self.tau0lists_genfunc1e1 = []
        self.tau0lists_genfunc1e2 = []
        self.tau0lists_genfunc1e3 = []
        self.tau0lists_genfunc1e4 = []
        self.tau0lists_genfunc1e5 = []
        # self.tau0lists_exact = []
        enum=0
        fnames = os.listdir(dirname)
        for fname in fnames:
            j=0
            with open(dirname + fname,"r",encoding='utf-8') as fp:
                while 1:                    
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    tau = float(linedata.split()[0])
                    tau0 = float(linedata.split()[1])
                    if j < 1e0:
                        self.taulists_genfunc1e0.append(tau)
                        self.tau0lists_genfunc1e0.append(tau0)
                    if j < 1e1:
                        self.taulists_genfunc1e1.append(tau)
                        self.tau0lists_genfunc1e1.append(tau0)
                    if j < 1e2:
                        self.taulists_genfunc1e2.append(tau)
                        self.tau0lists_genfunc1e2.append(tau0)
                    if j < 1e3:
                        self.taulists_genfunc1e3.append(tau)
                        self.tau0lists_genfunc1e3.append(tau0)
                    if j < 1e4:
                        self.taulists_genfunc1e4.append(tau)
                        self.tau0lists_genfunc1e4.append(tau0)
                    # if j < 1e5:
                    #     self.taulists_genfunc1e5.append(tau)
                    #     self.tau0lists_genfunc1e5.append(tau0)
                    # if int(linedata) == 1:
                    #     enum+=1
                    j+=1
            #         if 0<tau<self.tau_max:
            #             self.taulists_exact.append(tau)
            # if i==int((self.end-self.start)/100):
            #     print("+1")
        print(enum)
        print("finish reading taulist")
    def plot_rem_tau_gen_func_mixed_differentSpl(self,sn, bins_trct, dirname):
        self.sn = sn
        self.read_rem_tau_gen_func_differentSpl(dirname)
        self.gendos_printf_tau_genfunc(self.taulists_genfunc1e0,self.tau0lists_genfunc1e0,1e0*(self.end-self.start),bins_trct,dirname)
        self.gendos_printf_tau_genfunc(self.taulists_genfunc1e1,self.tau0lists_genfunc1e1,1e1*(self.end-self.start),bins_trct,dirname)
        self.gendos_printf_tau_genfunc(self.taulists_genfunc1e2,self.tau0lists_genfunc1e2,1e2*(self.end-self.start),bins_trct,dirname)
        self.gendos_printf_tau_genfunc(self.taulists_genfunc1e3,self.tau0lists_genfunc1e3,1e3*(self.end-self.start),bins_trct,dirname)
        self.gendos_printf_tau_genfunc(self.taulists_genfunc1e4,self.tau0lists_genfunc1e4,1e4*(self.end-self.start),bins_trct,dirname)
        # self.gendos_printf_tau_genfunc(self.taulists_genfunc1e5,self.tau0lists_genfunc1e5,1e5*(self.end-self.start),bins_trct,dirname)
        # fitdos = self.__plot_xylog_tau_genfunc("tau_gen_func_fitfunc_BinsTrct={bins_trct}_bins={bins}".format(bins_trct=bins_trct,bins=self.bins)
        #     ,"pdf"
        #     ,lsp
        #     ,dos
        #     ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        # fname_fit = self.dir + "pdf-tau_fit_genfunc_mixed_{model}{a}_T={b}_timemax={c}.t".format(a=self.num
        #             ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model)
        # self.printf_with_width(lsp,fitdos,width,fname_fit)
    def gendos_printf_tau_genfunc(self, lists, lists0, splNum, bins_trct, dirname):
        # lists = [ele for ele in lists if ele > 0]
        # if not lists == []:
        lsp, dos, width = self.gen_doslist_mixed_linearFine(lists, bins_trct, self.bins)
        del lists
        fname = self.dir + "pdf-tau_mixed_Tsplnum={spl}_{dir}_{model}{a}_T={b}_timemax={c}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model,spl="%.1le"%splNum,dir=dirname.split("/")[-2])
        self.printf_with_width(lsp,dos,width,fname)
        self.__plot_xylog_tau_genfunc("tau_mixed_fitfunc_Tsplnum={spl}_{dir}_{model}{a}_T={b}_timemax={c}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model,spl="%.1le"%splNum,dir=dirname.split("/")[-2])
            ,"pdf"
            ,lsp
            ,dos
            ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

        lsp, dos, width = self.gen_doslist_mixed_linearFine(lists0, bins_trct, self.bins)
        del lists0
        fname = self.dir + "pdf-tau0_mixed_Tsplnum={spl}_{dir}_{model}{a}_T={b}_timemax={c}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model,spl="%.1le"%splNum,dir=dirname.split("/")[-2])
        self.printf_with_width(lsp,dos,width,fname)
        self.__plot_xylog_tau_genfunc("tau0_mixed_fitfunc_Tsplnum={spl}_{dir}_{model}{a}_T={b}_timemax={c}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model,spl="%.1le"%splNum,dir=dirname.split("/")[-2])
            ,"pdf"
            ,lsp
            ,dos
            ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        # else:
        #     lsp,dos,width = [],[],[]
        return lsp, dos, width

    def read_rem_tau_gen_func(self):
        self.taulists_genfunc = []
        self.taulists_exact = []
        enum=0
        mkdir(self.dir + "taulists")
        for i in range(self.start, self.end):
            last_time = 1
            j=0
            tau_str = "taulists/taulists_genfunc_SN={sn}_{model}{a}_T={b}_timemax={c}_{d}.t".format(sn="%.0le"%self.sn,a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            with open(self.dir + tau_str,"r",encoding='utf-8') as fp:
                while 1:                    
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    tau = float(linedata)
                    self.taulists_genfunc.append(tau)
                    if int(linedata) == 1:
                        enum+=1
            #         if 0<tau<self.tau_max:
            #             self.taulists_exact.append(tau)
            # if i==int((self.end-self.start)/100):
            #     print("+1")
        print(enum)
        print("finish reading taulist")
    def plot_rem_tau_gen_func(self,sn,tau_max):
        self.tau_max = tau_max
        self.sn = sn
        self.read_rem_tau_gen_func()
        lsp, dos = self.gen_doslist(self.taulists_genfunc,int(max(self.taulists_genfunc)))
        self.__plot_xylog_tau("tau_gen_func"
            ,"pdf"
            ,lsp
            ,dos
            ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
    def plot_rem_tau_gen_func_exact(self,sn,tau_max):
        self.tau_max = tau_max
        self.sn = sn
        self.read_rem_tau_gen_func()
        bins = int(max(self.taulists_exact)-min(self.taulists_exact))
        print(max(self.taulists_exact),min(self.taulists_exact))
        lsp, dos = self.gen_doslist(self.taulists_exact, bins)
        print(dos)
        self.__plot_xylog_tau("tau_gen_func_tau=0-{}".format("%.0le"%self.taumax)
            ,"pdf"
            ,lsp
            ,dos
            ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
    def plot_rem_tau_gen_func_mixed(self,sn, bins_trct):
        self.sn = sn
        self.read_rem_tau_gen_func()
        lsp, dos, width= self.gen_doslist_mixed_linearFine(self.taulists_genfunc, bins_trct, self.bins)
        fname = self.dir + "pdf-tau_genfunc_mixed_{model}{a}_T={b}_timemax={c}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model)
        self.printf_with_width(lsp,dos,width,fname)
        fitdos = self.__plot_xylog_tau_genfunc("tau_gen_func_fitfunc_BinsTrct={bins_trct}_bins={bins}".format(bins_trct=bins_trct,bins=self.bins)
            ,"pdf"
            ,lsp
            ,dos
            ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        fname_fit = self.dir + "pdf-tau_fit_genfunc_mixed_{model}{a}_T={b}_timemax={c}.t".format(a=self.num
                    ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model)
        self.printf_with_width(lsp,fitdos,width,fname_fit)
    def __plot_xylog_tau_genfunc(self,xlabel,ylabel,x,y,label):
        title = ylabel +"-"+ xlabel +"_xylog"
        img_path = ylabel +"-"+ xlabel +"_xylog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(title)
        plt.plot(x,y,label=label)
        x2 = linspace(10**(np.log10(max(x))*2/3), max(x),1000)
        y2 = x2**(-1.0*self.temperature*self.betac-1)*100
        self.a = 1.0*self.temperature*self.betac
        plt.plot(x2,y2,label="y=x^(-1-alpha)) alpha={}".format(1.0*self.temperature*self.betac),color="pink")
        # logx = np.log(x)
        # logy = np.log(y)
        para, pcov = curve_fit(self.__fit_tau_genfunc_no_a,np.array(x),np.array(y),maxfev = 10000)
        fit_y = self.__fit_tau_genfunc_no_a(x, para[0], para[1]) #tau0, x, y
        # plt.plot(x,fit_y,label="Smirnov tau0={tau0} x={x} y={y}".format(tau0="%.2f"%para[0],x="%.2f"%para[1],y="%.2f"%para[2]),color="green")
        plt.plot(x,fit_y,label="Smirnov tau0={tau0} x={x} y={y}".format(tau0="%.2f"%para[0],x="%.3f"%self.a,y="%.2f"%para[1]),color="green")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
        return fit_y   
    def __fit_tau_genfunc(self,tau,tau0,a,b):
        return ((1.0+a)/b)**(a/b) * tau0**a * b/math.gamma(a/b)  *  tau**(-a-1.0)  *  np.exp(-1.0*tau0**b*(1.0+a)/b * tau**(-b))
    def __fit_tau_genfunc_no_a(self,tau,tau0,b):
        a = 1.0*self.temperature*self.betac
        return ((1.0+a)/b)**(a/b) * tau0**a * b/math.gamma(a/b)  *  tau**(-a-1.0)  *  np.exp(-1.0*tau0**b*(1.0+a)/b * tau**(-b))
    def __fit_tau_genfunc_xylog(self,logtau,tau0,a,b):
        return (a/b)*np.log((1.0+a)/b) + a*np.log(tau0) + log(b)-np.log(math.gamma(a/b))  +  (-a-1.0)*logtau   -1.0*tau0**b*(1.0+a)/b * np.exp(logtau)**(-b)
    def printf_with_width(self,lsp,dos,width,path):
        with open(path,'w+',encoding='utf-8') as fp:
            for i in range(len(lsp)):
                #print(len(lsp))
                fp.write("%le"%lsp[i]+"\t"+"%le"%dos[i]+"\t"+"%le"%width[i]+"\n")

    def plot_rangee1(self):
        self.read_data_dynamics()
        self.lsp1, self.dos1 = self.gen_doslist(self.r1e1,self.bins)
        self.lsp2, self.dos2 = self.gen_doslist(self.r2e1,self.bins)
        self.lsp3, self.dos3 = self.gen_doslist(self.r3e1,self.bins)
        self.__plot_re1()
    def __plot_re1(self):
        xlabel = "e1"
        ylabel = "p(e1)"
        title = ylabel +"-"+ xlabel
        img_path = ylabel +"-"+ xlabel
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.plot(self.lsp1, self.dos1,label="e0=-0.9- -0.7")
        plt.plot(self.lsp2, self.dos2,label="e0=-0.7- -0.5")
        plt.plot(self.lsp3, self.dos3,label="e0=-0.5- -0.3")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()

    def read_tautoE_t(self):
        self.taulists_range2 = []
        self.taulists_range3 = []
        self.taulist_range2 = []
        self.taulist_spe2 = []
        self.timelist = [i*(i+1) for i in range(35000)]
        self.ttelist = [0.0 for i in range(35000)]
        for i in range(self.start, self.end):
            # self.taulist_range = []
            j=0
            tau_str = "tau-t_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            with open(self.dir + tau_str,"r",encoding='utf-8') as fp:
                while 1:                    
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    (time,tau) = [t(s) for t,s in zip((float,float),linedata.split())]
                    if time>1e9:
                        break
                    self.ttelist[j] += -1.0*np.log(tau)*self.temperature/(self.end-self.start)/self.num
                    j+=1
    def plot_tautoE_t(self):
        self.read_tautoE_t()
        self.__plot_xlog("time"
                ,"tau_to_E"
                ,self.timelist
                ,self.ttelist
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

    def read_tau_t_corr_dynamics_from_time_cumulationcum(self):
        self.time_corr_dynamics_from_time = [j*(j+1) for j in range(self.__time_to_int(1e8*2))]
        self.taut_corr_dynamics_from_time = [0 for j in range(self.__time_to_int(1e8*2))]
        self.taut_1sample_corr_dynamics_from_time = [j*(j+1) for j in range(self.__time_to_int(1e8*2))]
        for i in range(self.start, self.end):
            last_time = 1
            j=0
            tau_str = "time_corr_dynamics_from_time_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            with open(self.dir + tau_str,"r",encoding='utf-8') as fp:
                while 1:                    
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    (time,tau) = [t(s) for t,s in zip((float,float),linedata.split())]
                    jlast = self.__time_to_int(last_time)
                    jnow = self.__time_to_int(time)
                    for j in range(jlast,jnow):
                        self.taut_corr_dynamics_from_time[j] += tau/(self.end-self.start)
                    last_time = time
    def plot_tau_t_corr_dynamics_from_time_cumulationcum(self):
        mkdir(self.dir + "/plot/tau-t_corr_dynamics")
        self.read_tau_t_corr_dynamics_from_time_cumulationcum()
        self.__plot("time_corr_dynamics_from_time"
                ,"trapping_time"
                ,self.time_corr_dynamics_from_time
                ,self.taut_corr_dynamics_from_time
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xlog("time_corr_dynamics_from_time"
                ,"trapping_time"
                ,self.time_corr_dynamics_from_time
                ,self.taut_corr_dynamics_from_time
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xylog("time_corr_dynamics_from_time"
                ,"trapping_time"
                ,self.time_corr_dynamics_from_time
                ,self.taut_corr_dynamics_from_time
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
    def __plot_tau_t_corr_dynamics_from_time_cumulationcum_1sample(self,i):
        self.__plot_1sample("time_corr_dynamics", "tau", self.time_corr_dynamics_from_time, self.taut_1sample_corr_dynamics_from_time
                        , "{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model),i,"tau-t_corr_dynamics")
        self.__plot_1sample_xlog("time_corr_dynamics", "tau", self.time_corr_dynamics_from_time, self.taut_1sample_corr_dynamics_from_time
                        , "{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model),i,"tau-t_corr_dynamics")
        self.__plot_1sample_xylog("time_corr_dynamics", "tau", self.time_corr_dynamics_from_time, self.taut_1sample_corr_dynamics_from_time
                        , "{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model),i,"tau-t_corr_dynamics")

    
    def read_data_e_tau_mc(self):
        self.taulists_range = []
        self.e0lists_range = []
        self.eslists_range = []
        self.eblists_range = []
        self.taulist_range = []
        self.taulist_spe = []
        for i in range(self.start,self.end):
            self.taulist = []
            self.eb1list = []

            time = 0
            jlast_last = 0
            act_str = "activation-t_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            with open(self.dir + act_str,'r') as fp:
                #fp.readline()
                while 1:
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    else:
                        # if(len(linedata.split())<2):
                        #     print(i)
                        timelast = time
                        try:
                            (time,e0,es,e1,tip0,saddle,tip1,effe0,effes,effe1,effs,tau0,taus, tau) = [t(s) for t,s in zip((int,float,float,float,int,int,int,float,float,float,int,int,int,int),linedata.split())]
                        except:
                            print(act_str)
                        
                        if time>1e8 and time<9*1e8:
                            # self.taulists_range.append(float(tau))
                            # self.e0lists_range.append(e0/self.num)
                            # self.eslists_range.append(es/self.num)
                            # self.eblists_range.append(es-e0)
                            if time == 200010306:
                                self.taulist_spe.append(float(tau))
                            if i == 1:
                                self.taulist_range.append(float(tau))
        print("finish: read and construct data from dynamics.")
    def read_tau_t_mc(self):
        self.taulists_range2 = []
        self.taulists_range3 = []
        self.taulist_range2 = []
        self.taulist_spe2 = []
        self.taulist_1e0 = []
        self.taulist_1e1 = []
        self.taulist_1e2 = []
        self.taulist_1e3 = []
        self.taulist_1e4 = []
        self.taulist_1e5 = []
        self.taulist_1e6 = []
        self.taulist_1e7 = []
        for i in range(self.start, self.end):
            # self.taulist_range = []
            j=0
            tau_str = "tau-t_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            with open(self.dir + tau_str,"r",encoding='utf-8') as fp:
                while 1:                    
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    (time,tau) = [t(s) for t,s in zip((float,float),linedata.split())]

                    if time>1e9:
                        break
                    # if time>1e8 and time<9*1e8:
                        # self.taulists_range2.append(tau)
                    if time == 0:
                        self.taulist_1e0.append(float(tau))
                    if time == 12:
                        self.taulist_1e1.append(float(tau))
                    if time == 110:
                        self.taulist_1e2.append(float(tau))
                    if time == 1056:
                        self.taulist_1e3.append(float(tau))
                    if time == 10100:
                        self.taulist_1e4.append(float(tau))
                    if time == 100172:
                        self.taulist_1e5.append(float(tau))
                    if time == 1001000:
                        self.taulist_1e6.append(float(tau))
                    if time == 10001406:
                        self.taulist_1e7.append(float(tau))
                    if time == 200010306:
                        self.taulist_spe2.append(float(tau))
                    if i == 1:
                        self.taulist_range2.append(float(tau))
    
            # tau_str2 = "tau-t_equal_interval_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            # with open(self.dir + tau_str,"r",encoding='utf-8') as fp:
            #     while 1:                    
            #         linedata = fp.readline()
            #         if linedata.strip()=="":
            #             break
            #         (time,tau) = [t(s) for t,s in zip((float,float),linedata.split())]
            #         if time>1e9:
            #             break
            #         if time>1e8 and time<9*1e8:
            #             self.taulists_range3.append(tau)
        print("finish reading tau-t mc.")
    def plot_tau_dos_mc_compared_different_tw(self):
        # self.read_data_e_tau_mc()
        self.read_tau_t_mc()
        self.gen_tau_dos_mc_compared()

        self.__plot_xylog_tau_dos_compared_different_tw()
    def gen_tau_dos_mc_compared(self):
        # self.lsp_taulist_range, self.dos_taulist_range = self.gen_doslist_log(self.taulist_range,int(self.bins/2))
        # del self.taulist_range
        self.lsp_taulist_range2, self.dos_taulist_range2 = self.gen_doslist_log(self.taulist_range2,int(self.bins/2))
        del self.taulist_range2
        # print(self.taulist_spe)
        # self.lsp_taulist_spe, self.dos_taulist_spe = self.gen_doslist_log(self.taulist_spe,int(self.bins/100))
        # del self.taulist_spe
        self.lsp_taulist_spe2, self.dos_taulist_spe2 = self.gen_doslist_log(self.taulist_spe2,int(self.bins/5))
        del self.taulist_spe2
        self.lsp_taulist_1e0, self.dos_taulist_1e0 = self.gen_doslist_log(self.taulist_1e0,int(self.bins/5))
        del self.taulist_1e0
        self.lsp_taulist_1e1, self.dos_taulist_1e1 = self.gen_doslist_log(self.taulist_1e1,int(self.bins/5))
        del self.taulist_1e1
        self.lsp_taulist_1e2, self.dos_taulist_1e2 = self.gen_doslist_log(self.taulist_1e2,int(self.bins/5))
        del self.taulist_1e2
        self.lsp_taulist_1e3, self.dos_taulist_1e3 = self.gen_doslist_log(self.taulist_1e3,int(self.bins/5))
        del self.taulist_1e3
        self.lsp_taulist_1e4, self.dos_taulist_1e4 = self.gen_doslist_log(self.taulist_1e4,int(self.bins/5))
        del self.taulist_1e4
        self.lsp_taulist_1e5, self.dos_taulist_1e5 = self.gen_doslist_log(self.taulist_1e5,int(self.bins/5))
        del self.taulist_1e5
        self.lsp_taulist_1e6, self.dos_taulist_1e6 = self.gen_doslist_log(self.taulist_1e6,int(self.bins/5))
        del self.taulist_1e6
        self.lsp_taulist_1e7, self.dos_taulist_1e7 = self.gen_doslist_log(self.taulist_1e7,int(self.bins/5))
        del self.taulist_1e7
    def __plot_xylog_tau_dos_compared2(self):
        xlabel = "tau_trange(1e8,9e8)_compared_different_samples"
        ylabel = "prob"
        title = ylabel +"-"+ xlabel +"_xylog"
        img_path = ylabel +"-"+ xlabel +"_xylog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(title)
        x1 = self.lsp_taulist_range
        y1 = self.dos_taulist_range
        label1 = "trange(1e8,9e8)_tau_1_basin"
        x2 = self.lsp_taulist_range2
        y2 = self.dos_taulist_range2
        label2 = "trange(1e8,9e8)_tau_2_i(i+1)"
        # x3 = self.lsp_taulist_spe
        # y3 = self.dos_taulist_spe
        # label3 = "tau=2e8_tau_3_basin"
        x4 = self.lsp_taulist_spe2
        y4 = self.dos_taulist_spe2
        label4 = "tau=2e8_tau_4_i(i+1)"
        x3 = linspace(10**(np.log10(max(x1))*2/3), max(x1),1000)
        y3 = x3**(-1.0*self.temperature*self.betac)*100
        x5 = linspace(10**(np.log10(max(x1))*2/3), max(x1),1000)
        y5 = x5**(-1.0*self.temperature*self.betac-1)*100
        plt.plot(x1,y1,label=label1,color="blue")
        plt.plot(x2,y2,label=label2,color="green")
        plt.plot(x3,y3,label="-alpha",color="yellow")
        plt.plot(x4,y4,label=label4,color="pink")
        plt.plot(x5,y5,label="-alpha-1",color="black")
        # plt.plot(x3,y3,label="y=x^(-1-index)) alpha={}".format(1.0*self.temperature*self.betac/2),color="pink")
        # plt.plot(x4,y4,label="y=x^(-1-index)) alpha={}".format(1.0*self.temperature*self.betac),color="yellow")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
    def __plot_xylog_tau_dos_compared_different_tw(self):
        xlabel = "tau_different_tw=1e2-1e7"
        ylabel = "pdf"
        title = ylabel +"-"+ xlabel +"_xylog"
        img_path = ylabel +"-"+ xlabel +"_xylog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(title)
        x00 = self.lsp_taulist_1e0
        y00= self.dos_taulist_1e0
        x0 = self.lsp_taulist_1e1
        y0 = self.dos_taulist_1e1

        x1 = self.lsp_taulist_1e2
        y1 = self.dos_taulist_1e2
        x2 = self.lsp_taulist_1e3
        y2 = self.dos_taulist_1e3
        x3 = self.lsp_taulist_1e4
        y3 = self.dos_taulist_1e4
        x4 = self.lsp_taulist_1e5
        y4 = self.dos_taulist_1e5
        x5 = self.lsp_taulist_1e6
        y5 = self.dos_taulist_1e6
        x6 = self.lsp_taulist_1e7
        y6 = self.dos_taulist_1e7

        xl1 = linspace(10**(np.log10(max(x3))*1/6), 10**(np.log10(max(x3))*2/3),1000)
        yl1 = xl1**(-1.0*self.temperature*self.betac)*0.00001
        xl2 = linspace(10**(np.log10(max(x3))*2/3), max(x3),1000)
        yl2 = xl2**(-1.0*self.temperature*self.betac-1)*100


        plt.plot(x00,y00,label="tw=1e0",color="purple")
        plt.plot(x0,y0,label="tw=1e1",color="violet")
        plt.plot(x1,y1,label="tw=1e2",color="blue")
        plt.plot(x2,y2,label="tw=1e3",color="green")
        plt.plot(x3,y3,label="tw=1e4",color="yellow")
        plt.plot(x4,y4,label="tw=1e5",color="pink")
        plt.plot(x5,y5,label="tw=1e6",color="black")
        plt.plot(x6,y6,label="tw=1e7",color="gold")
        plt.plot(xl1,yl1,label="-alpha",color="teal")
        plt.plot(xl2,yl2,label="-alpha-1",color="grey")
        
        # plt.plot(xl2,yl2,label="y=x^(-1-alpha)) alpha={}".format(1.0*self.temperature*self.betac),color="pink")
        # plt.plot(x3,y3,label="y=x^(-1-index)) alpha={}".format(1.0*self.temperature*self.betac/2),color="pink")
        # plt.plot(x4,y4,label="y=x^(-1-index)) alpha={}".format(1.0*self.temperature*self.betac),color="yellow")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
    def __plot_xylog_tau_dos_genfunc_compared_different_tw(self):
        xlabel = "tau_genfunc_different_tw=1e0-1e7"
        ylabel = "pdf"
        title = ylabel +"-"+ xlabel +"_xylog"
        img_path = ylabel +"-"+ xlabel +"_xylog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(title)
        x00 = self.lsp_taulist_1e0
        y00= self.dos_taulist_1e0
        x0 = self.lsp_taulist_1e1
        y0 = self.dos_taulist_1e1

        x1 = self.lsp_taulist_1e2
        y1 = self.dos_taulist_1e2
        x2 = self.lsp_taulist_1e3
        y2 = self.dos_taulist_1e3
        x3 = self.lsp_taulist_1e4
        y3 = self.dos_taulist_1e4
        x4 = self.lsp_taulist_1e5
        y4 = self.dos_taulist_1e5
        x5 = self.lsp_taulist_1e6
        y5 = self.dos_taulist_1e6
        x6 = self.lsp_taulist_1e7
        y6 = self.dos_taulist_1e7

        xl1 = linspace(10**(np.log10(max(x3))*1/6), 10**(np.log10(max(x3))*2/3),1000)
        yl1 = xl1**(-1.0*self.temperature*self.betac)*0.00001
        xl2 = linspace(10**(np.log10(max(x3))*2/3), max(x3),1000)
        yl2 = xl2**(-1.0*self.temperature*self.betac-1)*100


        plt.plot(x00,y00,label="tw=1e0",color="purple")
        plt.plot(x0,y0,label="tw=1e1",color="violet")
        plt.plot(x1,y1,label="tw=1e2",color="blue")
        plt.plot(x2,y2,label="tw=1e3",color="green")
        plt.plot(x3,y3,label="tw=1e4",color="yellow")
        plt.plot(x4,y4,label="tw=1e5",color="pink")
        plt.plot(x5,y5,label="tw=1e6",color="black")
        plt.plot(x6,y6,label="tw=1e7",color="gold")
        plt.plot(xl1,yl1,label="-alpha",color="teal")
        plt.plot(xl2,yl2,label="-alpha-1",color="grey")
        
        # plt.plot(xl2,yl2,label="y=x^(-1-alpha)) alpha={}".format(1.0*self.temperature*self.betac),color="pink")
        # plt.plot(x3,y3,label="y=x^(-1-index)) alpha={}".format(1.0*self.temperature*self.betac/2),color="pink")
        # plt.plot(x4,y4,label="y=x^(-1-index)) alpha={}".format(1.0*self.temperature*self.betac),color="yellow")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()

    def plot_e_tau_dos_mc(self):
        self.read_data_e_tau_mc()
        self.read_tau_t_mc()
        self.lsp_taulist_trange, self.dos_taulist_trange = self.gen_doslist_log(self.taulists_range,int(self.bins/2))
        del self.taulists_range
        self.lsp_taulist_trange2, self.dos_taulist_trange2 = self.gen_doslist_log(self.taulists_range2,int(self.bins/2))
        del self.taulists_range2
        self.lsp_taulist_trange3, self.dos_taulist_trange3 = self.gen_doslist_log(self.taulists_range3,int(self.bins/2))
        del self.taulists_range3

        self.lsp_e0list_trange, self.dos_e0list_trange = self.gen_doslist(self.e0lists_range,int(self.bins/2))
        del self.e0lists_range
        self.lsp_eslist_trange, self.dos_eslist_trange = self.gen_doslist(self.eslists_range,int(self.bins/2))
        del self.eslists_range
        self.lsp_eblist_trange, self.dos_eblist_trange = self.gen_doslist(self.eblists_range,int(self.bins/2))
        del self.eblists_range
        self.lsp_arrhtau_trange, self.dos_arrhtau_trange = self.treeArrhenius(self.lsp_eblist_trange, self.dos_eblist_trange)

        self.__plot("e0_range(1e8,9e8)"
                ,"prob"
                ,self.lsp_e0list_trange
                ,self.dos_e0list_trange
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("es_range(1e8,9e8)"
                ,"prob"
                ,self.lsp_eslist_trange
                ,self.dos_eslist_trange
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("eb_range(1e8,9e8)"
                ,"prob"
                ,self.lsp_eblist_trange
                ,self.dos_eblist_trange
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_ylog("eb_range(1e8,9e8)"
                ,"prob"
                ,self.lsp_eblist_trange
                ,self.dos_eblist_trange
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xylog_e_tau_dos_mc()
        self.__plot_xylog_tau_dos_compared()
    def __plot_xylog_e_tau_dos_mc(self):
        xlabel = "tau_trange(1e8,9e8)"
        ylabel = "prob"
        title = ylabel +"-"+ xlabel +"_xylog"
        img_path = ylabel +"-"+ xlabel +"_xylog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(title)
        x1 = self.lsp_taulist_trange
        y1 = self.dos_taulist_trange
        label1 = "trange(1e8,9e8)_tau"
        x2 = self.lsp_arrhtau_trange
        y2 = self.dos_arrhtau_trange
        label2 = "trange(1e8,9e8)_eb+Arrhenius"
        x3 = linspace(10**(np.log10(max(x1))*2/3), max(x1),1000)
        y3 = x3**(-1.0*self.temperature*self.betac*0.1-1)*100
        x4 = linspace(10**(np.log10(max(x1))*2/3), max(x1),1000)
        y4 = x4**(-1.0*self.temperature*self.betac-1)*100
        plt.plot(x1,y1,label=label1,color="blue")
        plt.plot(x2,y2,label=label2,color="green")
        plt.plot(x3,y3,label="y=x^(-1-index)) alpha={}".format(1.0*self.temperature*self.betac/2),color="pink")
        plt.plot(x4,y4,label="y=x^(-1-index)) alpha={}".format(1.0*self.temperature*self.betac),color="yellow")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
    def __plot_xylog_tau_dos_compared(self):
        xlabel = "tau_trange(1e8,9e8)_compared"
        ylabel = "prob"
        title = ylabel +"-"+ xlabel +"_xylog"
        img_path = ylabel +"-"+ xlabel +"_xylog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(title)
        x1 = self.lsp_taulist_trange
        y1 = self.dos_taulist_trange
        label1 = "trange(1e8,9e8)_tau_1_basin"
        x2 = self.lsp_taulist_trange2
        y2 = self.dos_taulist_trange2
        label2 = "trange(1e8,9e8)_tau_2_i(i+1)"
        x3 = self.lsp_taulist_trange3
        y3 = self.dos_taulist_trange3
        label3 = "trange(1e8,9e8)_tau_3_equal_interval"
        # x3 = linspace(10**(np.log10(max(x1))*2/3), max(x1),1000)
        # y3 = x3**(-1.0*self.temperature*self.betac*0.1-1)*100
        # x4 = linspace(10**(np.log10(max(x1))*2/3), max(x1),1000)
        # y4 = x4**(-1.0*self.temperature*self.betac-1)*100
        plt.plot(x1,y1,label=label1,color="blue")
        plt.plot(x2,y2,label=label2,color="green")
        plt.plot(x3,y3,label=label3,color="yellow")
        # plt.plot(x3,y3,label="y=x^(-1-index)) alpha={}".format(1.0*self.temperature*self.betac/2),color="pink")
        # plt.plot(x4,y4,label="y=x^(-1-index)) alpha={}".format(1.0*self.temperature*self.betac),color="yellow")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
        


    def read_tau_t(self):
        self.time_model = [i*(i+1) for i in range(self.__time_to_int(1e9))]
        self.taut_model = [0 for j in range(self.__time_to_int(1e9))]
        self.taut_model_1sample = [0 for j in range(self.__time_to_int(1e9))]
        self.taulists_range = []
        self.tau0lists_range_pair = []
        self.tau1lists_range_pair = []
        self.timelists_model = []
        self.taulists_model = []
        self.taut1lists = []
        self.taut2lists = []
        self.taut3lists = []
        self.taut4lists = []

        for i in range(self.start, self.end):
            self.taulist_range = []
            j=0
            tau_str = "tau-t_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            with open(self.dir + tau_str,"r",encoding='utf-8') as fp:
                while 1:                    
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    (time,tau) = [t(s) for t,s in zip((float,float),linedata.split())]
                    if time>1e9:
                        break
                    # if time>1e8 and time<9*1e8:
                    #     self.taulists_range.append(tau)
                    #     self.taulist_range.append(tau)
                    if time > 0:
                        self.timelists_model.append(time)
                        self.taulists_model.append(tau)
                    if time > 0 and time <= 100:
                        self.taut1lists.append(tau)
                    if time >100 and time <= 1e4:
                        self.taut2lists.append(tau)
                    if time >1e4 and time <= 1e6:  
                        self.taut3lists.append(tau) 
                    if time >1e6 and time <= 1e8:  
                        self.taut4lists.append(tau)  
                    self.taut_model[j] += tau/(self.end - self.start)
            #         self.taut_model_1sample[j] = tau
                    j+=1
            # self.taut_model_1sample = self.taut_model_1sample[:-1]
            # self.tau0lists_range_pair.extend(self.taulist_range[:-1])
            # self.tau1lists_range_pair.extend(self.taulist_range[1:])
            # if i<10:
            #     self.__plot_tau_t_1sample(i)
            # self.taut_model_1sample.append(0)
        # self.taut_model = self.taut_model[:-1]
    def plot_tau_dos(self):
        self.read_tau_t()
        self.lsp_taulist_trange, self.dos_taulist_trange = self.gen_doslist_log(self.taulists_range,int(self.bins/2))
        del self.taulists_range
        self.__plot_xylog_tau("tau_range(1e8,9e8)"
                ,"pdf"
                ,self.lsp_taulist_trange
                ,self.dos_taulist_trange
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("tau_range(1e8,9e8)"
                ,"pdf"
                ,self.lsp_taulist_trange
                ,self.dos_taulist_trange
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        dynamicstau_range_str = self.dir + "p_dynamics(tau)_range(1e8,9e8)_{model}{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model)
        self.printf(self.lsp_taulist_trange, self.dos_taulist_trange, dynamicstau_range_str)

        self.deal_with_taulists_range_pair()
    def deal_with_taulists_range_pair(self):
        prob, x, y, m = self.__hist2d_xylog("tau0"
            ,"tau1"
            ,"prob density_xylog"
            ,self.tau0lists_range_pair
            ,self.tau1lists_range_pair
            ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
            ,self.__imgpath("tau0-tau1_pairprob_range(1e8,9e8)_xylog")
            ,self.bins + 1)
        
        prob_str = self.dir + "prob_dynamics(tau)_range(1e8,9e8)_{model}{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model)
        tau_str = self.dir + "tau0-tau1_prob_dynamics_range(1e8,9e8)_{model}{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model)
        self.printf2d(x,y,prob,tau_str,prob_str)
    def plot_tau_specific_time_dos(self):
        self.lsp_taut1list, self.dos_taut1list = self.gen_doslist_log(self.taut1lists,int(self.bins))
        self.lsp_taut2list, self.dos_taut2list = self.gen_doslist_log(self.taut2lists,int(self.bins))
        self.lsp_taut3list, self.dos_taut3list = self.gen_doslist_log(self.taut3lists,int(self.bins))
        self.lsp_taut4list, self.dos_taut4list = self.gen_doslist_log(self.taut4lists,int(self.bins))
        self.__plot_tau_specific_time_dos()

    def plot_taut_dos(self):
        self.read_tau_t()
        self.plot_tau_specific_time_dos()
        # prob, x, y, m = self.__hist2d_xylog("time"
        #     ,"tau"
        #     ,"prob density_xylog"
        #     ,self.timelists_model
        #     ,self.taulists_model
        #     ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        #     ,self.__imgpath("time-tau_prob_xylog")
        #     ,self.bins + 1)
        
        # prob_str = self.dir + "prob_dynamics(tau)_range(1e8,9e8)_{model}{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model)
        # tau_str = self.dir + "tau0-tau1_prob_dynamics_range(1e8,9e8)_{model}{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model)
        # self.printf2d(x,y,prob,tau_str,prob_str)


    def read_tau_t_hopping_simulaion(self):
        self.time_hopping = [i*(i+1) for i in range(self.__time_to_int(1e9) )]
        self.taut_hopping = [0 for j in range(self.__time_to_int(1e9) )]
        for i in range(self.start, self.end):
            self.taulist_range = []
            j=0
            tau_str = "tau-t_hopping_analytics_smirnov_bins={bins}_trct={trct}_T={b}_timemax={c}_{d}.t".format(
                bins="%.0le"%self.bins,trct="%.0le"%self.trct,a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i)
            with open(self.dir + tau_str,"r",encoding='utf-8') as fp:
                while 1:                    
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    (time,tau) = [t(s) for t,s in zip((float,float),linedata.split())]
                    if time>1e9:
                        break
                    self.taut_hopping[j] += tau/(self.end - self.start)
                    j+=1 
    def plot_tau_t_ana_smirnov(self,taum,tau0,bins,trct):
        self.bins = bins
        self.trct = trct
        mkdir(self.dir + "plot/tau-t")
        self.read_tau_t()
        self.read_tau_t_hopping_simulaion_smirnov()
        self.taum=taum
        self.tau0=tau0

        self.__plot_xylog_tau_t("time_vs_analytic_bins={bins}_trct={trct}".format(bins="%.0le"%bins,trct="%.0le"%trct)
                ,"trapping_time"
                ,self.time_model
                ,self.taut_model
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))


    def gen_tau_dos_genfunc_compared(self):
        self.lsp_taulist_1e0, self.dos_taulist_1e0 = self.gen_doslist_log(self.taulist_1e0,int(self.bins/10))
        del self.taulist_1e0
        self.lsp_taulist_1e1, self.dos_taulist_1e1 = self.gen_doslist_log(self.taulist_1e1,int(self.bins/10))
        del self.taulist_1e1
        self.lsp_taulist_1e2, self.dos_taulist_1e2 = self.gen_doslist_log(self.taulist_1e2,int(self.bins/10))
        del self.taulist_1e2
        self.lsp_taulist_1e3, self.dos_taulist_1e3 = self.gen_doslist_log(self.taulist_1e3,int(self.bins/10))
        del self.taulist_1e3
        self.lsp_taulist_1e4, self.dos_taulist_1e4 = self.gen_doslist_log(self.taulist_1e4,int(self.bins/10))
        del self.taulist_1e4
        self.lsp_taulist_1e5, self.dos_taulist_1e5 = self.gen_doslist_log(self.taulist_1e5,int(self.bins/10))
        del self.taulist_1e5
        self.lsp_taulist_1e6, self.dos_taulist_1e6 = self.gen_doslist_log(self.taulist_1e6,int(self.bins/10))
        del self.taulist_1e6
        self.lsp_taulist_1e7, self.dos_taulist_1e70 = self.gen_doslist_log(self.taulist_1e7,int(self.bins/10))
        del self.taulist_1e7
    def read_tau_t_genfunc(self):
        self.taulist_1e0 = []
        self.taulist_1e1 = []
        self.taulist_1e2 = []
        self.taulist_1e3 = []
        self.taulist_1e4 = []
        self.taulist_1e5 = []
        self.taulist_1e6 = []
        self.taulist_1e7 = []
        bins = 1e+08
        trct = 1e+08
        for i in range(self.start, self.end):
            # self.taulist_range = []
            j=0
            tau_str = "tau-t_hopping_genfunc_mixed_bins={bins}_trct={trct}_T={b}_timemax={c}_{d}.t".format(bins="%.0le"%bins,trct="%.0le"%trct,a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i)
            with open(self.dir + tau_str,"r",encoding='utf-8') as fp:
                while 1:                    
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    (time,tau) = [t(s) for t,s in zip((float,float),linedata.split())]

                    if time>1e9:
                        break
                    if time == 0:
                        self.taulist_1e0.append(float(tau))
                    if time == 12:
                        self.taulist_1e1.append(float(tau))
                    if time == 110:
                        self.taulist_1e2.append(float(tau))
                    if time == 1056:
                        self.taulist_1e3.append(float(tau))
                    if time == 10100:
                        self.taulist_1e4.append(float(tau))
                    if time == 100172:
                        self.taulist_1e5.append(float(tau))
                    if time == 1001000:
                        self.taulist_1e6.append(float(tau))
                    if time == 10001406:
                        self.taulist_1e7.append(float(tau))
                    # if time == 200010306:
                    #     self.taulist_spe2.append(float(tau))
        print("finish reading tau-t hopping.")
    def get_Eb_dos_Arrhenius_compared_different_tw(self):
        self.lsp_treetau, self.dos_treetau = self.treeArrhenius(1, dos_treeleb)
    def treeArrhenius(self,lsp,dos):
        tau0 = 1
        lspout = [tau0*np.exp(ele/self.temperature) for ele in lsp]
        dosout = [dos[i]*self.temperature/lspout[i] for i in range(len(dos))]
        return lspout, dosout
    def treeArrhenius_EbTotau(self,tau0,lsp,dos):
        lspout = [self.temperature*np.log(ele/tau0) for ele in lsp]
        dosout = [lsp[i]*dos[i]/self.temperature for i in range(len(dos))]
        return lspout, dosout
    def plot_tau_dos_genfunc_compared_different_tw(self):
        self.read_tau_t_genfunc()
        self.gen_tau_dos_genfunc_compared()
        self.__plot_xylog_tau_dos_genfunc_compared_different_tw()
    def read_tau_t_hopping_genfunc_mixed(self):
        self.time_hopping = [i*(i+1) for i in range(self.__time_to_int(1e9) )]
        self.taut_hopping = [0 for j in range(self.__time_to_int(1e9) )]

        for i in range(self.start, self.end):
            self.taulist_range = []
            j=0
            tau_str = "tau-t_hopping_genfunc_mixed_bins={bins}_trct={trct}_T={b}_timemax={c}_{d}.t".format(bins="%.0le"%self.bins,trct="%.0le"%self.trct,a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i)
            with open(self.dir + tau_str,"r",encoding='utf-8') as fp:
                while 1:                    
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    (time,tau) = [t(s) for t,s in zip((float,float),linedata.split())]
                    if time>1e9:
                        break
                    self.taut_hopping[j] += tau/(self.end - self.start)
                    j+=1 
    def read_tau_t_hopping_fit_genfunc_mixed(self):
        self.time_hopping_fit = [i*(i+1) for i in range(self.__time_to_int(1e9) )]
        self.taut_hopping_fit = [0 for j in range(self.__time_to_int(1e9) )]
        for i in range(self.start, self.end):
            self.taulist_range = []
            j=0
            tau_str = "tau-t_hopping_fit_genfunc_mixed_bins={bins}_trct={trct}_T={b}_timemax={c}_{d}.t".format(bins="%.0le"%self.bins,trct="%.0le"%self.trct,a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i)
            with open(self.dir + tau_str,"r",encoding='utf-8') as fp:
                while 1:                    
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    (time,tau) = [t(s) for t,s in zip((float,float),linedata.split())]
                    if time>1e9:
                        break
                    self.taut_hopping_fit[j] += tau/(self.end - self.start)
                    j+=1 
    def plot_tau_t_genfunc_mixed(self,taum,tau0,bins,trct):
        self.bins = bins
        self.trct = trct
        mkdir(self.dir + "plot/tau-t")
        self.read_tau_t()
        self.read_tau_t_hopping_genfunc_mixed()
        self.read_tau_t_hopping_fit_genfunc_mixed()
        self.taum=taum
        self.tau0=tau0

        self.__plot_xylog_tau_t_genfunc_mixed("time_vs_genfunc_mixed_bins={bins}_trct={trct}".format(bins="%.0le"%bins,trct="%.0le"%trct)
                ,"trapping_time"
                ,self.time_model
                ,self.taut_model
                ,"MC {model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
    def __plot_xylog_tau_t_genfunc_mixed(self,xlabel,ylabel,x,y,label):
        title = ylabel +"-"+ xlabel +"_xylog"
        img_path = ylabel +"-"+ xlabel +"_xylog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(title)
        t=linspace(1, self.taum,100000)
        # y2=self.tau_t_analytic1(t,self.taum)
        # y3=self.tau_t_analytic2(t,self.taum,self.tau0)
        plt.xlim(1,8e8)
        x4=self.time_hopping
        y4=self.taut_hopping
        x5=self.time_hopping_fit
        y5=self.taut_hopping_fit
        plt.plot(x,y,label=label,color="blue")
        # plt.plot(t,y2,label="ana1",color="green")
        # plt.plot(t,y3,label="ana2",color="pink")
        plt.plot(x4,y4,label="hopping simulation",color="gold")
        plt.plot(x5,y5,label="hopping simulation by fitting with Levy distribution",color="tan")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()


    def read_tau_t_hopping_simulaion(self):
        self.time_hopping = [i*(i+1) for i in range(self.__time_to_int(1e9) )]
        self.taut_hopping = [0 for j in range(self.__time_to_int(1e9) )]
        for i in range(self.start, self.end):
            self.taulist_range = []
            j=0
            tau_str = "tau-t_hopping_analytics_bins={bins}_trct={trct}_T={b}_timemax={c}_{d}.t".format(bins="%.0le"%self.bins,trct="%.0le"%self.trct,a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i)
            with open(self.dir + tau_str,"r",encoding='utf-8') as fp:
                while 1:                    
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    (time,tau) = [t(s) for t,s in zip((float,float),linedata.split())]
                    if time>1e9:
                        break
                    self.taut_hopping[j] += tau/(self.end - self.start)
                    j+=1 
    def read_tau_t_hopping_simulaion_log(self):
        self.time_hopping = [i*(i+1) for i in range(self.__time_to_int(1e9) )]
        self.taut_hopping = [0 for j in range(self.__time_to_int(1e9) )]
        for i in range(self.start, self.end):
            self.taulist_range = []
            j=0
            tau_str = "tau-t_hopping_analytics_log_bins={bins}_trct={trct}_T={b}_timemax={c}_{d}.t".format(bins="%.0le"%self.bins,trct="%.0le"%self.trct,a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i)
            with open(self.dir + tau_str,"r",encoding='utf-8') as fp:
                while 1:                    
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    (time,tau) = [t(s) for t,s in zip((float,float),linedata.split())]
                    if time>1e9:
                        break
                    self.taut_hopping[j] += tau/(self.end - self.start)
                    j+=1                    
    def tau_t_analytic1(self,t,taum):
        x=self.betac * self.temperature
        return ((2.0-x)*(taum)**(1-x)*t**(x) - t)/(2.0-x)/(1.0-x)
    def tau_t_analytic2(self,t,taum,tau0):
        x=self.betac * self.temperature
        logC=-1.0*np.log(t**(-x+1.0) - (1-x)*tau0**(-x+1.0) - x*t*taum**(-x))
        logup=np.log(-t**(-x+2.0) + (2.0-x)*t*taum**(-x+1.0))
        return np.exp(logC+logup)/(1.0-x)/(2.0-x)
    def __plot_xylog_tau_t(self,xlabel,ylabel,x,y,label):
        title = ylabel +"-"+ xlabel +"_xylog"
        img_path = ylabel +"-"+ xlabel +"_xylog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(title)
        plt.xlim(1,5e8)
        t=linspace(1, self.taum,100000)
        y2=self.tau_t_analytic1(t,self.taum)
        y3=self.tau_t_analytic2(t,self.taum,self.tau0)
        x4=self.time_hopping
        y4=self.taut_hopping
        plt.plot(x,y,label=label,color="blue")
        # plt.plot(t,y2,label="ana1",color="green")
        # plt.plot(t,y3,label="ana2",color="pink")
        plt.plot(x4,y4,label="hopping simulation",color="gold")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
    def plot_tau_t_ana(self,taum,tau0,bins,trct):
        self.bins = bins
        self.trct = trct
        mkdir(self.dir + "plot/tau-t")
        self.read_tau_t()
        self.read_tau_t_hopping_simulaion()
        self.taum=taum
        self.tau0=tau0
        # self.__plot("time"
        #         ,"trapping_time"
        #         ,self.time_model
        #         ,self.taut_model
        #         ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        # self.__plot_xlog("time"
        #         ,"trapping_time"
        #         ,self.time_model
        #         ,self.taut_model
        #         ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xylog_tau_t("time_vs_analytic_bins={bins}_trct={trct}".format(bins="%.0le"%bins,trct="%.0le"%trct)
                ,"trapping_time"
                ,self.time_model
                ,self.taut_model
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
    def plot_tau_t_ana_log(self,taum,tau0,bins,trct):
        self.bins = bins
        self.trct = trct
        mkdir(self.dir + "plot/tau-t")
        self.read_tau_t()
        self.read_tau_t_hopping_simulaion_log()
        self.taum=taum
        self.tau0=tau0

        self.__plot_xylog_tau_t("time_vs_analytic_bins={bins}_trct={trct}".format(bins="%.0le"%bins,trct="%.0le"%trct)
                ,"trapping_time"
                ,self.time_model
                ,self.taut_model
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
    def plot_tau_t(self):
        mkdir(self.dir + "plot/tau-t")
        self.read_tau_t()
        
        # self.__plot("time"
        #         ,"trapping_time"
        #         ,self.time_model
        #         ,self.taut_model
        #         ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        # self.__plot_xlog("time"
        #         ,"trapping_time"
        #         ,self.time_model
        #         ,self.taut_model
        #         ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xylog("time"
                ,"trapping_time"
                ,self.time_model
                ,self.taut_model
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
    def __plot_tau_t_1sample(self,i):
        self.__plot_1sample("time", "tau", self.time_model, self.taut_model_1sample
                        , "{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model),i,"tau-t")
        self.__plot_1sample_xlog("time", "tau", self.time_model, self.taut_model_1sample
                        , "{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model),i,"tau-t")
        self.__plot_1sample_xylog("time", "tau", self.time_model, self.taut_model_1sample
                        , "{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model),i,"tau-t")
    def __plot_tau_specific_time_dos(self):
        self.__plot_xylog_tau("tau_trange(1,1e2)"
                ,"pdf"
                ,self.lsp_taut1list
                ,self.dos_taut1list
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("tau_trange(1,1e2)"
                ,"pdf"
                ,self.lsp_taut1list
                ,self.dos_taut1list
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

        self.__plot_xylog_tau("tau_trange(1e2,1e4)"
                ,"pdf"
                ,self.lsp_taut2list
                ,self.dos_taut2list
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("tau_trange(1e2,1e4)"
                ,"pdf"
                ,self.lsp_taut2list
                ,self.dos_taut2list
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

        self.__plot_xylog_tau("tau_trange(1e4,1e6)"
                ,"pdf"
                ,self.lsp_taut3list
                ,self.dos_taut3list
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("tau_trange(1e4,1e6)"
                ,"pdf"
                ,self.lsp_taut3list
                ,self.dos_taut3list
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

        self.__plot_xylog_tau("tau_trange(1e6,1e8)"
                ,"pdf"
                ,self.lsp_taut4list
                ,self.dos_taut4list
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("tau_trange(1e6,1e8)"
                ,"pdf"
                ,self.lsp_taut4list
                ,self.dos_taut4list
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

        


    def read_tau_t_corr_dynamics(self):
        self.time_corr_dynamics = [i*(i+1) for i in range(self.__time_to_int(1e8))]
        self.taut_corr_dynamics = [0 for j in range(self.__time_to_int(1e8))]
        self.taut_corr_dynamics_1sample = [0 for j in range(self.__time_to_int(1e8))]

        for i in range(self.start, self.end):
            j=0
            tau_str = "tau-t_corr_dynamics_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            with open(self.dir + tau_str,"r",encoding='utf-8') as fp:
                while 1:                    
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    (time,tau) = [t(s) for t,s in zip((float,float),linedata.split())]
                    if time>1e8:
                        break
                    self.taut_corr_dynamics[j] += tau/(self.end - self.start)
                    self.taut_corr_dynamics_1sample[j] = tau
                    j+=1
            if i<200:
                self.__plot_tau_t_corr_dynamics_1sample(i)    
    def plot_tau_t_corr_dynamics(self):
        mkdir(self.dir + "/plot/tau-t_corr_dynamics")
        self.read_tau_t_corr_dynamics()
        self.__plot("time_corr_dynamics"
                ,"trapping_time"
                ,self.time_corr_dynamics
                ,self.taut_corr_dynamics
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xlog("time_corr_dynamics"
                ,"trapping_time"
                ,self.time_corr_dynamics
                ,self.taut_corr_dynamics
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xylog("time_corr_dynamics"
                ,"trapping_time"
                ,self.time_corr_dynamics
                ,self.taut_corr_dynamics
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
    def __plot_tau_t_corr_dynamics_1sample(self,i):
        self.__plot_1sample("time", "tau", self.time_corr_dynamics, self.taut_corr_dynamics_1sample
                        , "{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model),i,"tau-t_corr_dynamics")
        self.__plot_1sample_xlog("time", "tau", self.time_corr_dynamics, self.taut_corr_dynamics_1sample
                        , "{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model),i,"tau-t_corr_dynamics")
        self.__plot_1sample_xylog("time", "tau", self.time_corr_dynamics, self.taut_corr_dynamics_1sample
                        , "{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model),i,"tau-t_corr_dynamics")

    def read_tau_t_corr_dynamics_trange(self):
        self.time_corr_dynamics_trange = [i*(i+1) for i in range(self.__time_to_int(1e9))]
        self.taut_corr_dynamics_trange = [0 for j in range(self.__time_to_int(1e9))]
        self.taut_corr_dynamics_trange_1sample = [0 for j in range(self.__time_to_int(1e9))]

        for i in range(self.start, self.end):
            j=0
            tau_str = "tau-t_corr_dynamics_trange(1e8,9e8)_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            with open(self.dir + tau_str,"r",encoding='utf-8') as fp:
                while 1:                    
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    (time,tau) = [t(s) for t,s in zip((float,float),linedata.split())]
                    if time>1e9:
                        break
                    self.taut_corr_dynamics_trange[j] += tau/(self.end - self.start)
                    self.taut_corr_dynamics_trange_1sample[j] = tau
                    j+=1
            if i<200:
                self.__plot_tau_t_corr_dynamics_trange_1sample(i)    
    def plot_tau_t_corr_dynamics_trange(self):
        mkdir(self.dir + "/plot/tau-t_corr_dynamics_trange(1e8,9e8)")
        self.read_tau_t_corr_dynamics_trange()
        self.__plot("time_corr_dynamics_trange(1e8,9e8)"
                ,"trapping_time"
                ,self.time_corr_dynamics_trange
                ,self.taut_corr_dynamics_trange
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xlog("time_corr_dynamics_trange(1e8,9e8)"
                ,"trapping_time"
                ,self.time_corr_dynamics_trange
                ,self.taut_corr_dynamics_trange
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xylog("time_corr_dynamics_trange(1e8,9e8)"
                ,"trapping_time"
                ,self.time_corr_dynamics_trange
                ,self.taut_corr_dynamics_trange
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
    def __plot_tau_t_corr_dynamics_trange_1sample(self,i):
        self.__plot_1sample("time", "tau", self.time_corr_dynamics_trange, self.taut_corr_dynamics_trange_1sample
                        , "{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model),i,"tau-t_corr_dynamics_trange(1e8,9e8)")
        self.__plot_1sample_xlog("time", "tau", self.time_corr_dynamics_trange, self.taut_corr_dynamics_trange_1sample
                        , "{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model),i,"tau-t_corr_dynamics_trange(1e8,9e8)")
        self.__plot_1sample_xylog("time", "tau", self.time_corr_dynamics_trange, self.taut_corr_dynamics_trange_1sample
                        , "{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model),i,"tau-t_corr_dynamics_trange(1e8,9e8)")

    def plot_gentau(self):
        self.read_gentau()
        self.__plot("Delta_gen"
                ,"remaining prob(Delta,{t})".format(t="%.0le"%self.begint)
                ,self.ageing_func_timelist_gen
                ,self.ageing_func_rate_gen
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xlog("Delta_gen"
                ,"remaining prob(Delta,{t})".format(t="%.0le"%self.begint)
                ,self.ageing_func_timelist_gen
                ,self.ageing_func_rate_gen
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xylog("Delta_gen"
                ,"remaining prob(Delta,{t})".format(t="%.0le"%self.begint)
                ,self.ageing_func_timelist_gen
                ,self.ageing_func_rate_gen
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
    def read_gentau(self):
        j=0
        dir_str = self.dir
        ageing_func_hit_gen = np.array([0 for i in range(self.__time_to_int(self.timemax - self.begint))])
        ageing_func_total_gen = np.array([0 for i in range(self.__time_to_int(self.timemax - self.begint))])
        for i in range(self.start, self.end):
            j=0
            tau_str = "C(Delta,1e4)_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            with open(dir_str + tau_str,"r",encoding='utf-8') as fp:
                while 1:
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    (delta,Enum) = [t(s) for t,s in zip((int,int),linedata.split())]
                    ageing_func_hit_gen[j] += Enum
                    j+=1
            ageing_func_total_gen += 1
                   
        self.ageing_func_rate_gen = [ageing_func_hit_gen[i]/ageing_func_total_gen[i] for i in range(self.__time_to_int(self.timemax/2)) if ageing_func_total_gen[i]!=0]        
        self.ageing_func_timelist_gen = [i*(i+1)+self.begint for i in range(self.__time_to_int(self.timemax/2)) if ageing_func_total_gen[i]!=0]

    def plot_gentau_allrange(self):
        self.read_gentau_allrange()
        self.__plot("Delta_gen_allrange"
                ,"remaining prob(Delta,{t})".format(t="%.0le"%self.begint)
                ,self.ageing_func_timelist_gen_allrange
                ,self.ageing_func_rate_gen_allrange
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xlog("Delta_gen_allrange"
                ,"remaining prob(Delta,{t})".format(t="%.0le"%self.begint)
                ,self.ageing_func_timelist_gen_allrange
                ,self.ageing_func_rate_gen_allrange
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xylog("Delta_gen_allrange"
                ,"remaining prob(Delta,{t})".format(t="%.0le"%self.begint)
                ,self.ageing_func_timelist_gen_allrange
                ,self.ageing_func_rate_gen_allrange
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
    def read_gentau_allrange(self):
        j=0
        dir_str = self.dir
        ageing_func_hit_gen = np.array([0 for i in range(self.__time_to_int(self.timemax - self.begint))])
        ageing_func_total_gen = np.array([0 for i in range(self.__time_to_int(self.timemax - self.begint))])
        for i in range(self.start, self.end):
            j=0
            tau_str = "C(Delta,1e4)_tau_allrange_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            with open(dir_str + tau_str,"r",encoding='utf-8') as fp:
                while 1:
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    (delta,Enum) = [t(s) for t,s in zip((int,int),linedata.split())]
                    ageing_func_hit_gen[j] += Enum
                    j+=1
            ageing_func_total_gen += 1
                   
        self.ageing_func_rate_gen_allrange = [ageing_func_hit_gen[i]/ageing_func_total_gen[i] for i in range(self.__time_to_int(self.timemax/2)) if ageing_func_total_gen[i]!=0]        
        self.ageing_func_timelist_gen_allrange = [i*(i+1)+self.begint for i in range(self.__time_to_int(self.timemax/2)) if ageing_func_total_gen[i]!=0]
    def read_gentau_trange(self):
        j=0
        dir_str = self.dir
        ageing_func_hit_gen = np.array([0 for i in range(self.__time_to_int(self.timemax)*2)])
        ageing_func_total_gen = np.array([0 for i in range(self.__time_to_int(self.timemax)*2)])
        for i in range(self.start, self.end):
            j=0
            #tau_str = "C(Delta,0)_random_init_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model,t="%.0le"%self.begint)
            tau_str = "C(Delta,{tw})_range(1e8,9e8)_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model,tw="%.0le"%self.begint)
            with open(dir_str + tau_str,"r",encoding='utf-8') as fp:
                while 1:
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    (delta,Enum) = [t(s) for t,s in zip((int,int),linedata.split())]
                    ageing_func_hit_gen[j] += Enum
                    j+=1
            ageing_func_total_gen += 1             
        self.ageing_func_rate_gen_trange = [ageing_func_hit_gen[i]/ageing_func_total_gen[i] for i in range(self.__time_to_int(self.timemax/2)) if ageing_func_total_gen[i]!=0]        
        self.ageing_func_timelist_gen_trange = [i*(i+1)+self.begint for i in range(self.__time_to_int(self.timemax/2)) if ageing_func_total_gen[i]!=0]
    def plot_gentau_trange(self):
        self.read_gentau_trange()
        self.__plot("Delta_gen_trange(1e8,9e8)"
                ,"remaining prob(Delta,{t})".format(t="%.0le"%self.begint)
                ,self.ageing_func_timelist_gen_trange
                ,self.ageing_func_rate_gen_trange
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xlog("Delta_gen_trange(1e8,9e8)"
                ,"remaining prob(Delta,{t})".format(t="%.0le"%self.begint)
                ,self.ageing_func_timelist_gen_trange
                ,self.ageing_func_rate_gen_trange
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xylog("Delta_gen_trange(1e8,9e8)"
                ,"remaining prob(Delta,{t})".format(t="%.0le"%self.begint)
                ,self.ageing_func_timelist_gen_trange
                ,self.ageing_func_rate_gen_trange
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
    def read_corrfunc_mc(self):
        ageing_func_hit_beginfixed = np.array([0 for i in range(self.__time_to_int2(self.timemax)*2)])
        ageing_func_total_beginfixed = np.array([0 for i in range(self.__time_to_int2(self.timemax)*2)])

        for i in range(self.start,self.end):
            self.taulist = []
            self.eb1list = []
            time = 0
            jlast_last = 0
            act_str = "activation-t_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            with open(self.dir + act_str,'r') as fp:
                #fp.readline()
                while 1:
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    else:
                        # if(len(linedata.split())<2):
                        #     print(i)
                        timelast = time
                        try:
                            (time,e0,es,e1,tip0,saddle,tip1,effe0,effes,effe1,effs,tau0,taus, tau) = [t(s) for t,s in zip((int,float,float,float,int,int,int,float,float,float,int,int,int,int),linedata.split())]
                        except:
                            print(act_str)

                        if time>self.begint and self.begint>=timelast:
                            j_beginfixed = self.__time_to_int2(time - self.begint)
                            jlist_beginfixed = [k for k in range(0,j_beginfixed)]
                            ageing_func_hit_beginfixed[jlist_beginfixed] += 1
                            ageing_func_total_beginfixed += 1

        self.ageing_func_rate_beginfixed = [ageing_func_hit_beginfixed[i]/ageing_func_total_beginfixed[i] for i in range(self.__time_to_int(self.timemax/2)) if ageing_func_total_beginfixed[i]!=0]        
        self.ageing_func_timelist_beginfixed = [i*(i+1)/2+self.begint for i in range(self.__time_to_int(self.timemax/2)) if ageing_func_total_beginfixed[i]!=0]
        print("finish: read and construct data from dynamics.")
    def plot_corrfunc_compared_img(self,tw):
        self.begint = tw
        self.read_gentau_trange()
        self.read_corrfunc_mc()
        self.__plot_corrfunc_compared_xlog()
        self.__plot_corrfunc_compared_xylog()
    def __plot_corrfunc_compared_xlog(self):
        xlabel = "Delta"
        ylabel = "remaining prob(Delta,{t})".format(t="%.0le"%self.begint)
        title = ylabel +"-"+ xlabel+"_compared_xlog"
        img_path = ylabel +"-"+ xlabel +"_compared_xlog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.title(title)
        x1 = self.ageing_func_timelist_beginfixed
        y1 = self.ageing_func_rate_beginfixed
        label1 = "MC_{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        x2 = self.ageing_func_timelist_gen_trange
        y2 = self.ageing_func_rate_gen_trange
        label2 = "TrapModel_trange(1e8,9e8)_{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        plt.plot(x1,y1,label=label1,color="pink")
        plt.plot(x2,y2,label=label2,color="blue")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
    def __plot_corrfunc_compared_xylog(self):
        xlabel = "Delta"
        ylabel = "remaining prob(Delta,{t})".format(t="%.0le"%self.begint)
        title = ylabel +"-"+ xlabel +"_compared_xylog"
        img_path = ylabel +"-"+ xlabel +"_compared_xylog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(title)
        x1 = self.ageing_func_timelist_beginfixed
        y1 = self.ageing_func_rate_beginfixed
        label1 = "MC_{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        x2 = self.ageing_func_timelist_gen_trange
        y2 = self.ageing_func_rate_gen_trange
        label2 = "TrapModel_trange(1e8,9e8)_{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        plt.plot(x1,y1,label=label1,color="pink")
        plt.plot(x2,y2,label=label2,color="blue")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()



####
    def read_taulist_hopping_debug(self,di,time):
        self.taulist_hopping = []
        i=0
        taulist_str = "taulist_hopping_debug_n={di}_t={time}_{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,di="%.0e"%di,time="%.0e"%time)
        with open(self.dir + taulist_str,"r",encoding='utf-8') as fp:
            while 1:
                linedata = fp.readline()
                if linedata.strip() == '':
                    break
                self.taulist_hopping.append(double(linedata))
    def plot_dostau_hopping_debug(self,time,di):
        self.read_taulist_hopping(di,time)
        lsp, dos = self.gen_doslist_log(self.taulist_hopping,int(self.bins/4))
        self.__plot_xylog_tau2("tau_hopping_debug_t={time}_n={di}".format(time="%.0le"%time,di="%.0e"%di)
                ,"pdf"
                ,lsp
                ,dos
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
####

    def read_correlation_t_hopping(self,tw):
        self.corrlist = []
        self.timelist = []
        i=0
        corrt_str = "C(Delta,{tw})_{model}{a}_T={b}_timemax={c}.t".format(tw="%.0e"%tw,a=self.num
                ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model)
        with open(self.dir +corrt_str,"r",encoding='utf-8') as fp:
            while 1:
                linedata = fp.readline()
                if linedata.strip() == '':
                    break
                (time,corr) = [t(s) for t,s in zip((float,float),linedata.split())]
                self.timelist.append(time)
                self.corrlist.append(corr)
    def plot_correlation_t_hopping(self,tw):
        self.begint = tw
        self.read_correlation_t_hopping(tw)
        self.read_corrfunc_mc()
        self.__plot_corrfunc_compared_xlog2(tw)
        self.__plot_corrfunc_compared_xylog2(tw)
    def hopping_beta(self,t,tw,x):
        return (-beta.pdf(1.0-tw/t,1.0-x,x)+math.pi/np.sin(math.pi*x) )*np.sin(math.pi*x)/math.pi
    def get_hopping_beta(self,tlist,tw,x):
        outlist = []
        for t in tlist:
            outlist.append(self.hopping_beta(t,tw,x))
            # print(self.hopping_beta(t,tw,x))
        return outlist
    def bouchaud_inte_func(self,u):
        x = self.alpha
        return np.sin(math.pi * x)/math.pi * (1-u)**(x-1)*u**(-x)
    def bouchaud_func(self,t,tw):
        return integrate.quad(self.bouchaud_inte_func,(t-tw)/t,1)[0]
    def get_bouchaud_func(self,tlist,tw):
        outlist = []
        for t in tlist:
            outlist.append(self.bouchaud_func(t,tw))
            # print(self.hopping_beta(t,tw,x))
        return outlist
    def __plot_corrfunc_compared_xylog2(self,tw):
        xlabel = "time"
        ylabel = "remaining prob(time,{t})".format(t="%.0le"%tw)
        title = ylabel +"-"+ xlabel +"_compared_xylog"
        img_path = ylabel +"-"+ xlabel +"_compared_xylog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(title)
        x1 = self.ageing_func_timelist_beginfixed
        y1 = self.ageing_func_rate_beginfixed
        label1 = "MC_{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        x2 = self.timelist
        y2 = self.corrlist
        label2 = "hopping_{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        x3 = self.timelist
        y3 = self.get_bouchaud_func(self.timelist,tw)
        x4 = self.timelist
        y4 = x4 ** (-self.alpha)/(tw**(-self.alpha))
        plt.plot(x1,y1,label=label1,color="pink")
        plt.plot(x2,y2,label=label2,color="blue")
        plt.plot(x3,y3,label="Bouchaud",color="green")
        plt.plot(x4,y4,label="power func: -alpha",color="orange")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
    def __plot_corrfunc_compared_xlog2(self,tw):
        xlabel = "time"
        ylabel = "remaining prob(time,{t})".format(t="%.0le"%tw)
        title = ylabel +"-"+ xlabel +"_compared_xlog"
        img_path = ylabel +"-"+ xlabel +"_compared_xlog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.title(title)
        x1 = self.ageing_func_timelist_beginfixed
        y1 = self.ageing_func_rate_beginfixed
        label1 = "MC_{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        x2 = self.timelist
        y2 = self.corrlist
        label2 = "hopping_{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        x3 = self.timelist
        y3 = self.get_bouchaud_func(self.timelist,tw)
        # print(y3)
        x4 = self.timelist
        y4 = x4 ** (-self.alpha)/(tw**(-self.alpha))
        plt.plot(x1,y1,label=label1,color="pink")
        plt.plot(x2,y2,label=label2,color="blue")
        plt.plot(x3,y3,label="Bouchaud",color="green")
        plt.plot(x4,y4,label="power func: -alpha",color="orange")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()

    def read_startlist_hopping(self,di,time):
        self.startlist_hopping = []
        taulist_str = "startlist_hopping_n={di}_t={time}_{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,di="%.0e"%di,time="%.0e"%time)
        with open(self.dir + taulist_str,"r",encoding='utf-8') as fp:
            while 1:
                linedata = fp.readline()
                if linedata.strip() == '':
                    break
                self.startlist_hopping.append(double(linedata))
    def plot_pstart_hopping(self,di,time):
        self.read_startlist_hopping(di,time)
        lsp, dos = self.gen_doslist_log(self.startlist_hopping,int(self.bins/4))
        self.__plot_xylog_tau2("taustart_hopping_t={time}_n={di}".format(time="%.0le"%time,di="%.0e"%di)
                ,"pdf"
                ,lsp
                ,dos
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)) 
    
    def read_dtwlist_hopping(self,di,time):
        self.dtwlist_hopping = []
        taulist_str = "dtwlist_hopping_n={di}_t={time}_{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,di="%.0e"%di,time="%.0e"%time)
        with open(self.dir + taulist_str,"r",encoding='utf-8') as fp:
            while 1:
                linedata = fp.readline()
                if linedata.strip() == '':
                    break
                self.dtwlist_hopping.append(double(linedata))
    def plot_pdtw_hopping(self,di,time):
        self.read_dtwlist_hopping(di,time)
        lsp, dos = self.gen_doslist_log(self.dtwlist_hopping,int(self.bins/4))
        self.__plot_xylog_tau2("dtw_hopping_t={time}_n={di}".format(time="%.0le"%time,di="%.0e"%di)
                ,"pdf"
                ,lsp
                ,dos
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))   

    def read_taulist_hopping_analytics(self,di,time):
        self.taulist_hopping = []
        taulist_str = "taulist_hopping_analytics_n={di}_t={time}_{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,di="%.0e"%di,time="%.0e"%time)
        with open(self.dir + taulist_str,"r",encoding='utf-8') as fp:
            while 1:
                linedata = fp.readline()
                if linedata.strip() == '':
                    break
                self.taulist_hopping.append(double(linedata))
        print("finish reading taulist_hopping_analytics.\n")
        
    def plot_dostau_hopping_analytics(self,di,time):
        self.read_taulist_hopping_analytics(di,time)
        lsp, dos = self.gen_doslist(self.taulist_hopping,int(self.bins/4))
        self.__plot_xylog_tau2("tau_hopping_analytics_t={time}_n={di}".format(time="%.0le"%time,di="%.0e"%di)
                ,"pdf"
                ,lsp
                ,dos
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
    def plot_dostau_hopping_analytics_size_trct(self,di,time,bins,trct):
        self.read_taulist_hopping_analytics(di,time)
        lsp, dos = self.gen_doslist(self.taulist_hopping,int(bins))
        self.__plot_xylog_tau2("tau_hopping_analytics_t={time}_n={di}_bins={bins}_trct={trct}".format(time="%.0le"%time,di="%.0e"%di,bins=bins,trct=trct)
                ,"pdf"
                ,lsp
                ,dos
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        # fname = self.dir + "pdf_hopping_range(1e8,9e8)_{model}{a}_T={b}_timemax={c}.t".format(a=self.num
        #     ,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model)
        # self.printf(lsp, dos, fname)

    def read_taulist_hopping_analytics_tau0(self,taunum,time,bins,tau0,trct):
        self.taulist_hopping = []
        for i in range(self.start,self.end):
            taulist_str = "taulist_hopping_analytics_T={temp}_tau0={tau0}_t={time}_taunum={taunum}_trct={trct}_{d}.t".format(time="%.2le"%time
                ,taunum="%.0e"%taunum,bins="%.0e"%bins,trct="%.0e"%trct,temp="%.2lf"%self.temperature,tau0="%.2lf"%tau0,d="%06d"%i)        
            with open(self.dir + taulist_str,"r",encoding='utf-8') as fp:
                while 1:
                    linedata = fp.readline()
                    if linedata.strip() == '':
                        break
                    self.taulist_hopping.append(double(linedata))
        print("finish reading taulist_hopping_analytics_tau0.\n")
    def plot_dostau_hopping_analytics_tau0_trct(self,taunum,time,bins,tau0,trct):
        self.read_taulist_hopping_analytics_tau0(taunum,time,bins,tau0,trct)
        lsp, dos = self.gen_doslist(self.taulist_hopping,int(bins))
        self.__plot_xylog_tau2("tau_hopping_analytics_T={temp}_tau0={tau0}_t={time}_taunum={taunum}_bins={bins}_trct={trct}".format(time="%.2le"%time
            ,taunum="%.0e"%taunum,bins="%.0e"%bins,trct="%.0e"%trct,temp="%.2lf"%self.temperature,tau0="%.2lf"%tau0)
                ,"pdf"
                ,lsp
                ,dos
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        fname = self.dir + "pdf-tau_hopping_analytics_T={temp}_tau0={tau0}_t={time}_taunum={taunum}_bins={bins}_trct={trct}.t".format(time="%.2le"%time
            ,taunum="%.0e"%taunum,bins="%.0e"%bins,trct="%.0e"%trct,temp="%.2lf"%self.temperature,tau0="%.2lf"%tau0)
        self.printf(lsp, dos, fname)

    def read_taulist_hopping_analytics_smirnov(self,di,time):
        self.taulist_hopping = []
        taulist_str = "taulist_hopping_analytics_smirnov_n={di}_t={time}_{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,di="%.0e"%di,time="%.0e"%time)
        with open(self.dir + taulist_str,"r",encoding='utf-8') as fp:
            while 1:
                linedata = fp.readline()
                if linedata.strip() == '':
                    break
                self.taulist_hopping.append(double(linedata))
        print("finish reading taulist_hopping_analytics.\n")
    def plot_dostau_hopping_analytics_smirnov_size_trct(self,di,time,bins,trct,y):
        self.read_taulist_hopping_analytics_smirnov(di,time)
        lsp, dos = self.gen_doslist(self.taulist_hopping,int(bins))
        self.__plot_xylog_tau2("tau_hopping_analytics_smirnov_exact_y={y}_t={time}_n={di}_bins={bins}_trct={trct}".format(time="%.0le"%time,di="%.0e"%di,bins=bins,trct=trct,y=y)
                ,"pdf"
                ,lsp
                ,dos
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))  
        # lsp, dos = self.gen_doslist_log(self.taulist_hopping,self.bins)
        # self.__plot_xylog_tau2("tau_hopping_analytics_smirnov_log_t={time}_n={di}_bins={bins}_trct={trct}".format(time="%.0le"%time,di="%.0e"%di,bins=bins,trct=trct)
        #         ,"pdf"
        #         ,lsp
        #         ,dos
        #         ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))  



    def read_taulist_hopping_analytics_log(self,di,time):
        self.taulist_hopping = []
        taulist_str = "taulist_hopping_analytics_log_n={di}_t={time}_{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,di="%.0e"%di,time="%.0e"%time)
        with open(self.dir + taulist_str,"r",encoding='utf-8') as fp:
            while 1:
                linedata = fp.readline()
                if linedata.strip() == '':
                    break
                self.taulist_hopping.append(double(linedata))
        print("\nfinish reading taulist_hopping_analytics_log.")
    def plot_dostau_hopping_analytics_log(self,di,time):
        self.read_taulist_hopping_analytics_log(di,time)
        lsp, dos = self.gen_doslist_log(self.taulist_hopping,int(self.bins/4))
        self.__plot_xylog_tau2("tau_hopping_analytics_log_t={time}_n={di}".format(time="%.0le"%time,di="%.0e"%di)
                ,"pdf"
                ,lsp
                ,dos
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
    def plot_dostau_hopping_analytics_log_size_trct(self,di,time,size,trct):
        self.read_taulist_hopping_analytics_log(di,time)
        lsp, dos = self.gen_doslist_log(self.taulist_hopping,int(self.bins))
        self.__plot_xylog_tau2("tau_hopping_analytics_log_t={time}_n={di}_size={size}_trct={trct}".format(time="%.0le"%time,di="%.0e"%di,size=size,trct=trct)
                ,"pdf"
                ,lsp
                ,dos
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))   


    def read_taulist_hopping(self,di,time):
        self.taulist_hopping = []
        taulist_str = "taulist_hopping_n={di}_t={time}_{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,di="%.0e"%di,time="%.0e"%time)
        with open(self.dir + taulist_str,"r",encoding='utf-8') as fp:
            while 1:
                linedata = fp.readline()
                if linedata.strip() == '':
                    break
                self.taulist_hopping.append(double(linedata))
    def plot_dostau_hopping(self,time,di):
        self.read_taulist_hopping(di,time)
        lsp, dos = self.gen_doslist_log(self.taulist_hopping,int(self.bins/4))
        self.__plot_xylog_tau2("tau_hopping_t={time}_n={di}".format(time="%.0le"%time,di="%.0e"%di)
                ,"pdf"
                ,lsp
                ,dos
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))   
    # def __plot_xylog_tau2(self,xlabel,ylabel,x,y,label):
    #     title = ylabel +"-"+ xlabel +"_xylog"
    #     img_path = ylabel +"-"+ xlabel +"_xylog"
    #     img_path = self.__imgpath(img_path)
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)
    #     plt.xscale("log")
    #     plt.yscale("log")
    #     plt.title(title)
    #     plt.plot(x,y,label=label)
    #     x2 = linspace(10**(np.log10(max(x))*2/3), max(x),1000)
    #     y2 = x2**(-1.0*self.temperature*self.betac-1)*100
    #     plt.plot(x2,y2,label="y=x^(-1-alpha)) alpha={}".format(1.0*self.temperature*self.betac),color="pink")
    #     x3 = linspace(10**(np.log10(max(x))*1/6), 10**(np.log10(max(x))*2/3),1000)
    #     y3 = x3**(-1.0*self.temperature*self.betac+0.4)*0.00001
    #     plt.plot(x3,y3,label="y=x^(0.4-alpha)) alpha={}".format(1.0*self.temperature*self.betac),color="coral")
    #     plt.grid(alpha=0.3)
    #     plt.legend()
    #     plt.savefig(img_path)
    #     plt.clf()  
    def __plot_xylog_tau2(self,xlabel,ylabel,x,y,label):
        title = ylabel +"-"+ xlabel +"_xylog"
        img_path = ylabel +"-"+ xlabel +"_xylog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.xlim(100,10000)
        #plt.ylim(5e-4,1e-3)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(title)
        plt.plot(x,y,label=label)
        x2 = linspace(10**(np.log10(max(x))*2/3), max(x),1000)
        y2 = x2**(-1.0*self.temperature*self.betac-1)*100
        plt.plot(x2,y2,label="y=x^(-1-alpha)) alpha={}".format(1.0*self.temperature*self.betac),color="pink")
        x3 = linspace(10**(np.log10(max(x))*1/6), 10**(np.log10(max(x))*2/3),1000)
        y3 = x3**(-1.0*self.temperature*self.betac)*0.01
        plt.plot(x3,y3,label="y=x^(-alpha)) alpha={}".format(1.0*self.temperature*self.betac),color="coral")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()  

    def sample_from_tau_dis(self):
        xlist1 = []
        xlist2 = []
        pxlist1 = []
        self.xlist = []
        self.pxlist = []

        j=0
        dynamicstau_range_str = self.dir + "p_dynamics(tau)_range(1e8,9e8)_{model}{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model)
        with open(dynamicstau_range_str,"r",encoding='utf-8') as fp:
            while 1:
                linedata = fp.readline()
                if linedata.strip()=="":
                    break
                # (time,e,e0,dec,dec2,es) = [t(s) for t,s in zip((int,float,float,int,int,float),linedata.split())]
                # self.ebart[j]+=e/(self.end - self.start)/self.num
                # self.e0bart[j]+=e0/(self.end - self.start)/self.num
                # self.dec_t[j] = dec
                # self.esbart[j]+=es/(self.end - self.start)/self.num
                # j+=1 
                (x,px) = [t(s) for t,s in zip((float,float),linedata.split())] 
                self.xlist.append(x)
                pxlist1.append(px)
        rate = self.xlist[1]/self.xlist[0]
        xlist1 = [np.sqrt(self.xlist[i]*self.xlist[i+1]) for i in range(len(self.xlist)-1)]
        xlist1.insert(0,xlist1[0]/rate)
        xlist1.append(xlist1[-1]*rate)
        xlist2 = [xlist1[i+1]-xlist1[i] for i in range(len(xlist1)-1)]
        self.pxlist = [xlist2[i]*pxlist1[i] for i in range(len(pxlist1))]
                
        print(sum(self.pxlist))
        # self.__plot_tau_t_from_eqdis(1e2)
        # self.__plot_tau_t_from_eqdis(1e4)
        # self.__plot_tau_t_from_eqdis(1e6)
        self.__plot_tau_t_from_eqdis(2e8)
    def __plot_tau_t_from_eqdis(self,tw):
        slist = []
        slist2 = []
        samplenum = 100000
        for i in range(samplenum):
            slist=rv_discrete(values=(self.xlist,self.pxlist)).rvs(size=10000)
            slist2.append(self.get_tau(slist,tw))

        self.lsp_slist2, self.dos_slist2 = self.gen_doslist_log(slist2,int(self.bins/4))
        self.__plot_xylog_tau2("tau_t={}".format("%.0le"%tw)
                ,"pdf"
                ,self.lsp_slist2
                ,self.dos_slist2
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))    
    def get_tau(self,list,tw):
        j=1
        sum=list[0]
        while 1:
            if sum > tw:
                break
            sum+=list[j]
            j+=1
        return list[j-1]
    
    def plot_tau_t_2point_hopping(self):
        samplenum = 5000
        tsize = 50
        self.nlist = [i+1 for i in range(tsize)]
        # print(self.__tau_t_2point_hopping(2,samplenum,tsize)[0])
        ptau1list = [self.__tau_t_2point_hopping_power(i,samplenum,tsize)[0] for i in range(tsize)]
        ptau2list = [self.__tau_t_2point_hopping_power(i,samplenum,tsize)[1] for i in range(tsize)]
        self.__plot("tau1_2points_hopping"
                ,"pdf"
                ,self.nlist
                ,ptau1list
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)) 
        self.__plot("tau2_2points_hopping"
                ,"pdf"
                ,self.nlist
                ,ptau2list
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)) 
    def __tau_t_2point_hopping_equal(self,tw,samplenum,tsize):
        xlist = [1,3]
        pxlist = [0.5,0.5]
        slist = []
        slist2 = []
        for i in range(samplenum):
            slist=rv_discrete(values=(xlist,pxlist)).rvs(size=tsize+1)
            # print(slist)
            slist2.append(self.get_tau(slist,tw))
            # print(slist,self.get_tau(slist,tw))
        ptau1 = slist2.count(xlist[0])/samplenum
        ptau2 = slist2.count(xlist[1])/samplenum
        return [ptau1,ptau2]
    def __tau_t_2point_hopping_power(self,tw,samplenum,tsize):
        alpha = 0.88305
        xlist = [1,10]
        pxlist = [xlist[0]**(-alpha)/(xlist[0]**(-alpha)+xlist[1]**(-alpha)),xlist[1]**(-alpha)/(xlist[0]**(-alpha)+xlist[1]**(-alpha))]
        slist = []
        slist2 = []
        for i in range(samplenum):
            slist=rv_discrete(values=(xlist,pxlist)).rvs(size=tsize+1)
            # print(slist)
            slist2.append(self.get_tau(slist,tw))
            # print(slist,self.get_tau(slist,tw))
        ptau1 = slist2.count(xlist[0])/samplenum
        ptau2 = slist2.count(xlist[1])/samplenum
        return [ptau1,ptau2]

    def read_e_t_sizes(self):
        self.timelist3 = [i*(i+1)/2 for i in range(1,self.__time_to_int2(self.timemax))]
        self.ebart_16 = [0 for i in range(1,self.__time_to_int2(self.timemax))]
        self.ebart_12 = [0 for i in range(1,self.__time_to_int2(self.timemax))]
        self.ebart_8 = [0 for i in range(1,self.__time_to_int2(self.timemax))]
        dir_str = self.dir
        self.correlation_t = [0 for i in range(1,self.__time_to_int2(self.timemax))]
        for i in range(self.start+200, self.end+200):
            Et_str = "E-t_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            j=0
            with open(dir_str + Et_str,"r",encoding='utf-8') as fp:
                while 1:
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    # (time,e,e0,dec,dec2,es) = [t(s) for t,s in zip((int,float,float,int,int,float),linedata.split())]
                    # self.ebart[j]+=e/(self.end - self.start)/self.num
                    # self.e0bart[j]+=e0/(self.end - self.start)/self.num
                    # self.dec_t[j] = dec
                    # self.esbart[j]+=es/(self.end - self.start)/self.num
                    # j+=1 
                    (time,e) = [t(s) for t,s in zip((int,float),linedata.split())] 
                    self.ebart_16[j]+=e/(self.end - self.start)/self.num
                    # self.e0bart[j]+=e0/(self.end - self.start)/self.num
                    # self.dec_t[j] = dec
                    # self.esbart[j]+=es/(self.end - self.start)/self.num
                    j+=1
        self.num = 12
        for i in range(self.start, self.end):
            Et_str = "E-t_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            j=0
            with open(self.dir + Et_str,"r",encoding='utf-8') as fp:
                while 1:
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    # (time,e,e0,dec,dec2,es) = [t(s) for t,s in zip((int,float,float,int,int,float),linedata.split())]
                    # self.ebart[j]+=e/(self.end - self.start)/self.num
                    # self.e0bart[j]+=e0/(self.end - self.start)/self.num
                    # self.dec_t[j] = dec
                    # self.esbart[j]+=es/(self.end - self.start)/self.num
                    # j+=1 
                    (time,e) = [t(s) for t,s in zip((int,float),linedata.split())] 
                    self.ebart_12[j]+=e/(self.end - self.start)/self.num
                    # self.e0bart[j]+=e0/(self.end - self.start)/self.num
                    # self.dec_t[j] = dec
                    # self.esbart[j]+=es/(self.end - self.start)/self.num
                    j+=1
        self.num = 8
        for i in range(self.start+100, self.end+100):
            Et_str = "E-t_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            j=0
            with open(self.dir + Et_str,"r",encoding='utf-8') as fp:
                while 1:
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    # (time,e,e0,dec,dec2,es) = [t(s) for t,s in zip((int,float,float,int,int,float),linedata.split())]
                    # self.ebart[j]+=e/(self.end - self.start)/self.num
                    # self.e0bart[j]+=e0/(self.end - self.start)/self.num
                    # self.dec_t[j] = dec
                    # self.esbart[j]+=es/(self.end - self.start)/self.num
                    # j+=1 
                    (time,e) = [t(s) for t,s in zip((int,float),linedata.split())] 
                    self.ebart_8[j]+=e/(self.end - self.start)/self.num
                    # self.e0bart[j]+=e0/(self.end - self.start)/self.num
                    # self.dec_t[j] = dec
                    # self.esbart[j]+=es/(self.end - self.start)/self.num
                    j+=1 
    def plot_xlog_e_t_compared3(self):
        self.read_e_t_sizes()
        self.__plot_xlog_e_t_compared3()
    def __plot_xlog_e_t_compared3(self):
        xlabel = "time_xlog_compared_N=8-16"
        ylabel = "Energy"
        title = ylabel +"-"+ xlabel+"_xlog"
        img_path = ylabel +"-"+ xlabel +"_xlog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.title(title)
        x = self.timelist3
        y1 = self.ebart_8
        label1 = "N=8"
        y2 = self.ebart_12
        label2 = "N=12"
        y3 = self.ebart_16
        label3 = "N=16"
        y4 = [-1.0*np.sqrt(2*np.log(2)) for i in range(len(x))]
        label4 = "eth"
        plt.plot(x,y1,label=label1,color="blue")
        plt.plot(x,y2,label=label2,color="green")
        plt.plot(x,y3,label=label3,color="yellow")
        plt.plot(x,y4,label=label4,color="black")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
    def read_e_t_mc(self):
        self.timelistsii1 = []
        self.eslistsii1 = []
        self.e0listsii1 = []
        self.eblistsii1 = []
        self.eslistsii1_range = []
        self.e0listsii1_range = []
        self.eblistsii1_range = []

        self.timelistsii2 = [k*(k+1) for k in range(self.__time_to_int2(1e9))]
        self.esbarlist = [0 for k in range(self.__time_to_int2(1e9))]
        self.e0barlist = [0 for k in range(self.__time_to_int2(1e9))]
        self.ebbarlist = [0 for k in range(self.__time_to_int2(1e9))]

        for i in range(self.start,self.end):
            self.taulist = []
            self.eb1list = []
            time = 0
            jlast_last = 0
            act_str = "activation-t_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            with open(self.dir + act_str,'r') as fp:
                #fp.readline()
                while 1:
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    else:
                        # if(len(linedata.split())<2):
                        #     print(i)
                        timelast = time
                        
                        try:
                            (time,e0,es,e1,tip0,saddle,tip1,effe0,effes,effe1,effs,tau0,taus, tau) = [t(s) for t,s in zip((int,float,float,float,int,int,int,float,float,float,int,int,int,int),linedata.split())]
                        except:
                            print(act_str)
                        jlast = self.__time_to_int2(timelast)
                        jnow = self.__time_to_int2(time)
                        for j in range(jlast,jnow):
                            self.timelistsii1.append(j*(j+1))
                            self.eslistsii1.append(es)
                            self.eblistsii1.append(es-e0)
                            self.e0listsii1.append(e0)
                        # if time > 1e8 and time < 9e8:
                        #     for j in range(jlast,jnow):
                                
                        #         self.eslistsii1_range.append(es/self.num)
                        #         self.eblistsii1_range.append((es-e0)/self.num)
                        #         self.e0listsii1_range.append(e0/self.num)

                        for j in range(jlast,jnow): 
                            if j < self.__time_to_int2(1e9):
                                self.e0barlist[j] += e0/self.num/(self.end - self.start)
                                self.esbarlist[j] += es/self.num/(self.end - self.start)
                                self.ebbarlist[j] += -1.0*(es-e0)/self.num/(self.end - self.start)
                        lenen = len(self.e0barlist)
                        self.timelistsii2 = self.timelistsii2[:lenen]
        print("finish: read and construct data from dynamics.")
    def plot_et_dos(self):
        self.read_e_t_mc()
        lspes, doses = self.gen_doslist(self.eslistsii1_range,self.bins)
        lspeb, doseb = self.gen_doslist(self.eblistsii1_range,self.bins)
        lspe0, dose0 = self.gen_doslist(self.e0listsii1_range,self.bins)
        # self.__plot_xlog_e_t_compared()
        self.__plot("prob"
                ,"es_time_i(i+1)"
                ,lspes
                ,doses
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("prob"
                ,"eb_time_i(i+1)"
                ,lspeb
                ,doseb
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("prob"
                ,"e0_time_i(i+1)"
                ,lspe0
                ,dose0
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        # prob, x, y, m = self.__hist2d_xlog("time"
        #     ,"es"
        #     ,"prob density_xlog"
        #     ,self.timelistsii1
        #     ,self.eslistsii1
        #     ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        #     ,self.__imgpath("time-es_prob_xylog")
        #     ,self.bins + 1)
        # prob, x, y, m = self.__hist2d_xlog("time"
        #     ,"eb"
        #     ,"prob density_xlog"
        #     ,self.timelistsii1
        #     ,self.eblistsii1
        #     ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        #     ,self.__imgpath("time-eb_prob_xylog")
        #     ,self.bins + 1)
        # prob, x, y, m = self.__hist2d_xlog("time"
        #     ,"e0"
        #     ,"prob density_xlog"
        #     ,self.timelistsii1
        #     ,self.e0listsii1
        #     ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        #     ,self.__imgpath("time-e0_prob_xylog")
        #     ,self.bins + 1)
    def __plot_xlog_e_t_compared(self):
        xlabel = "time_xlog_compared"
        ylabel = "Energy"
        title = ylabel +"-"+ xlabel+"_xlog"
        img_path = ylabel +"-"+ xlabel +"_xlog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.title(title)
        x = self.timelistsii2
        y1 = self.e0barlist
        label1 = "e0bar"
        y2 = self.esbarlist
        label2 = "esbar"
        y3 = self.ebbarlist
        label3 = "ebbar"
        plt.plot(x,y1,label=label1,color="blue")
        plt.plot(x,y2,label=label2,color="green")
        plt.plot(x,y3,label=label3,color="yellow")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()

        

    
    def plot_gentau_corr_tree(self):
        self.read_gentau_corr_tree()
        self.__plot("Delta_gen_corr_tree"
                ,"remaining prob(Delta,{t})".format(t="%.0le"%self.begint)
                ,self.ageing_func_timelist_gen_corr_tree
                ,self.ageing_func_rate_gen_corr_tree
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xlog("Delta_gen_corr_tree"
                ,"remaining prob(Delta,{t})".format(t="%.0le"%self.begint)
                ,self.ageing_func_timelist_gen_corr_tree
                ,self.ageing_func_rate_gen_corr_tree
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xylog("Delta_gen_corr_tree"
                ,"remaining prob(Delta,{t})".format(t="%.0le"%self.begint)
                ,self.ageing_func_timelist_gen_corr_tree
                ,self.ageing_func_rate_gen_corr_tree
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
    def read_gentau_corr_tree(self):
        j=0
        dir_str = self.dir
        ageing_func_hit_gen = np.array([0 for i in range(self.__time_to_int(self.timemax - self.begint))])
        ageing_func_total_gen = np.array([0 for i in range(self.__time_to_int(self.timemax - self.begint))])
        for i in range(self.start, self.end):
            j=0
            tau_str = "C(Delta,1e4)_corr_tree_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            with open(dir_str + tau_str,"r",encoding='utf-8') as fp:
                while 1:
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    (delta,Enum) = [t(s) for t,s in zip((int,int),linedata.split())]
                    ageing_func_hit_gen[j] += Enum
                    j+=1
            ageing_func_total_gen += 1
                   
        self.ageing_func_rate_gen_corr_tree = [ageing_func_hit_gen[i]/ageing_func_total_gen[i] for i in range(self.__time_to_int(self.timemax/2)) if ageing_func_total_gen[i]!=0]        
        self.ageing_func_timelist_gen_corr_tree = [i*(i+1)+self.begint for i in range(self.__time_to_int(self.timemax/2)) if ageing_func_total_gen[i]!=0]

                


    def read_data_range(self):
        dir_str = self.dir
        range_str = "range_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
        for i in range(self.start, self.end):
            with open(dir_str + range_str,"r",encoding='utf-8') as fp:
                linedata = fp.readline()
                (e0,es,effe0,effes,tau0,taus, tau) = [t(s) for t,s in zip((float,float,float,float,int,int,int),linedata.split())]
                self.mine0 = e0
                self.mines = es
                self.mineffe0 = effe0
                self.mineffes = effes
                self.mintau0 = tau0
                self.mintaus = taus
                self.mintau = tau
                linedata = fp.readline()
                (e0,es,effe0,effes,tau0,taus, tau) = [t(s) for t,s in zip((float,float,float,float,int,int,int),linedata.split())]
                self.maxe0 = e0
                self.maxes = es
                self.maxeffe0 = effe0
                self.maxeffes = effes
                self.maxtau0 = tau0
                self.maxtaus = taus
                self.maxtau = tau

    def read_data_lowest_barriers(self):

        dir_str = self.dir
        for i in range(self.start, self.end):
            lowest_barriers_str = "lowest_barriers_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            with open(dir_str + lowest_barriers_str,"r",encoding='utf-8') as fp:
                while 1:
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    (eb,effeb,ineren,entropy) = [t(s) for t,s in zip((float,float,float,float),linedata.split())]
                    self.treeleb.append(eb)
                    self.treeleffeb.append(effeb)
                    self.treeentropy.append(entropy)
    def __plot_lowest_barriers(self):
        self.read_data_lowest_barriers()
        lsp_treeleb, dos_treeleb = self.gen_doslist(self.treeleb, self.bins)
        self.lsp_treeleb, self.dos_treeleb =  lsp_treeleb, dos_treeleb
        self.treeleb_range = [ele for ele in self.treeleb if np.exp(ele/self.temperature)>self.taumin and np.exp(ele/self.temperature)<=self.taumax]
        lsp_treeleb_range, dos_treeleb_range = self.gen_doslist(self.treeleb_range, self.bins)
        self.lsp_treeleb, self.dos_treeleb =  lsp_treeleb, dos_treeleb
        del self.treeleb
        del self.treeleb_range
        lsp_treeleffleb, dos_treeleffleb = self.gen_doslist(self.treeleffeb, self.bins)
        del self.treeleffeb
        bol_dos_treeleb, bol_dos_treeleb2 = self.Bolzmann_mod(lsp_treeleb, dos_treeleb)
        bol_dos_treeleffleb, bol_dos_treeleffleb2 = self.Bolzmann_mod(lsp_treeleffleb, dos_treeleffleb)
        lsp_treeentropy, dos_treeentropy = self.gen_doslist(self.treeentropy,self.bins)        

        self.lsp_treetau, self.dos_treetau = self.treeArrhenius(lsp_treeleb, dos_treeleb)
        self.lsp_treetau_range, self.dos_treetau_range = self.treeArrhenius(lsp_treeleb_range, dos_treeleb_range)
        self.lsp_treeefftau, self.dos_treeefftau = self.treeArrhenius(lsp_treeleffleb, dos_treeleffleb)#lowest eff barrier

        
        
        
        self.__plot("Tree_lowest_barriers"
        ,"pdf"
        ,lsp_treeleb
        ,dos_treeleb
        ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("Tree_lowest_effbarriers"
        ,"pdf"
        ,lsp_treeleffleb
        ,dos_treeleffleb
        ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        #ylog:
        self.__plot_ylog_lb("Tree_lowest_barriers"
        ,"pdf"
        ,lsp_treeleb
        ,dos_treeleb
        ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_ylog_lb("Tree_lowest_effbarriers"
        ,"pdf"
        ,lsp_treeleffleb
        ,dos_treeleffleb
        ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

        self.__plot_ylog_lb("Tree_lowest_barriers_Tmod"
        ,"pdf"
        ,lsp_treeleb
        ,bol_dos_treeleb
        ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_ylog_lb("Tree_lowest_effbarriers_Tmod"
        ,"pdf"
        ,lsp_treeleffleb
        ,bol_dos_treeleffleb
        ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

        self.__plot_ylog_lb("Tree_lowest_barriers_Tcmod"
        ,"pdf"
        ,lsp_treeleb
        ,bol_dos_treeleb2
        ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_ylog_lb("Tree_lowest_effbarriers_Tcmod"
        ,"pdf"
        ,lsp_treeleffleb
        ,bol_dos_treeleffleb2
        ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

        self.__plot("Tree_entropy"
        ,"pdf"
        ,lsp_treeentropy
        ,dos_treeentropy
        ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_ylog_entropy("Tree_entropy"
        ,"pdf"
        ,lsp_treeentropy
        ,dos_treeentropy
        ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

        self.__plot_xylog_tau("treetau"
            ,"p(tau)"
            ,self.lsp_treetau
            ,self.dos_treetau
            ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xylog_tau("tree eff tau"
            ,"p(eff tau)"
            ,self.lsp_treeefftau
            ,self.dos_treeefftau
            ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

        print("finish: plot_lowest_barriers.")

    def read_data_lowest_barriers_muchdata(self):
        dir_str = self.treedir #
        for i in range(self.start, self.endtree):
            lowest_barriers_str = "lowest_barriers_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            with open(dir_str + lowest_barriers_str,"r",encoding='utf-8') as fp:
                while 1:
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    (eb,effeb,ineren,entropy) = [t(s) for t,s in zip((float,float,float,float),linedata.split())]
                    self.treeleb.append(eb)
                    self.treeleffeb.append(effeb)
                    self.treeentropy.append(entropy)
    def plot_lowest_barriers(self):
        mkdir(self.dir + "plot/")
        self.read_data_lowest_barriers_muchdata()
        lsp_treeleb, dos_treeleb = self.gen_doslist(self.treeleb, self.bins)
        self.treeleb_range = [ele for ele in self.treeleb if np.exp(ele/self.temperature)>self.taumin and np.exp(ele/self.temperature)<=self.taumax]
        lsp_treeleb_range, dos_treeleb_range = self.gen_doslist(self.treeleb_range, self.bins)
        self.lsp_treeleb, self.dos_treeleb =  lsp_treeleb, dos_treeleb
        del self.treeleb
        del self.treeleb_range
        lsp_treeleffleb, dos_treeleffleb = self.gen_doslist(self.treeleffeb, self.bins)
        del self.treeleffeb
        lsp_treeentropy, dos_treeentropy = self.gen_doslist(self.treeentropy,self.bins)

        self.lsp_treetau, self.dos_treetau = self.treeArrhenius(lsp_treeleb, dos_treeleb)
        self.lsp_treetau_range, self.dos_treetau_range = self.treeArrhenius(lsp_treeleb_range, dos_treeleb_range)
        self.lsp_treeefftau, self.dos_treeefftau = self.treeArrhenius(lsp_treeleffleb, dos_treeleffleb)#lowest eff barrier
        treetau_range_str = self.dir + "p_tree(tau)_range_{model}{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model)
        self.printf(self.lsp_treetau_range, self.dos_treetau_range, treetau_range_str)
        treetau_str = self.dir + "p_tree(tau)_{model}{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model)
        self.printf(self.lsp_treetau, self.dos_treetau, treetau_str)

        # bol_dos_treeleb, bol_dos_treeleb2 = self.Bolzmann_mod(lsp_treeleb, dos_treeleb)
        # bol_dos_treeleffleb, bol_dos_treeleffleb2 = self.Bolzmann_mod(lsp_treeleffleb, dos_treeleffleb)
        self.__plot("Tree_lowest_barriers"
        ,"pdf"
        ,lsp_treeleb
        ,dos_treeleb
        ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("Tree_lowest_effbarriers"
        ,"pdf"
        ,lsp_treeleffleb
        ,dos_treeleffleb
        ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        #ylog:
        self.__plot_ylog_lb("Tree_lowest_barriers"
        ,"pdf"
        ,lsp_treeleb
        ,dos_treeleb
        ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_ylog_lb("Tree_lowest_effbarriers"
        ,"pdf"
        ,lsp_treeleffleb
        ,dos_treeleffleb
        ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

        self.__plot("Tree_entropy"
        ,"pdf"
        ,lsp_treeentropy
        ,dos_treeentropy
        ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_ylog_entropy("Tree_entropy"
        ,"pdf"
        ,lsp_treeentropy
        ,dos_treeentropy
        ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

        self.__plot_xylog_tau("treetau"
            ,"p(tau)"
            ,self.lsp_treetau
            ,self.dos_treetau
            ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xylog_tau("tree eff tau"
            ,"p(eff tau)"
            ,self.lsp_treeefftau
            ,self.dos_treeefftau
            ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

        print("finish: plot_lowest_barriers.")
    def treeArrhenius(self,lsp,dos):
        lspout = [np.exp(ele/self.temperature) for ele in lsp]
        dosout = [dos[i]*self.temperature/lspout[i] for i in range(len(dos))]
        return lspout, dosout
    def printf(self,lsp,dos,path):
        with open(path,'w+',encoding='utf-8') as fp:
            for i in range(len(lsp)):
                #print(len(lsp))
                fp.write("%le"%lsp[i]+" "+"%le"%dos[i]+"\n")


    def read_data_E_t(self):
        self.dose0_105 = []
        self.dose0_10011 = []
        self.dose0_1000405 = []
        self.dose0_100005153 = []
        self.dose0_500022876 = []
        dir_str = self.dir
        self.correlation_t = [0 for i in range(1,self.__time_to_int2(self.timemax))]
        for i in range(self.start, self.end):
            Et_str = "E-t_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            j=0
            with open(dir_str + Et_str,"r",encoding='utf-8') as fp:
                while 1:
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    # (time,e,e0,dec,dec2,es) = [t(s) for t,s in zip((int,float,float,int,int,float),linedata.split())]
                    # self.ebart[j]+=e/(self.end - self.start)/self.num
                    # self.e0bart[j]+=e0/(self.end - self.start)/self.num
                    # self.dec_t[j] = dec
                    # self.esbart[j]+=es/(self.end - self.start)/self.num
                    # j+=1 
                    (time,e,e0,dec,dec2) = [t(s) for t,s in zip((int,float,float,int,int),linedata.split())] 
                    # self.ebart[j]+=e/(self.end - self.start)/self.num
                    # self.e0bart[j]+=e0/(self.end - self.start)/self.num
                    # self.dec_t[j] = dec
                    if time == 105:
                        self.dose0_105.append(e0/self.num)
                    if time == 10011:
                        self.dose0_10011.append(e0/self.num)
                    if time == 1000405:
                        self.dose0_1000405.append(e0/self.num)
                    if time == 100005153:
                        self.dose0_100005153.append(e0/self.num)
                    if time == 500022876:
                        self.dose0_500022876.append(e0/self.num)
                    # self.esbart[j]+=es/(self.end - self.start)/self.num
                    j+=1 

            # self.cal_correlation_t(self.dec_t)
    def plot_dose0_at_specific_time(self):
        self.read_data_E_t()
        self.read_data_tree() 
        self.lsp1,self.dos1 = self.gen_doslist(self.dose0_105,int(self.bins/10))
        self.lsp2,self.dos2 = self.gen_doslist(self.dose0_10011,int(self.bins/10))
        self.lsp3,self.dos3 = self.gen_doslist(self.dose0_1000405,int(self.bins/10))
        self.lsp4,self.dos4 = self.gen_doslist(self.dose0_100005153,int(self.bins/10))
        self.lsp5,self.dos5 = self.gen_doslist(self.dose0_500022876,int(self.bins/10))
        self.lsp_teslist, self.dos_teslist = self.gen_doslist(self.treeeslists,self.bins)
        del self.treeeslists
        self.lsp_te0list, self.dos_te0list = self.gen_doslist(self.treee0lists,self.bins)
        del self.treee0lists
        self.lsp_teblist, self.dos_teblist = self.gen_doslist(self.treeeblists,self.bins)
        del self.treeeblists

        #eb and es dos time vs tree
        # self.lsp_te0list, self.dos_te0list = self.gen_doslist(self.treee0lists,self.bins)
        self.__plot_dosE0_at_time_vs_tree()
        self.__plot("Tree E0"
                ,"pdf"
                ,self.lsp_te0list
                ,self.dos_te0list
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

        self.__plot("e0_t=105"
                ,"dos"
                ,self.lsp1
                ,self.dos1
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("e0_t=10011"
                ,"dos"
                ,self.lsp2
                ,self.dos2
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("e0_t=1000405"
                ,"dos"
                ,self.lsp3
                ,self.dos3
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("e0_t=100005153"
                ,"dos"
                ,self.lsp4
                ,self.dos4
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("e0_t=500022876"
                ,"dos"
                ,self.lsp5
                ,self.dos5
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
    def __plot_dosE0_at_time_vs_tree(self):
        xlabel = "e0_different_time_or_tree"
        ylabel = "dos"
        title = ylabel +"-"+ xlabel
        img_path = ylabel +"-"+ xlabel
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.plot(self.lsp1,self.dos1,label="t=105",color="green")
        plt.plot(self.lsp2,self.dos2,label="t=10011",color="blue")
        plt.plot(self.lsp3,self.dos3,label="t=1000405",color="pink")
        plt.plot(self.lsp4,self.dos4,label="t=100005153",color="coral")
        plt.plot(self.lsp5,self.dos5,label="t=500022876",color="black")
        plt.plot(self.lsp_te0list,self.dos_te0list,label="tree",color="grey")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()

    def cal_correlation_t(self, dec_t):
        cor_t_now = [self.correlation(dec_t[0],dec_t[i]) for i in range(len(dec_t))]
        self.correlation_t = [self.correlation_t[i]+cor_t_now[i]/self.end for i in range(len(dec_t))]
        del cor_t_now
    def correlation(self,dec1,dec2):
        same = [1.0 for i in range(self.num) if ((dec1>>i)%2)==((dec2>>i)%2)]
        return sum(same)/self.num             
    def __plot_e_t_xlog(self,xlabel,ylabel,label):
        title = ylabel +"-"+ xlabel
        img_path = ylabel +"-"+ xlabel
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xscale("log")
        Min = min(self.timelist)
        Max = max(self.timelist)
        theo_x = np.logspace(1/3*np.log10(Max), 2/3*np.log10(Max))
        theo_y = -1.0*self.temperature*np.log(theo_x)/self.num - 0.5
        plt.plot(self.timelist,self.ebart,label=label+" E from C",color="blue")
        plt.plot(self.timelist,self.e0bart,label=label+" Etip0 from C",color="green")
        # plt.plot(self.e1timelist,self.e0barlist,label=label+" Etip0 python",color="black")
        plt.plot(theo_x, theo_y, label=label+" E=-T*log(t)/N",color="pink")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
    def __plot_E_t(self):
        self.read_data_E_t()
        self.__plot_e_t_xlog("time"
            ,"E"
            ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

        self.__plot("time"
                ,"correlation(0,t)"
                ,self.timelist
                ,self.correlation_t
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xlog("time"
                ,"correlation(0,t)"
                ,self.timelist
                ,self.correlation_t
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xylog("time"
                ,"correlation(0,t)"
                ,self.timelist
                ,self.correlation_t
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

        self.__plot("time"
                ,"Es"
                ,self.timelist
                ,self.esbart
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xlog("time"
                ,"Es"
                ,self.timelist
                ,self.esbart
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        print("finish: plot E-t.")
    def read_data_E_t2(self):
        dir_str = self.dir
        for i in range(self.start, self.end):
            Et_str = "E-t_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
            j=0
            with open(dir_str + Et_str,"r",encoding='utf-8') as fp:
                while 1:
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    (time,e) = [t(s) for t,s in zip((int,float),linedata.split())]
                    self.ebart[j]+=e/(self.end - self.start)/self.num
                    j+=1
    def __plot_e_t_xlog2(self,xlabel,ylabel,label):
        title = ylabel +"-"+ xlabel
        img_path = ylabel +"-"+ xlabel
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xscale("log")
        plt.xlim(100,1e8)
        plt.ylim(-0.8,-0.4)
        Min = min(self.timelist)
        Max = max(self.timelist)
        theo_x = np.logspace(1/3*np.log10(100), 2/3*np.log10(1e8))
        theo_y = -1.0*self.temperature*np.log(theo_x)/self.num - 0.5
        # xx = self.timelist
        # yy = self.ebart
        # fit_x = [ele for ele in xx if ele>=100]
        # fit_y = [yy[i] for i in range(len(yy)) if xx[i]>=100]
        # para, pcov = curve_fit(self.fit_func2,np.array(fit_x),np.array(fit_y),maxfev = 10000)
        # fit_y = self.fit_func2(self.timelist, para[0], para[1], para[2])
        # plt.plot(self.timelist,fit_y,label=label+" {a}+{b}*x^{c}".format(a="%.2f"%para[0],b="%.2f"%para[1],c="%.2f"%para[2]),color="black")
        plt.plot(self.timelist,self.ebart,label=label+" E from C",color="blue")
        plt.plot(theo_x, theo_y, label=label+" E=-T*log(t)/N",color="pink")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
    def plot_E_t2(self):
        mkdir(self.dir + "plot/")
        self.read_data_E_t2()
        self.__plot_e_t_xlog2("time"
        ,"E"
        ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
    def fit_func2(self,x,a,b,c):
        return a + b*x**c

    def read_data_EEs_t_1sample(self,i):
        timelist = []
        elist = []
        dir_str = self.dir
        EEst_str = "EEs-t_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
        with open(dir_str + EEst_str,"r",encoding='utf-8') as fp:
            while 1:
                linedata = fp.readline()
                if linedata.strip()=="":
                    break
                (time,e) = [t(s) for t,s in zip((int,float),linedata.split())]
                timelist.append(time)
                elist.append(e/self.num)
        return timelist, elist
    def __plot_ees_t(self,xlabel,ylabel,x,y,label,i):
        title = ylabel +"-"+ xlabel
        img_path = ylabel +"-"+ xlabel
        img_path = self.__imgpath2(img_path,i)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.plot(x,y,label=label)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
    def __plot_ees_t_xlog(self,xlabel,ylabel,x,y,label,i):
        title = ylabel +"-"+ xlabel +"_xlog"
        img_path = ylabel +"-"+ xlabel +"_xlog"
        img_path = self.__imgpath2(img_path,i)
        plt.xscale("log")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.plot(x,y,label=label)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
    def __plot_EEs_t(self):
        mkdir(self.dir + "plot/EEs-t/")
        for i in range(self.start, self.end):
            timelist, elist = self.read_data_EEs_t_1sample(i)
            self.__plot_ees_t("time", "EEs_1sample", timelist, elist
                        , "{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model),i)
            self.__plot_ees_t_xlog("time", "EEs_1sample", timelist, elist
                        , "{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model),i)
            del timelist
            del elist
        print("finish: plot_EEs_t.")

    def read_data_EEs_t_1sample(self,i):
        timelist = []
        elist = []
        dir_str = self.dir
        EEst_str = "EEs-t_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
        with open(dir_str + EEst_str,"r",encoding='utf-8') as fp:
            while 1:
                linedata = fp.readline()
                if linedata.strip()=="":
                    break
                (time,e) = [t(s) for t,s in zip((int,float),linedata.split())]
                timelist.append(time)
                elist.append(e/self.num)
        return timelist, elist
    def __plot_1sample(self,xlabel,ylabel,x,y,label,i,dirname):
        title = ylabel +"-"+ xlabel
        img_path = ylabel +"-"+ xlabel
        img_path = self.__imgpath3(img_path,dirname,i)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.plot(x,y,label=label)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
    def __plot_1sample_xlog(self,xlabel,ylabel,x,y,label,i,dirname):
        title = ylabel +"-"+ xlabel +"_xlog"
        img_path = ylabel +"-"+ xlabel +"_xlog"
        img_path = self.__imgpath3(img_path,dirname,i)
        plt.xscale("log")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.plot(x,y,label=label)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
    def __plot_1sample_xylog(self,xlabel,ylabel,x,y,label,i,dirname):
        title = ylabel +"-"+ xlabel +"_xylog"
        img_path = ylabel +"-"+ xlabel +"_xylog"
        img_path = self.__imgpath3(img_path,dirname,i)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.plot(x,y,label=label)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
    def __plot_EEs_t(self):
        mkdir(self.dir + "plot/EEs-t/")
        for i in range(self.start, self.end):
            timelist, elist = self.read_data_EEs_t_1sample(i)
            self.__plot_ees_t("time", "EEs_1sample", timelist, elist
                        , "{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model),i)
            self.__plot_ees_t_xlog("time", "EEs_1sample", timelist, elist
                        , "{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model),i)
            del timelist
            del elist
        print("finish: plot_EEs_t.")
    
                    
    def read_data_tree(self):
        self.treeeblists = []
        for i in range(self.start,self.end):
            # activation_pairs_str = "T={cc}/activation_pairs_rem{aa}_{bb}.t".format(aa=self.num,bb="%06d"%i,cc="%.2f"%self.temperature)
            # with open(activation_pairs_str,"r") as fp:
            #     while 1:
            #         linedata = fp.readline()
            #         if linedata.strip()=="":
            #             break
            #         else:
            #             #(de) = [t(s) for t,s in zip((float),linedata.split())]
            #             #(time, e0, es, e1, tip0, saddle, tip1, time_s, tau_tip, tau_saddle) = [t(s) for t,s in zip((int,float,float,float,int,int,int,int,int,int),linedata.split())]
            #             self.treeblists.append(float(linedata.strip()))

            bar_str = "data/N={dd}_T={cc}_{ee}/{model}{aa}_{bb}.bar".format(aa=self.num,bb="%06d"%i,cc="%.2f"%self.temperature,dd=self.num,ee="%.0le"%self.timemax,model=self.model)
            with open(bar_str,"r") as fp:
                fp.readline()
                linedata = str(fp.readline())
                (index, tipconf, tipen, link, sen) = [t(s) for t,s in zip((int,str,float,int,float),linedata.split())]
                self.treee0lists.append(tipen/self.num)

                while 1:
                    linedata = fp.readline()
                    if linedata.strip()=="":
                        break
                    else:
                        (index, tipconf, tipen, link, diffen) = [t(s) for t,s in zip((int,str,float,int,float),linedata.split())]
                        self.treee0lists.append(tipen/self.num)
                        self.treeeslists.append(diffen/self.num+tipen/self.num)
                        self.treeeblists.append(diffen/self.num)
    def __plot_treeE_dos(self):
        self.read_data_tree() 
        lsp_teslist, dos_teslist = self.gen_doslist(self.treeeslists,self.bins)
        del self.treeeslists
        lsp_te0list, dos_te0list = self.gen_doslist(self.treee0lists,self.bins)
        del self.treee0lists
        lsp_teblist, dos_teblist = self.gen_doslist(self.treeeblists,self.bins)
        del self.treeeblists
        bol_dos_te0list, bol_dos_te0list2 = self.Bolzmann_mod(lsp_te0list, dos_te0list)
        bol_dos_teslist, bol_dos_teslist2 = self.Bolzmann_mod(lsp_teslist, dos_teslist)
        self.__plot("Tree Es"
                ,"pdf"
                ,lsp_teslist
                ,dos_teslist
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("Tree E0"
                ,"pdf"
                ,lsp_te0list
                ,dos_te0list
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("Tree Es_Tmod"
                ,"pdf"
                ,lsp_teslist
                ,bol_dos_teslist
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("Tree E0_Tmod"
                ,"pdf"
                ,lsp_te0list
                ,bol_dos_te0list
                ,"{model}{a}_T={b}".format(a=self.num,b="%.DOS2f"%self.temperature,model=self.model))
        self.__plot("Tree Es_Tcmod"
                ,"pdf"
                ,lsp_teslist
                ,bol_dos_teslist2
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("Tree E0_Tcmod"
                ,"pdf"
                ,lsp_te0list
                ,bol_dos_te0list2
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        print("finish: plot DOS-E in tree. :)")


    # def check_tau0(self):

    # def __plot_etip(self):
    #     self.time_output, self.e1_output = Funcbar_log(self.timelists,self.e1lists,int(self.bins/4)) 
    #     self.__plot_xlog("time_xlog_logbins"
    #             ,"etip"
    #             ,self.time_output
    #             ,self.e1_output
    #             ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))


    def gen_doslist(self,lists,bin_num):
        bins = np.linspace(min(lists), max(lists),bin_num + 1)
        doslist, b, c = plt.hist(lists,bins=bins,density=1)
        plt.clf()
        lsplist = [(b[i]+b[i+1])/2 for i in range(len(b) - 1)]
        return lsplist, doslist

    def gen_doslist_log(self,lists,bin_num):
        lists2 = [ele for ele in lists if ele>0]
        bins = np.logspace(np.log10(min(lists2)), np.log10(max(lists2)),bin_num + 1)
        doslist, b, c = plt.hist(lists2,bins=bins,density=1)
        plt.clf()
        del lists2 
        lsplist = [10**( (np.log10(b[i]) + np.log10(b[i+1]))/2 )  for i in range(len(b) - 1)]
        return lsplist, doslist

    def gen_doslist_mixed_linearFine(self,lists,bins_trct,bins_num_log):
        # lists2 = [ele for ele in lists if ele>1]
        # lists2 = lists
        bins_line = np.linspace(1.0, bins_trct - 1, bins_trct - 1)
        # print(max(lists2))
        bins_log = np.logspace(np.log10(bins_trct), np.log10(max(lists)), bins_num_log + 1)
        bins = np.hstack([bins_line,bins_log])
        # print(bins)
        doslist, b, c = plt.hist(lists,bins=bins,density=1)
        plt.clf()
        # del lists2
        lsplist_line = bins_line
        lsplist_log = np.array([np.sqrt(bins_log[i]*bins_log[i+1]) for i in range(bins_num_log)])
        lsplist = np.hstack([lsplist_line,lsplist_log])
        widthlist_line = np.array([1.0 for i in range(bins_trct-1)])
        widthlist_log = np.array([bins_log[i+1]-bins_log[i] for i in range(bins_num_log)])
        widthlist = np.hstack([widthlist_line,widthlist_log])
        print(widthlist)
        # print(lsplist)
        
        print(lsplist[0],doslist[0])
        return lsplist, doslist, widthlist

    def gen_doslist_mixed(self,lists,bins_trct,bins_num_log):
        lists2 = [ele for ele in lists if ele>1]
        # lists2 = lists
        bins_line = np.linspace(2.0, bins_trct - 1, int(bins_trct/10))
        bins_log = np.logspace(np.log10(bins_trct), np.log10(max(lists2)), bins_num_log + 1)
        bins = np.hstack([bins_line,bins_log])
        # print(bins)
        doslist, b, c = plt.hist(lists2,bins=bins,density=1)
        plt.clf()
        del lists2
        lsplist_line = bins_line
        lsplist_log = np.array([np.sqrt(bins_log[i]*bins_log[i+1]) for i in range(bins_num_log)])
        lsplist = np.hstack([lsplist_line,lsplist_log])
        widthlist_line = np.array([1.0 for i in range(bins_trct-2)])
        widthlist_log = np.array([bins_log[i+1]-bins_log[i] for i in range(bins_num_log)])
        widthlist = np.hstack([widthlist_line,widthlist_log])
        print(widthlist)
        # print(lsplist)
        
        print(lsplist[0],doslist[0])
        return lsplist, doslist, widthlist        
            

    def __time_to_int(self,t):
        return int(math.ceil(np.sqrt(1+4*t)*0.5-0.5))

    def __time_to_int2(self,t):
        return int(math.ceil(np.sqrt(1+8*t)*0.5-0.5))

    def __imgpath(self,datatype):
        return self.dir +"plot/"+ "{datatype}_{model}{a}_T={b}_timemax={c}_samplenum={d}.png".format(datatype=datatype,a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model,d=int(self.end-self.start))
    
    def __imgpath2(self,datatype,i):
        return self.dir +"plot/EEs-t/{datatype}_{model}{a}_T={b}_timemax={c}_{d}.png".format(datatype=datatype,a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)
    def __imgpath3(self,datatype,dirname,i):
        return self.dir +"plot/"+dirname+"/{datatype}_{model}{a}_T={b}_timemax={c}_{d}.png".format(datatype=datatype,a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,d="%06d"%i,model=self.model)

    def __plot_ylog_lb(self,xlabel,ylabel,x,y,label):
        title = ylabel +"-"+ xlabel +"_ylog"
        img_path = ylabel +"-"+ xlabel +"_ylog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # Min = min(x)
        Max = max(x)
        theo_x = np.linspace(Max/2,Max*0.7,100)
        theo_y = np.exp(-1.*theo_x*self.betac)*1000
        plt.plot(theo_x,theo_y,label="y=exp(-Eb/Tc)",color="green")
        plt.yscale("log")
        plt.title(title)
        plt.plot(x,y,label=label)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
    
    def __plot(self,xlabel,ylabel,x,y,label):
        title = ylabel +"-"+ xlabel
        img_path = ylabel +"-"+ xlabel
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.plot(x,y,label=label)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()


    def __plot_ylog(self,xlabel,ylabel,x,y,label):
        title = ylabel +"-"+ xlabel +"_ylog"
        img_path = ylabel +"-"+ xlabel +"_ylog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale("log")
        plt.title(title)
        plt.plot(x,y,label=label)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()

    def __plot_ylog_entropy(self,xlabel,ylabel,x,y,label):
        title = ylabel +"-"+ xlabel +"_ylog"
        img_path = ylabel +"-"+ xlabel +"_ylog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale("log")
        plt.title(title)
        plt.plot(x,y,label=label)
        plt.axvline(np.log(2),label="log(2)",color='green')
        plt.axvline(np.log(3),label="log(3)",color='yellow')
        plt.axvline(np.log(4),label="log(4)",color='pink')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()

    def __plot_xlog(self,xlabel,ylabel,x,y,label):
        title = ylabel +"-"+ xlabel+"_xlog"
        img_path = ylabel +"-"+ xlabel +"_xlog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.title(title)
        plt.plot(x,y,label=label)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()

    def __plot_xylog(self,xlabel,ylabel,x,y,label):
        title = ylabel +"-"+ xlabel +"_xylog"
        img_path = ylabel +"-"+ xlabel +"_xylog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(title)
        plt.plot(x,y,label=label)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
    
    def __plot_xylog_tau(self,xlabel,ylabel,x,y,label):
        title = ylabel +"-"+ xlabel +"_xylog"
        img_path = ylabel +"-"+ xlabel +"_xylog"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(title)
        plt.plot(x,y,label=label)
        x2 = linspace(10**(np.log10(max(x))*2/3), max(x),1000)
        y2 = x2**(-1.0*self.temperature*self.betac-1)*100
        plt.plot(x2,y2,label="y=x^(-1-alpha)) alpha={}".format(1.0*self.temperature*self.betac),color="pink")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()

    def __plot_xylog_tau_allrange(self):
        xlabel,ylabel = "tau","p(tau)"
        title = ylabel +"-"+ xlabel +"_xylog"
        label = "{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        img_path = ylabel +"-"+ xlabel +"_xylog_tree&dynamics_allrange"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(title)
        plt.plot(self.lsp_taulist, self.dos_taulist,label=label+"dynamics")
        plt.plot(self.lsp_treetau, self.dos_treetau,label=label+"tree")
        taumax = max(self.lsp_taulist)
        x2 = linspace(10**(np.log10(taumax)*2/3), taumax,1000)
        y2 = x2**(-1.0*self.temperature*self.betac-1)*100
        plt.plot(x2,y2,label="y=x^(-1-index)) alpha={}".format(1.0*self.temperature*self.betac),color="pink")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()

    def __plot_xylog_tau_range(self):
        xlabel,ylabel = "tau","p(tau)"
        title = ylabel +"-"+ xlabel +"_xylog"
        label = "{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        img_path = ylabel +"-"+ xlabel +"_xylog_tree&dynamics"
        img_path = self.__imgpath(img_path)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(title)
        plt.plot(self.lsp_taulist_range, self.dos_taulist_range,label=label+"dynamics")
        plt.plot(self.lsp_treetau_range, self.dos_treetau_range,label=label+"tree")
        x2 = linspace(10**(np.log10(self.taumax)*2/3), self.taumax,1000)
        y2 = x2**(-1.0*self.temperature*self.betac-1)*100
        plt.plot(x2,y2,label="y=x^(-1-index)) alpha={}".format(1.0*self.temperature*self.betac),color="pink")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(img_path)
        plt.clf()

    def __hist2d(self,xlabel,ylabel,title,x,y,label,img_path,bins):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        x,y,prob,m = hist2d(x, y, bins=bins, norm=LogNorm(),label=label,density=1)
        colorbar()
        plt.savefig(img_path)
        plt.clf()
        return x,y,prob,m

    def __hist2d_ylog(self,xlabel,ylabel,title,x,y,label,img_path,bins):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.yscale("log")
        hist2d(x, y, bins=bins, norm=LogNorm(),label=label,density=1)
        colorbar()
        plt.savefig(img_path)
        plt.clf()

    def fit_func(self,x,b):
        return np.exp(b*x)

    def __hist2d_taulog(self,xlabel,ylabel,title,x,y,label,img_path,bins):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.yscale("log")
        yy = [i for i in y if i>0]
        xx = [x[j] for j in range(len(x)) if y[j]>0]
        xbins = np.linspace(min(xx),max(xx),bins)
        ybins = np.logspace(np.log10(min(yy)), np.log10(max(yy)),bins)
        hist2d(xx, yy, bins=(xbins,ybins), norm=LogNorm(),label=label,density=1)
        # logy = [log10(i) for i in y if i>0]
        # xx = [i for i in x]
        # index = [i for i,x in enumerate(y) if x<=0]
        # for i in index:
        #     xx.pop(i)

        # xxbar, yy_output = Funcbar(xx, yy, bins)
        # yybar, xx_output = Funcbar_log(yy, xx, bins)
        xxbar, yy_output = [],[]
        yybar, xx_output = [],[]
        plt.plot(xxbar,yy_output,color="pink",label="<tau>(Eb)")
        plt.plot(xx_output,yybar,color="grey",label="tau(<Eb>)")
        del xxbar
        del yy_output
        del yybar
        del xx_output
        

        #theoretical result:
        tau = np.exp(xbins/self.temperature)
        plt.plot(xbins,tau,color="black",label="tau=exp(Eb/T)")
        #fir for all data:
        fit_x = [ele for ele in xx if ele<3.5]
        fit_y = [yy[i] for i in range(len(yy)) if xx[i]<3.5]
        para, pcov = curve_fit(self.fit_func,np.array(fit_x),np.array(fit_y))
        tau_fit = self.fit_func(xbins,para[0])
        plt.plot(xbins,tau_fit,color="green",label="tau=exp(Eb/{bb})".format(bb="%.2f"%(1.0/para[0])))
        colorbar()
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
        del xx
        del yy
        del fit_x
        del fit_y

    def __hist2d_xylog(self,xlabel,ylabel,title,x,y,label,img_path,bins):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xscale("log")
        plt.yscale("log")
        yy = [i for i in y if i>0]
        xx = [x[j] for j in range(len(x)) if y[j]>0]
        xbins = np.logspace(np.log10(min(xx)), np.log10(max(xx)),bins)
        ybins = np.logspace(np.log10(min(yy)), np.log10(max(yy)),bins)
        x,y,prob,m = hist2d(xx, yy, bins=(xbins,ybins), norm=LogNorm(),label=label,density=1)
        # logy = [log10(i) for i in y if i>0]
        # xx = [i for i in x]
        # index = [i for i,x in enumerate(y) if x<=0]
        # for i in index:
        #     xx.pop(i)

        colorbar()
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
        del xx
        del yy
        return x,y,prob,m

    def __hist2d_xlog(self,xlabel,ylabel,title,x,y,label,img_path,bins):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xscale("log")
        xx = [i for i in x if i>0]
        yy = [y[j] for j in range(len(y)) if x[j]>0]
        xbins = np.logspace(np.log10(min(xx)), np.log10(max(xx)),bins)
        ybins = np.linspace(min(yy), max(yy),bins)
        x,y,prob,m = hist2d(xx, yy, bins=(xbins,ybins), norm=LogNorm(),label=label,density=1)
        # logy = [log10(i) for i in y if i>0]
        # xx = [i for i in x]
        # index = [i for i,x in enumerate(y) if x<=0]
        # for i in index:
        #     xx.pop(i)

        colorbar()
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
        del xx
        del yy
        return x,y,prob,m


    def deal_with_lists_for_alloc_memory(self):
        self.deal_with_taulists_pair()
        self.deal_with_eblists_pair()
        self.deal_with_eslists()
        self.deal_with_e1lists() 
        self.deal_with_e0lists()
        self.deal_with_effeb2()
        self.deal_with_effeb1()
        self.deal_with_eb2lists()
        self.deal_with_tauslists()
        self.deal_with_tau0lists()
        self.deal_with_taulists()
        self.deal_with_eb1lists()
    def deal_with_eslists(self):
        self.lsp_eslist, self.dos_eslist = self.gen_doslist(self.eslists,self.bins)
        del self.eslists
        # del self.eslists
    def deal_with_e1lists(self):
        #e1:
        self.__hist2d("E0"
                    ,"E1"
                    ,"prob density"
                    ,self.e0lists
                    ,self.e1lists
                    ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
                    ,self.__imgpath("E0-E1_prob")
                    ,self.bins)
        del self.e1lists
        print("finish: deal_with_e1lists.")
    def deal_with_e0lists(self):
        self.lsp_e0list, self.dos_e0list = self.gen_doslist(self.e0lists,self.bins)
        #already in e0lists:
        # self.__hist2d("E0"
        #     ,"E1"
        #     ,"prob density"
        #     ,self.e0lists
        #     ,self.e1lists
        #     ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        #     ,self.__imgpath("E0-E1_prob")
        #     ,self.bins)
        del self.e0lists
        print("finish: deal_with_e0lists.")
    def deal_with_effeb2(self):
        self.lsp_effeb2, self.dos_effeb2 = self.gen_doslist(self.effeb2lists,self.bins)
        del self.effeb2lists
        self.lsp_eb2list, self.dos_eb2list = self.gen_doslist(self.eb2lists,self.bins)
    def deal_with_effeb1(self):
        self.lsp_effeb1, self.dos_effeb1 = self.gen_doslist(self.effeb1lists,self.bins)
        self.__hist2d_taulog("effEb1"
                    ,"tau"
                    ,"prob density_ylog"
                    ,self.effeb1lists
                    ,self.taulists
                    ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
                    ,self.__imgpath("effEb1-tau_prob_ylog")
                    ,self.bins)
        self.__hist2d_taulog("effEb1"
                    ,"tau0"
                    ,"prob density_ylog"
                    ,self.effeb1lists
                    ,self.tau0lists
                    ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
                    ,self.__imgpath("effEb1-tau0_prob_ylog")
                    ,self.bins)
        del self.effeb1lists
        print("finish: deal_with_effeb1.")
    def deal_with_eb2lists(self):
        self.lsp_eb2list, self.dos_eb2list = self.gen_doslist(self.eb2lists,self.bins)
        self.__hist2d("Eb1"
                    ,"Eb2"
                    ,"prob density"
                    ,self.eb1lists
                    ,self.eb2lists
                    ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
                    ,self.__imgpath("Eb1-Eb2_prob")
                    ,self.bins)
        self.__hist2d("Eb2"
                    ,"taus"
                    ,"prob density"
                    ,self.eb2lists
                    ,self.tauslists
                    ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
                    ,self.__imgpath("Eb2-taus_prob")
                    ,self.bins)
        self.__hist2d_taulog("Eb2"
                    ,"taus"
                    ,"prob density_ylog"
                    ,self.eb2lists
                    ,self.tauslists
                    ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
                    ,self.__imgpath("Eb2-taus_prob_ylog")
                    ,self.bins)
        del self.eb2lists
        print("finish: deal_with_eb2lists.")
    def deal_with_tauslists(self):
        self.lsp_tauslist, self.dos_tauslist = self.gen_doslist_log(self.tauslists,int(self.bins/2))

        #exsit in eb2lists:
        # self.__hist2d("Eb2"
        #             ,"taus"
        #             ,"prob density"
        #             ,self.eb2lists
        #             ,self.tauslists
        #             ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        #             ,self.__imgpath("Eb2-taus_prob")
        #             ,self.bins)#  
        # self.__hist2d_taulog("Eb2"
        #             ,"taus"
        #             ,"prob density_ylog"
        #             ,self.eb2lists
        #             ,self.tauslists
        #             ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        #             ,self.__imgpath("Eb2-taus_prob_ylog")
        #             ,self.bins)#
        del self.tauslists
        print("finish: deal_with_tauslists.")
    def deal_with_tau0lists(self):
        self.lsp_tau0list, self.dos_tau0list = self.gen_doslist_log(self.tau0lists,int(self.bins/2))
        self.__hist2d("Eb1"
                    ,"tau0"
                    ,"prob density"
                    ,self.eb1lists
                    ,self.tau0lists
                    ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
                    ,self.__imgpath("Eb1-tau0_prob")
                    ,self.bins)#
        self.__hist2d_taulog("Eb1"
                    ,"tau0"
                    ,"prob density_ylog"
                    ,self.eb1lists
                    ,self.tau0lists
                    ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
                    ,self.__imgpath("Eb1-tau0_prob_ylog")
                    ,self.bins)#
        #exist in effeb1:
        # self.__hist2d_taulog("effEb1"
        #             ,"tau0"
        #             ,"prob density_ylog"
        #             ,self.effeb1lists
        #             ,self.tau0lists
        #             ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        #             ,self.__imgpath("effEb1-tau0_prob_ylog")
        #             ,self.bins)
        del self.tau0lists
        print("finish: deal_with_tau0lists.")
    def deal_with_taulists(self):
        self.lsp_taulist, self.dos_taulist = self.gen_doslist_log(self.taulists,int(self.bins/2))
        taulist_str = self.dir + "p_dynamics(tau)_{model}{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model)
        self.printf(self.lsp_taulist, self.dos_taulist, taulist_str)
        self.__hist2d("Eb1"
                    ,"tau"
                    ,"prob density"
                    ,self.eb1lists
                    ,self.taulists
                    ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
                    ,self.__imgpath("Eb1-tau_prob")
                    ,self.bins)#
        self.__hist2d_taulog("Eb1"
                    ,"tau"
                    ,"prob density_ylog"
                    ,self.eb1lists
                    ,self.taulists
                    ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
                    ,self.__imgpath("Eb1-tau_prob_ylog")
                    ,self.bins)#
        self.taulists = [ele for ele in self.taulists if ele>self.taumin and ele<=self.taumax]
        self.lsp_taulist_range, self.dos_taulist_range = self.gen_doslist_log(self.taulists,int(self.bins/2))
        taulist_range_str = self.dir + "p_dynamics(tau)_range_{model}{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model)
        self.printf(self.lsp_taulist_range, self.dos_taulist_range, taulist_range_str)
        # self.__hist2d_taulog("effEb1"
        #             ,"tau"
        #             ,"prob density_ylog"
        #             ,self.effeb1lists
        #             ,self.taulists
        #             ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        #             ,self.__imgpath("effEb1-tau_prob_ylog")
        #             ,self.bins)
        del self.taulists
        print("finish: deal_with_taulists.")
    def deal_with_taulists_pair(self):
        prob, x, y, m = self.__hist2d_xylog("tau0"
            ,"tau1"
            ,"prob density_xylog"
            ,self.tau0lists_pair
            ,self.tau1lists_pair
            ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
            ,self.__imgpath("tau0-tau1_pairprob_xylog")
            ,self.bins + 1)
        
        prob_str = self.dir + "prob_dynamics(tau)_{model}{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model)
        tau_str = self.dir + "tau0-tau1_prob_dynamics_{model}{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model)
        self.printf2d(x,y,prob,tau_str,prob_str)
        ###### need to dump 3 array of distribution of tau0-tau1
    def deal_with_eblists_pair(self):
        prob, x, y, m = self.__hist2d("Eb0"
            ,"Eb1"
            ,"prob density"
            ,self.eb0lists_pair
            ,self.eb1lists_pair
            ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
            ,self.__imgpath("Eb0-Eb1_pairprob")
            ,self.bins)
        xout, yout, probout = self.treeArrhenius2d(x,y,prob)
        prob_str = self.dir + "prob_tree(tau)_{model}{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model)
        tau_str = self.dir + "tau0-tau1_prob_tree_{model}{a}_T={b}_timemax={c}.t".format(a=self.num,b="%.2f"%self.temperature,c="%.0le"%self.timemax,model=self.model)
        self.printf2d(xout,yout,probout,tau_str,prob_str)
    # def treeArrhenius2d(self,lsp,dos):
    def treeArrhenius2d(self,x,lsp,prob):
        xout = [np.exp(ele/self.temperature) for ele in x]
        lspout = [np.exp(ele/self.temperature) for ele in lsp]
        dosout = [[] for i in range(len(x) - 1)]
        for j in range(len(x) - 1):
            dosout[j] = [prob[j][i]*self.temperature/np.sqrt(lspout[i]*lspout[i+1]) for i in range(len(x) - 1)]
        return xout, lspout, dosout
    def printf2d(self,x,y,prob,path1,path2):
        with open(path1,'w+',encoding='utf-8') as fp:
            for i in range(len(x)):
                fp.write("%le"%x[i]+" "+"%le"%y[i]+"\n")
        with open(path2,'w+',encoding='utf-8') as fp:
            # fp.write(len(x) - 1)
            # fp.write("\n")
            for i in range(len(x) - 1):
                for j in range(len(x) - 1):
                    fp.write("%le"%prob[i][j]+"\n")
    def deal_with_eb1lists(self):
        self.lsp_eb1list, self.dos_eb1list = self.gen_doslist(self.eb1lists,self.bins)
        #exists in eb2lists:
        # self.__hist2d("Eb1"
        #             ,"Eb2"
        #             ,"prob density"
        #             ,self.eb1lists
        #             ,self.eb2lists
        #             ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        #             ,self.__imgpath("Eb1-Eb2_prob")
        #             ,self.bins)
        # self.__hist2d("Eb1"
        #             ,"tau"
        #             ,"prob density"
        #             ,self.eb1lists
        #             ,self.taulists
        #             ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        #             ,self.__imgpath("Eb1-tau_prob")
        #             ,self.bins)
        # self.__hist2d("Eb1"
        #             ,"tau0"
        #             ,"prob density"
        #             ,self.eb1lists
        #             ,self.tau0lists
        #             ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        #             ,self.__imgpath("Eb1-tau0_prob")
        #             ,self.bins)
        # self.__hist2d_taulog("Eb1"
        #             ,"tau"
        #             ,"prob density_ylog"
        #             ,self.eb1lists
        #             ,self.taulists
        #             ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        #             ,self.__imgpath("Eb1-tau_prob_ylog")
        #             ,self.bins)
        # self.__hist2d_taulog("Eb1"
        #             ,"tau0"
        #             ,"prob density_ylog"
        #             ,self.eb1lists
        #             ,self.tau0lists
        #             ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
        #             ,self.__imgpath("Eb1-tau0_prob_ylog")
        #             ,self.bins)
        del self.eb1lists
        print("finish: deal_with_eb1lists.")

    




    def Bolzmann_mod(self,lsp,dos):
        bolzmann_factor = np.exp(-1.0*np.array(lsp) /self.temperature)
        bolzmann_factor2 = np.exp(-1.0*np.array(lsp) * self.betac)  
        dos_arr = np.array(dos)
        factor_sum = np.dot(bolzmann_factor,dos_arr)
        factor_sum2 = np.dot(bolzmann_factor2,dos_arr)
        delta = lsp[1] - lsp[0]
        dos_output = [dos_arr[i]*bolzmann_factor[i]/factor_sum/delta for i in range(len(dos))]  
        dos_output2 = [dos_arr[i]*bolzmann_factor2[i]/factor_sum2/delta for i in range(len(dos))] 
        return dos_output ,dos_output2

    def __plot_dynamicsE_dos(self):
        # self.lsp_eslist, self.dos_eslist = self.gen_doslist(self.eslists,self.bins)
        # del self.eslists

        # self.lsp_effeb2, self.dos_effeb2 = self.gen_doslist(self.effeb2lists,self.bins)
        # del self.effeb2lists
        # self.lsp_eb2list, self.dos_eb2list = self.gen_doslist(self.eb2lists,self.bins)

        # self.lsp_effeb1, self.dos_effeb1 = self.gen_doslist(self.effeb1lists,self.bins)

        # self.lsp_e0list, self.dos_e0list = self.gen_doslist(self.e0lists,self.bins)
        # self.lsp_eb1list, self.dos_eb1list = self.gen_doslist(self.eb1lists,self.bins)
                
        self.__plot("E0"
                ,"pdf"
                ,self.lsp_e0list
                ,self.dos_e0list
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("Es"
                ,"pdf"
                ,self.lsp_eslist
                ,self.dos_eslist
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("Eb1"
                ,"pdf"
                ,self.lsp_eb1list
                ,self.dos_eb1list
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("Eb2"
                ,"pdf"
                ,self.lsp_eb2list
                ,self.dos_eb2list
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        #eff Eb:
        self.__plot("effEb1"
                ,"pdf"
                ,self.lsp_effeb1
                ,self.dos_effeb1
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("effEb2"
                ,"pdf"
                ,self.lsp_effeb2
                ,self.dos_effeb2
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        #ylog:
        self.__plot_ylog("Eb1"
                ,"pdf"
                ,self.lsp_eb1list
                ,self.dos_eb1list
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_ylog("Eb2"
                ,"pdf"
                ,self.lsp_eb2list
                ,self.dos_eb2list
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))  
        self.__plot_ylog("effEb1"
                ,"pdf"
                ,self.lsp_effeb1
                ,self.dos_effeb1
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_ylog("effEb2"
                ,"pdf"
                ,self.lsp_effeb2
                ,self.dos_effeb2
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        #combine plot Eb1 & Eb2:
        self.__plot_ebdos()
        # self.__plot("Tree Barrier"
        #         ,"pdf"
        #         ,"DOS-Tree Barrier"
        #         ,self.lsp_tblist
        #         ,self.dos_tblist
        #         ,"rem{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature)
        #         ,self.__imgpath("DOS_TreeBarrier"))
        print("finish: plot DOS-E in dynamics.")

    def __plot_ebdos(self):
        plt.xlabel("Eb")
        plt.ylabel("pdf")
        plt.yscale("log")
        plt.title("DOS_Eb1Eb2&effEb1Eb2")
        Min = min(self.lsp_eb1list)
        Max = 1.2*max(self.lsp_eb1list) - min(self.lsp_eb1list)
        plt.xlim(Min,Max)
        x = np.linspace(0,3,100)
        y = np.exp(-1.*x*self.betac)*0.01
        plt.plot(x,y,label="y=exp(-Eb/Tc)",color="green")
        plt.plot(self.lsp_eb1list,self.dos_eb1list,label="Eb1",color="blue")
        plt.plot(self.lsp_eb2list,self.dos_eb2list,label="Eb2",color="pink")
        plt.plot(self.lsp_effeb1,self.dos_effeb1,label="effEb1",color="grey")
        plt.plot(self.lsp_effeb2,self.dos_effeb2,label="effEb2",color="black")
        plt.plot(self.lsp_treeleb, self.dos_treeleb,label="Tree Eb",color="violet")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(self.__imgpath("DOS_Eb1Eb2&effEb1Eb2_ylog"))
        plt.clf()

    def __plot_tau_dos(self):
        # self.dos_logtauslist = [i for i in self.dos_tauslist if i>0]
        # self.dos_logtau0list = [i for i in self.dos_tau0list if i>0]
        # self.dos_logtaulist = [i for i in self.dos_taulist if i>0]
        # self.lsp_logtauslist, self.dos_logtauslist = DOS2(self.dos_logtauslist,self.bins2)
        # self.lsp_logtau0list, self.dos_logtau0list = DOS2(self.dos_logtau0list,self.bins2)
        # self.lsp_logtaulist, self.dos_logtaulist = DOS2(self.dos_logtaulist,self.bins2)

        # self.lsp_tau0list, self.dos_tau0list = self.gen_doslist_log(self.tau0lists,int(self.bins/2))
        # self.lsp_tauslist, self.dos_tauslist = self.gen_doslist_log(self.tauslists,int(self.bins/2))
        # self.lsp_taulist, self.dos_taulist = self.gen_doslist_log(self.taulists,int(self.bins/2))

        self.__plot("tau0"
                ,"pdf"
                ,self.lsp_tau0list
                ,self.dos_tau0list
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("taus"
                ,"pdf"
                ,self.lsp_tauslist
                ,self.dos_tauslist
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("tau"
                ,"pdf"
                ,self.lsp_taulist
                ,self.dos_taulist
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

        #xylog:
        self.__plot_xylog_tau("tau0"
                ,"pdf"
                ,self.lsp_tau0list
                ,self.dos_tau0list
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xylog_tau("taus"
                ,"pdf"
                ,self.lsp_tauslist
                ,self.dos_tauslist
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xylog_tau("tau"
                ,"pdf"
                ,self.lsp_taulist
                ,self.dos_taulist
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

        self.__plot_xylog_tau_range()
        self.__plot_xylog_tau_allrange()

    def __plot_hist2d(self):
        self.__hist2d("E0"
                    ,"E1"
                    ,"prob density"
                    ,self.e0lists
                    ,self.e1lists
                    ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
                    ,self.__imgpath("E0-E1_prob")
                    ,self.bins)#
        # self.e0lists = []
        # self.e1lists = []

        self.__hist2d("Eb1"
                    ,"Eb2"
                    ,"prob density"
                    ,self.eb1lists
                    ,self.eb2lists
                    ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
                    ,self.__imgpath("Eb1-Eb2_prob")
                    ,self.bins)#

        self.__hist2d("Eb1"
                    ,"tau"
                    ,"prob density"
                    ,self.eb1lists
                    ,self.taulists
                    ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
                    ,self.__imgpath("Eb1-tau_prob")
                    ,self.bins)#
        self.__hist2d("Eb2"
                    ,"taus"
                    ,"prob density"
                    ,self.eb2lists
                    ,self.tauslists
                    ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
                    ,self.__imgpath("Eb2-taus_prob")
                    ,self.bins)#
        self.__hist2d("Eb1"
                    ,"tau0"
                    ,"prob density"
                    ,self.eb1lists
                    ,self.tau0lists
                    ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
                    ,self.__imgpath("Eb1-tau0_prob")
                    ,self.bins)#

        self.__hist2d_taulog("Eb1"
                    ,"tau"
                    ,"prob density_ylog"
                    ,self.eb1lists
                    ,self.taulists
                    ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
                    ,self.__imgpath("Eb1-tau_prob_ylog")
                    ,self.bins)#
        self.__hist2d_taulog("Eb2"
                    ,"taus"
                    ,"prob density_ylog"
                    ,self.eb2lists
                    ,self.tauslists
                    ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
                    ,self.__imgpath("Eb2-taus_prob_ylog")
                    ,self.bins)#
        self.__hist2d_taulog("Eb1"
                    ,"tau0"
                    ,"prob density_ylog"
                    ,self.eb1lists
                    ,self.tau0lists
                    ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
                    ,self.__imgpath("Eb1-tau0_prob_ylog")
                    ,self.bins)#
        # self.eb1lists = []
        # self.eb2lists = []
        # self.tauslists = []
        
        self.__hist2d_taulog("effEb1"
                    ,"tau"
                    ,"prob density_ylog"
                    ,self.effeb1lists
                    ,self.taulists
                    ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
                    ,self.__imgpath("effEb1-tau_prob_ylog")
                    ,self.bins)

        self.__hist2d_taulog("effEb1"
                    ,"tau0"
                    ,"prob density_ylog"
                    ,self.effeb1lists
                    ,self.tau0lists
                    ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model)
                    ,self.__imgpath("effEb1-tau0_prob_ylog")
                    ,self.bins)
        # self.effeb1lists = []
        # self.effeb2lists = []
        # self.taulists = []
        # self.tau0lists = []
        


    
    def __plot_eitaui_t(self):
        self.__plot("time"
                ,"e1"
                ,self.e1timelist
                ,self.e1barlist
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("time"
                ,"e0"
                ,self.e1timelist
                ,self.e0barlist
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("time"
                ,"tau0"
                ,self.e1timelist
                ,self.tau0barlist
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xylog("time"
                ,"tau0"
                ,self.e1timelist
                ,self.tau0barlist
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot("time"
                ,"tau1"
                ,self.tau1timelist
                ,self.tau1barlist
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xlog("time"
                ,"e0"
                ,self.e1timelist
                ,self.e0barlist
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        print("finish: plot_eitaui_t.")
    def __plot_remain_prob_t(self):
        self.__plot("time"
                ,"remaining prob(1.5tw,tw)"
                ,self.ageing_func_timelist
                ,self.ageing_func_rate
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xlog("time"
                ,"remaining prob(1.5tw,tw)"
                ,self.ageing_func_timelist
                ,self.ageing_func_rate
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xylog("time"
                ,"remaining prob(1.5tw,tw)"
                ,self.ageing_func_timelist
                ,self.ageing_func_rate
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

        self.__plot("time"
                ,"remaining prob(t+{tw},t)".format(tw="%.0le"%self.tw)
                ,self.ageing_func_timelist_fixed
                ,self.ageing_func_rate_fixed
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xlog("time"
                ,"remaining prob(t+{tw},t)".format(tw="%.0le"%self.tw)
                ,self.ageing_func_timelist_fixed
                ,self.ageing_func_rate_fixed
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xylog("time"
                ,"remaining prob(t+{tw},t)".format(tw="%.0le"%self.tw)
                ,self.ageing_func_timelist_fixed
                ,self.ageing_func_rate_fixed
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))

        self.__plot("Delta"
                ,"remaining prob(Delta,{t})".format(t="%.0le"%self.begint)
                ,self.ageing_func_timelist_beginfixed
                ,self.ageing_func_rate_beginfixed
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xlog("Delta"
                ,"remaining prob(Delta,{t})".format(t="%.0le"%self.begint)
                ,self.ageing_func_timelist_beginfixed
                ,self.ageing_func_rate_beginfixed
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xylog("Delta"
                ,"remaining prob(Delta,{t})".format(t="%.0le"%self.begint)
                ,self.ageing_func_timelist_beginfixed
                ,self.ageing_func_rate_beginfixed
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        print("finish: plot_remain_prob_t.")
    def __plot_Etip_t(self):
        self.__plot_xlog("time"
                ,"Etip_dt=1e6"
                ,self.e0timelist1
                ,self.e0barlist1
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        self.__plot_xlog("time"
                ,"Etip_dt=1e7"
                ,self.e0timelist2
                ,self.e0barlist2
                ,"{model}{a}_T={b}".format(a=self.num,b="%.2f"%self.temperature,model=self.model))
        print("finish: plot Etip-t by average in dt.")
    def plot_data2(self):
        #self.__plot(xlabel,ylabel,title,x,y,label,self.imgpath(datatype))
        mkdir(self.dir + "plot/")
        mkdir(self.dir + "plot/png/")
        # self.plot_lowest_barriers()
        # self.__plot_E_t()
        # self.__plot_EEs_t()
        # self.__plot_treeE_dos()
        
        self.read_data_dynamics()
        # self.deal_with_taulists_pair()
        # self.deal_with_eblists_pair()

        # self.__plot_Etip_t()
        # self.__plot_remain_prob_t()
        self.__plot_eitaui_t()
        ##
        
        # self.deal_with_lists_for_alloc_memory()
        # self.__plot_dynamicsE_dos()
        # self.__plot_tau_dos()
        print("finish: plot DOS-tau.")
        # self.__plot_hist2d()

    def plot_data(self):
        #self.__plot(xlabel,ylabel,title,x,y,label,self.imgpath(datatype))
        mkdir(self.dir + "plot/")
        mkdir(self.dir + "plot/png/")
        self.__plot_lowest_barriers()
        self.__plot_E_t()
        self.__plot_EEs_t()
        self.__plot_treeE_dos()
        
        self.read_data_dynamics()
        self.__plot_Etip_t()
        self.__plot_remain_prob_t()
        self.__plot_eitaui_t()
        ##
        self.deal_with_lists_for_alloc_memory()
        self.__plot_dynamicsE_dos()
        self.__plot_tau_dos()
        print("finish: plot DOS-tau.")
        # self.__plot_hist2d()

    def dump_list(self,l,fname):
        l = np.array(l)
        np.save("./data/N={dd}_T={cc}_{ee}/list/".format(dd=self.num,cc="%.2f"%self.temperature,ee="%.0le"%self.timemax)+fname,l)
    
    def read_list(self,fname):
        l = np.load("./data/N={dd}_T={cc}_{ee}/list/".format(dd=self.num,cc="%.2f"%self.temperature,ee="%.0le"%self.timemax)+fname)
        l.tolist()
        return l

    # def dump_data(self):
    #     self.dump_list(self., fname)

    # def plot_remain_prob(self):
        

    #     for i in range(10,samplenum):
    #         act_str = "activation-t_{model}{a}_T={b}_timemax={c}_{d}.t".format(a=num,b="%.2f"%temperature,c="%.0le"%TIME_MAX,d="%06d"%i,model=self.model)
    #         with open("./T={}/".format("%.2f"%temperature)+act_str,'r') as fp:
    #             #fp.readline()
    #             fp.readline()
    #             fp.readline()

    #             while 1:
    #                 linedata = fp.readline()
    #                 if linedata.strip()=="":
    #                     break
    #                 else:
    #                     if(len(linedata.split())<2):
    #                         print(i)
    #                     (time, e0) = [t(s) for t,s in zip((int,float),linedata.split())]
    #                     #(time, e0, es, e1, tip0, saddle, tip1, time_s, tau_tip, tau_saddle) = [t(s) for t,s in zip((int,float,float,float,int,int,int,int,int,int),linedata.split())]
    #                     time_index = int(time/dtime)
    #                     e0lists[time_index].append(e0/num)
