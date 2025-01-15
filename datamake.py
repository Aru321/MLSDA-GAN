# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 13:21:05 2021

@author: AruZeng
"""
import medpy
from medpy.io import load
from medpy.io import save
import numpy as np
import os
import SimpleITK as sitk
#用于数据切片
def Datamake(root_l,root_s):
    all_l_names=[]
    all_s_names=[]
    for root, dirs, files in os.walk(root_l):
        all_l_names=(files)
    for root, dirs, files in os.walk(root_s):
        all_s_names=(files)
    #
    all_l_name=[]
    all_s_name=[]
    for i in all_l_names:
        if os.path.splitext(i)[1] == ".img":
            #print(i)
            all_l_name.append(i)
    for i in all_s_names:
        if os.path.splitext(i)[1] == ".img":
            all_s_name.append(i)
    #
    print(all_l_name)
    #
    for file in all_l_name:
        image_path_l = os.path.join(root_l,file)
        image_l,h=load(image_path_l)
        image_l=np.array(image_l)
        #print(image_l.shape)
        cut_cnt=0
       # print(cut_cnt)
        for i in range(0,5):
            for j in range(0,5):
                for k in range(0,5):
                    image_cut=image_l[16*i:64+16*i,16*j:64+16*j,16*k:64+16*k]
                    savImg = sitk.GetImageFromArray(image_cut)
                    sitk.WriteImage(savImg,'./data/train_l_cut'+'/'+file+'_cut'+str(cut_cnt)+'.img')
                    cut_cnt+=1
                    
    for file in all_s_name:
        image_path_s = os.path.join(root_s,file)
        image_s,h=load(image_path_s)
        image_s=np.array(image_s)
        #print(image_l.shape)
        cut_cnt=0
        for i in range(0,5):
            for j in range(0,5):
                for k in range(0,5):
                    image_cut=image_s[16*i:64+16*i,16*j:64+16*j,16*k:64+16*k]
                    savImg = sitk.GetImageFromArray(image_cut)
                    sitk.WriteImage(savImg,'./data/train_s_cut'+'/'+file+'_cut'+str(cut_cnt)+'.img')
                    cut_cnt+=1
#
def DatamakeMuiti(root_l,root_s,root_mri):
    all_l_names=[]
    all_s_names=[]
    all_mri_names=[]
    for root, dirs, files in os.walk(root_l):
        all_l_names=(files)
    for root, dirs, files in os.walk(root_s):
        all_s_names=(files)
    for root, dirs, files in os.walk(root_mri):
        all_mri_names=(files)
    #
    all_l_name=[]
    all_s_name=[]
    all_mri_name=[]
    for i in all_l_names:
        if os.path.splitext(i)[1] == ".img":
            #print(i)
            all_l_name.append(i)
    for i in all_s_names:
        if os.path.splitext(i)[1] == ".img":
            all_s_name.append(i)
    for i in all_mri_names:
        if os.path.splitext(i)[1] == ".img":
            all_mri_name.append(i)
    #
    print(all_l_name)
    #
    for file in all_l_name:
        image_path_l = os.path.join(root_l,file)
        image_l,h=load(image_path_l)
        image_l=np.array(image_l)
        #print(image_l.shape)
        cut_cnt=0
       # print(cut_cnt)
        for i in range(0,5):
            for j in range(0,5):
                for k in range(0,5):
                    image_cut=image_l[16*i:64+16*i,16*j:64+16*j,16*k:64+16*k]
                    savImg = sitk.GetImageFromArray(image_cut)
                    sitk.WriteImage(savImg,'./data/train_l_cut'+'/'+file+'_cut'+str(cut_cnt)+'.img')
                    cut_cnt+=1
                    
    for file in all_s_name:
        image_path_s = os.path.join(root_s,file)
        image_s,h=load(image_path_s)
        image_s=np.array(image_s)
        #print(image_l.shape)
        cut_cnt=0
        for i in range(0,5):
            for j in range(0,5):
                for k in range(0,5):
                    image_cut=image_s[16*i:64+16*i,16*j:64+16*j,16*k:64+16*k]
                    savImg = sitk.GetImageFromArray(image_cut)
                    sitk.WriteImage(savImg,'./data/train_s_cut'+'/'+file+'_cut'+str(cut_cnt)+'.img')
                    cut_cnt+=1
    for file in all_mri_name:
        image_path_mri = os.path.join(root_mri,file)
        image_mri,h=load(image_path_mri)
        image_mri=np.array(image_mri)
        #print(image_l.shape)
        cut_cnt=0
        for i in range(0,5):
            for j in range(0,5):
                for k in range(0,5):
                    image_cut=image_mri[16*i:64+16*i,16*j:64+16*j,16*k:64+16*k]
                    savImg = sitk.GetImageFromArray(image_cut)
                    sitk.WriteImage(savImg,'./data/train_mri_cut'+'/'+file+'_cut'+str(cut_cnt)+'.img')
                    cut_cnt+=1
                    
if __name__ == '__main__':
    Datamake('./data/datastage2/train_ldata','./data/datastage2/train_sdata','./data/datastage2/train_mridata')
 