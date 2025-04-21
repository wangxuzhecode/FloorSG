#!/bin/bash

#python 1_filter.py I:\\FloorSG\\MRF_test\\S3DIS_area1\\MRF.txt I:\\FloorSG\\MRF_test\\S3DIS_area1_total\\MRF_filter.txt
#
#python 2_refine_roomseg.py I:\\FloorSG\\MRF_test\\S3DIS_area1\\label.txt I:\\FloorSG\\MRF_test\\S3DIS_area1_total\\MRF_filter.txt I:\\FloorSG\\MRF_test\\S3DIS_area1_total\\MRF_filter_refine.txt
#
#python 3_filter_2.py I:\\FloorSG\\MRF_test\\S3DIS_area1_total\\MRF_filter_refine.txt I:\\FloorSG\\MRF_test\\S3DIS_area1_total\\MRF_filter_refine_filter.txt
#
#python 4_process.py I:\\FloorSG\\MRF_test\\S3DIS_area1_total\\MRF_filter_refine_filter.txt I:\\FloorSG\\MRF_test\\S3DIS_area1_total\\MRF_filter_refine_filter_res.txt

python 5_padding.py I:\\FloorSG\\MRF_test\\S3DIS_area1_total\\MRF_filter_refine_filter_res.txt I:\\FloorSG\\MRF_test\\S3DIS_area1_total\\MRF_filter_refine_filter_res_pad.txt 8
python 5_padding.py I:\\FloorSG\\MRF_test\\S3DIS_area1_total\\MRF_filter_refine_filter_res_pad.txt I:\\FloorSG\\MRF_test\\S3DIS_area1_total\\MRF_filter_refine_filter_res_pad.txt 10