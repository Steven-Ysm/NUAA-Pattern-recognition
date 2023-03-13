```python
python tools/visual_log.py --mode val  --json_paths work_dirs/psp_r101_distill_psp_r18_40k_512x512_city/20221128_093755.log.json --select IoU.road,IoU.sidewalk,IoU.building,IoU.wall,IoU.fence  --pic_name IoU_1

python tools/visual_log.py --mode val  --json_paths work_dirs/psp_r101_distill_psp_r18_40k_512x512_city/20221128_093755.log.json --select IoU.pole,IoU.trafficlight,IoU.trafficsign,IoU.vegetation,IoU.terrain  --pic_name IoU_2

python tools/visual_log.py --mode val  --json_paths work_dirs/psp_r101_distill_psp_r18_40k_512x512_city/20221128_093755.log.json --select IoU.sky,IoU.person,IoU.rider,IoU.car,IoU.truck  --pic_name IoU_3

python tools/visual_log.py --mode val  --json_paths work_dirs/psp_r101_distill_psp_r18_40k_512x512_city/20221128_093755.log.json --select IoU.bus,IoU.train,IoU.motorcycle,IoU.bicycle  --pic_name IoU_4
```

Acc

```python
python tools/visual_log.py --mode val --json_paths work_dirs/psp_r101_distill_psp_r18_40k_512x512_city/20221128_093755.log.json --select Acc.road,Acc.sidewalk,Acc.building,Acc.wall,Acc.fence --pic_name Acc_1

python tools/visual_log.py --mode val --json_paths work_dirs/psp_r101_distill_psp_r18_40k_512x512_city/20221128_093755.log.json --select Acc.pole,Acc.trafficlight,Acc.trafficsign,Acc.vegetation,Acc.terrain --pic_name Acc_2

python tools/visual_log.py --mode val --json_paths work_dirs/psp_r101_distill_psp_r18_40k_512x512_city/20221128_093755.log.json --select Acc.sky,Acc.person,Acc.rider,Acc.car,Acc.truck --pic_name Acc_3

python tools/visual_log.py --mode val --json_paths work_dirs/psp_r101_distill_psp_r18_40k_512x512_city/20221128_093755.log.json --select Acc.bus,Acc.train,Acc.motorcycle,Acc.bicycle --pic_name Acc_4
```



```python
python tools/visual_log.py --mode val  --json_paths work_dirs/psp_r101_distill_psp_r18_40k_512x512_city/20221128_093755.log.json --select mAcc,aAcc  --pic_name acc
```



```python
python tools/visual_log.py --mode val  --json_paths work_dirs/psp_r101_distill_psp_r18_40k_512x512_city/20221128_093755.log.json --select mIoU  --pic_name mIoU
```



train

```python
python tools/visual_log_train.py --mode train  --json_paths work_dirs/psp_r101_distill_psp_r18_40k_512x512_city/20221128_093755.log.json --select decode.loss_ce --pic_name decode.loss_ce

python tools/visual_log_train.py --mode train  --json_paths work_dirs/psp_r101_distill_psp_r18_40k_512x512_city/20221128_093755.log.json --select decode.acc_seg --pic_name decode.acc_seg

python tools/visual_log_train.py --mode train  --json_paths work_dirs/psp_r101_distill_psp_r18_40k_512x512_city/20221128_093755.log.json --select aux.loss_ce --pic_name aux.loss_ce

python tools/visual_log_train.py --mode train  --json_paths work_dirs/psp_r101_distill_psp_r18_40k_512x512_city/20221128_093755.log.json --select aux.acc_seg --pic_name aux.acc_seg

python tools/visual_log_train.py --mode train  --json_paths work_dirs/psp_r101_distill_psp_r18_40k_512x512_city/20221128_093755.log.json --select loss_mgd_fea --pic_name loss_mgd_fea

python tools/visual_log_train.py --mode train  --json_paths work_dirs/psp_r101_distill_psp_r18_40k_512x512_city/20221128_093755.log.json --select loss_logit --pic_name loss_logit

python tools/visual_log_train.py --mode train  --json_paths work_dirs/psp_r101_distill_psp_r18_40k_512x512_city/20221128_093755.log.json --select loss --pic_name loss
```

test

```
python tools/test.py configs/pspnet/pspnet_r18-d8_512x512_40k_cityscapes.py deeplabv3_res18_new.pth --format-only --eval-options "imgfile_prefix=./pspnet_test_results" 
```

