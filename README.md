# POSCO_Project



## posco_zed_display [Main 코드]

jetson에서 zed 카메라 영상 실행 및 각종 이미지 처리[SR 적용, CLAHE 등]와 관련된 파일

- run the code 
```bash
python zed_display.py
```

- key board command

    l키 -> clahe 적용

    j키 -> 영상 centercrop

    k키 -> SR 적용



## super_resolution [추가 학습 시 사용]


1. Download datasets (/data 폴더로 옮겨 학습에 사용)

    Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

    [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
    
    [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)
    
    

2. Process dataset
    학습을 위해 데이터셋을 패치화시키는 과정 필요
   
```bash
python scripts/run.py
```

3. run the train code 
```bash
python train.py
```


-  참고
    Download original code [https://github.com/Lornatang/ESPCN-PyTorch]


