# POSCO_Project



## posco_zed_display

jetson에서 zed 카메라 영상 실행 및 각종 이미지 처리[SR 적용, CLAHE 등]와 관련된 파일

- run the code 
```bash
python zed_display.py
```

- key board command

    l키 -> clahe 적용

    j키 -> 영상 centercrop

    k키 -> SR 적용



## super_resolution
기존의 ESPCN 모델을 본 과제에 맞추어 reimplementation 한 파일


#### Download datasets
- Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

[Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
[Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)


#### Download original code
[https://github.com/Lornatang/ESPCN-PyTorch]


