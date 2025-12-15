## Training Data Format 

The training commands should work without any issues if the data is properly formatted following this structure. Feel free to create a github issue if this is not the case.

```
data_root_path (specified in config.yaml's dataset.data_root)
  ├── mvs-synth  
  │   └── ...
  ├── spring  
  │   └── ...
  ├── dynamicreplica  
  │   └── ...
  ├── pointodyssey  
  │   └── ...
  ├── tartanair  
  │   └── ...
```


### MVS-Synth
```
mvs-synth/GTAV_1080
  ├── 0001  
  │   └── images
  │     └── 0001.png
  │     └── 0002.png
  │     └── ...
  │   └── depths
  │     └── 0001.exr
  │     └── 0002.exr
  │     └── ...
  ├── 0002  
  │   └── images
  │   └── depths
  ├── ...
  ├── 0119
  │   └── images
  │   └── depths
```

### Spring
```
spring/train
  ├── 0001  
  │   └── frame_left
  │     └── frame_left_0001.png
  │     └── frame_left_0002.png
  │     └── ...
  │   └── disp1_left
  │     └── disp1_left_0001.dsp5
  │     └── disp1_left_0002.dsp5
  │     └── ...
  ├── 0002  
  │   └── frame_left
  │   └── disp1_left
  ├── ...
  ├── 0047
  │   └── frame_left
  │   └── disp1_left
```

### TartanAir
```
tartanair
  ├── abandonedfactory  
  │   └── Easy
  │     └── P000
  |          └── depth_left
  |            └── 000000_left_depth.npy 
  |          └── image_left
  |            └── 000000_left.png
  |     └── P001
  |          └── depth_left
  |            └── 000001_left_depth.npy 
  |          └── image_left
  |            └── 000001_left.png
  │     └── ...
  │   └── Hard
  │     └── P000
  │     └── P001
  │     └── ...
  ├── carwelding  
  │   └── Easy
  │   └── Hard
```

### PointOdyssey
```
pointodyssey/train
  ├── cnb_dlab_0215_3rd  
  │   └── depths
  │     └── depth_00000.png
  │     └── depth_00001.png
  │     └── ...
  │   └── rgbs
  │     └── rgb_00000.jpg
  │     └── rgb_00001.jpg
  │     └── ...
  ├── dancing  
  │   └── depths
  │   └── rgbs
  ├── ...
```

### DynamicReplica
```
dynamic_replica/train
  ├── 009850-3_obj_source_left  
  │   └── depths
  │     └── 009850-3_obj_source_left_0000.geometric.png
  │     └── 009850-3_obj_source_left_0001.geometric.png
  │     └── ...
  │   └── images
  │     └── 009850-3_obj_source_left-0000.png
  │     └── 009850-3_obj_source_left-0001.png
  │     └── ...
  ├── 00e2a3-3_obj_source_left  
  │   └── depths
  │   └── images
  ├── ...
```