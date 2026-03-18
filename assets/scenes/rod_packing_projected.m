load_obj = load('/Users/yeonsu/Harvard University Dropbox/Mahadevan Group/Soft Math Lab/ExperimentalData/x-ray-scans/rod-packing-hysteresis/2025-07-03_RodPackingHysteresis/zstack_e00.mat');
zstack = load_obj.zstack;
%%
img = zstack(:,:,500);

%%
imshow(img)

%%
for i = 1:size(zstack,3)
    img = zstack(:,:,i);
    
end