directory_ = 'Z:\LuisFel\Simultaneous whole-animal\41592_2014_BFnmeth2964_MOESM189_ESM\SIsoftware\Data\01_Raw\';
savePathIMG = 'Z:\LuisFel\Simultaneous whole-animal\41592_2014_BFnmeth2964_MOESM189_ESM\SIsoftware\Data\02_Rectified\';
directory = dir(directory_);
xCenter = 2054.600; %Fluor: 916.2000; Biolum: 2034.700
yCenter = 1148.700; %Fluor: 918.7000; Biolum: 1171.200
dx = 51.785; %Fluor: 29.8000; Biolum: 51.415;
Nnum = 11;
Crop = 1;
XcutLeft = 1;
YcutUp = 0;
YcutDown = 0;
XcutRight = 1;
stack = 0;
for ii=1:size(directory)
inputFileName = directory(ii).name
if inputFileName ~= "."
if inputFileName ~= ".."
if endsWith(inputFileName,".tif") == 1
%IMG_RAW = im2double(imread([inputFilePath inputFileName],'tiff'));
iminfo = imfinfo(fullfile(directory_, inputFileName),'tiff');
if stack == 1
display("Stack")
STACK = zeros(649, 627,31);
for k=1:size(iminfo)
    IMG_RAW = im2double(imread([directory_ inputFileName],'tiff', k));
    IMG_Rect = ImageRect(IMG_RAW, xCenter, yCenter, dx, Nnum, Crop, XcutLeft, XcutRight, YcutUp, YcutDown);
    IMG_Rect = (IMG_Rect - min(IMG_Rect(:))/(max(IMG_Rect(:)) - min(IMG_Rect(:))))*255.0;
    STACK(:,:,k) = IMG_Rect;
end
size(STACK)
saveastiff(uint8(STACK), [savePathIMG inputFileName(1:end-4) '.tif']);
else
    display("Single image")
    IMG_RAW = im2double(imread([directory_ inputFileName],'tiff'));
    IMG_Rect = ImageRect(IMG_RAW, xCenter, yCenter, dx, Nnum, Crop, XcutLeft, XcutRight, YcutUp, YcutDown);
    imwrite(IMG_Rect,  [savePathIMG inputFileName(1:end-4) '.tif'] );
end
%for i = 0:size(iminfo)
%imwrite(STACK(:,:,i), [savePathIMG inputFileName(1:end-4) '_N' num2str(Nnum) '.tif'],  'WriteMode', 'append');    
%end
%imwrite(STACK,  [savePathIMG inputFileName(1:end-4) '_N' num2str(Nnum) '.tif'] ); 
%imwrite(IMG_Rect,  [savePathIMG inputFileName(1:end-4) '_N' num2str(Nnum) '.tif'] );
end
end
end
end