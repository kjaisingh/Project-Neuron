function [] = readBinMov()

% set hyperparameters
nrow = 200;
ncol = 96;
% clip = 20;
movieFileName = 'movie.tif';

% read file into tmp vector
fid = fopen('movReg.bin');
tmp = fread(fid, '*uint16', 'l');
fclose(fid);

% reshape vector into appropriately oriented, 3D array
L = length(tmp) / (nrow * ncol);
mov = reshape(tmp, [ncol nrow L]);
mov = permute(mov, [2 1 3]);
% mov = mov(:, :, 1:clip);

% save individual frames as tif
options.compress = 'no';
for i=1:L
  tiff = mov(:, :, i);
  outputFileName = sprintf('frames/frame%d.tif', i);
  imwrite(tiff, outputFileName);
end

if nargout > 1
    nframe = L;
end