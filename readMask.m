function [] = readMask()

% set hyperparameters
nrow = 96;
ncol = 200;

% read file into data structures
Info = textscan(fopen('experimental_parameters.txt'), '%s');
xoffset = str2num(Info{1, 1}{33, 1}) - 288; 
yoffset = str2num(Info{1 ,1}{30, 1}) - 288;

% load masks
load masks-input; 
num_masks = length(pts_list);
ROI = zeros(ncol, nrow, num_masks);

% create polygon to generate mask
for i=1:num_masks;
    xv = pts_list{i}(:, 2)/2 - xoffset;
    yv = pts_list{i}(:, 1)/2 - yoffset;
    [xq, yq] = meshgrid(1:nrow, 1:ncol);
    in = inpolygon(xq, yq, xv, yv);
    ROI(:, :, i) = in;
end

disp(size(ROI));
for i=1:num_masks;
    disp(ROI(:, :, i))
    disp(nnz(ROI(:, :, i)));
end

% save ROIs
save('masks-output.mat', 'ROI');