addpath(genpath('.'));
[bbs, sca] = bb_scan([120; 100; 139; 129], [380, 640], 5);
disp(bbs(:,size(bbs, 2)))
