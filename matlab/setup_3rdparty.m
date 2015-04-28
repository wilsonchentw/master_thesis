function setup_3rdparty(root_dir)
    % Add libsvm, liblinear, vlfeat library path
    run(fullfile(root_dir, '../vlfeat/toolbox/vl_setup'));
    addpath(fullfile(root_dir, '../liblinear/matlab'));
    addpath(fullfile(root_dir, '../libsvm/matlab'));
    %addpath(fullfile(root_dir, '../liblinear-weights/matlab'));
    %addpath(fullfile(root_dir, '../libsvm-weights/matlab'));

    % Add SPAMS(SPArse Modeling Software) path
    addpath(fullfile(root_dir, '../spams/spams-matlab/test_release'));
    addpath(fullfile(root_dir, '../spams/spams-matlab/src_release'));
    addpath(fullfile(root_dir, '../spams/spams-matlab/build'));
end
