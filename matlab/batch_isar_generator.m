% BATCH_ISAR_GENERATOR  Batch ISAR image generation from NewFASANT simulations.
%
%   This script iterates over the dataset directory tree produced by
%   NewFASANT electromagnetic simulations and generates ISAR images at
%   multiple Gaussian noise levels.  The resulting PNG images are saved
%   with filenames that encode the class label and noise level, ready for
%   ingestion by the Python classification pipeline.
%
%   Directory layout expected
%   -------------------------
%   datasets_root/
%     +-- Caja/
%     |     +-- Caja_x/
%     |     |     +-- result/
%     |     |           +-- step0/RcsFieldRP.out
%     |     |           +-- step1/RcsFieldRP.out
%     |     |           +-- ...
%     |     +-- Caja_y/ ...
%     +-- Cilindro/ ...
%     +-- Cono/ ...
%
%   Author: Dario del Saz

%% ====================================================================
%  Configuration
%  ====================================================================
datasets_root = getenv('ISAR_DATASETS_ROOT');
if isempty(datasets_root)
    datasets_root = uigetdir(pwd, 'Select the root dataset directory');
    if isequal(datasets_root, 0)
        error('batch_isar_generator:noPath', ...
              'No dataset directory selected. Set the ISAR_DATASETS_ROOT environment variable or choose a folder.');
    end
end

num_steps     = 120;
noise_levels  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5];

%% ====================================================================
%  Discover class hierarchy
%  ====================================================================
main_classes = dir(datasets_root);
main_classes = main_classes([main_classes.isdir] & ...
               ~ismember({main_classes.name}, {'.', '..'}));

if isempty(main_classes)
    error('batch_isar_generator:noClasses', ...
          'No class directories found in %s.', datasets_root);
end

total_images = 0;
total_errors = 0;

%% ====================================================================
%  Main processing loop
%  ====================================================================
for i = 1:length(main_classes)
    main_class_path = fullfile(datasets_root, main_classes(i).name);

    sub_classes = dir(main_class_path);
    sub_classes = sub_classes([sub_classes.isdir] & ...
                  ~ismember({sub_classes.name}, {'.', '..'}));

    for j = 1:length(sub_classes)
        sub_class_path = fullfile(main_class_path, sub_classes(j).name);
        result_path    = fullfile(sub_class_path, 'result');

        if ~exist(result_path, 'dir')
            fprintf('[SKIP] No result/ folder in %s\n', sub_class_path);
            continue;
        end

        class_name    = sub_classes(j).name;
        output_folder = fullfile(sub_class_path, ...
                        ['isar_images_', lower(class_name), '_electro']);
        if ~exist(output_folder, 'dir')
            mkdir(output_folder);
        end

        fprintf('\n[INFO] Processing class: %s\n', class_name);

        for k = 1:num_steps
            step_folder = fullfile(result_path, sprintf('step%d', k - 1));
            data_file   = fullfile(step_folder, 'RcsFieldRP.out');

            if ~exist(data_file, 'file')
                fprintf('  [WARN] Missing: %s\n', data_file);
                total_errors = total_errors + 1;
                continue;
            end

            for sigma = noise_levels
                try
                    fig = figure('Visible', 'off');
                    parse_newfasant(data_file, sigma, 'PlotResult', true);

                    if sigma == 0
                        img_name = sprintf('%s%d.png', class_name, k);
                    else
                        img_name = sprintf('%s%d_gauss_%.2f.png', ...
                                           class_name, k, sigma);
                    end

                    saveas(fig, fullfile(output_folder, img_name));
                    close(fig);
                    total_images = total_images + 1;

                catch ME
                    fprintf('  [ERROR] step %d, sigma=%.2f in %s: %s\n', ...
                            k, sigma, class_name, ME.message);
                    total_errors = total_errors + 1;
                    if isvalid(fig)
                        close(fig);
                    end
                end
            end

            if mod(k, 20) == 0
                fprintf('  ... processed %d / %d steps\n', k, num_steps);
            end
        end
    end
end

%% ====================================================================
%  Summary
%  ====================================================================
fprintf('\n========================================\n');
fprintf('  COMPLETED\n');
fprintf('  Images generated : %d\n', total_images);
fprintf('  Errors / warnings: %d\n', total_errors);
fprintf('========================================\n');
