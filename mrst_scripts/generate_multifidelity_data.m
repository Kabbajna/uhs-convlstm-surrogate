%% MULTI-FIDELITY DEEPONET DATA GENERATION WITH ROBUST SAVING
% This script generates data at three fidelity levels with automatic saving
% and comprehensive error logging after EACH simulation

clear all; close all; clc;

%% CRITICAL SETUP - LOGGING AND DIRECTORIES
% Google Drive path for Narjisse
google_drive_path = '/Users/narjisse/Library/CloudStorage/GoogleDrive-kabbaj.narjisse@gmail.com/My Drive/Deeponet';

% Check if Google Drive is accessible
if ~exist(google_drive_path, 'dir')
    error('Google Drive path not found: %s\nPlease check that Google Drive is running and synced.', google_drive_path);
end

% Create base directory in Google Drive
base_dir = fullfile(google_drive_path, 'multifidelity_deeponet_data');
if ~exist(base_dir, 'dir')
    mkdir(base_dir);
    fprintf('Created directory: %s\n', base_dir);
end

% Create subdirectories for each fidelity level
hf_dir = fullfile(base_dir, 'high_fidelity');
mf_dir = fullfile(base_dir, 'medium_fidelity');
lf_dir = fullfile(base_dir, 'low_fidelity');
log_dir = fullfile(base_dir, 'logs');
backup_dir = fullfile(base_dir, 'backups');

dirs_to_create = {hf_dir, mf_dir, lf_dir, log_dir, backup_dir};
for i = 1:length(dirs_to_create)
    if ~exist(dirs_to_create{i}, 'dir')
        mkdir(dirs_to_create{i});
    end
end

%% INITIALIZE MASTER LOG FILE
master_log_file = fullfile(log_dir, sprintf('master_log_%s.txt', datestr(now, 'yyyymmdd_HHMMSS')));

% Also create a sync status file to monitor Google Drive uploads
sync_status_file = fullfile(log_dir, 'google_drive_sync_status.txt');
fid_sync = fopen(sync_status_file, 'w');
fprintf(fid_sync, 'Google Drive Sync Status - Started %s\n', datestr(now));
fprintf(fid_sync, 'Path: %s\n\n', base_dir);
fclose(fid_sync);

diary(master_log_file);
fprintf('================================================================================\n');
fprintf('MULTI-FIDELITY DEEPONET DATA GENERATION - STARTED AT %s\n', datestr(now));
fprintf('================================================================================\n');
fprintf('Data will be saved to Google Drive:\n');
fprintf('Path: %s\n\n', base_dir);

%% SIMULATION PARAMETERS
% Number of simulations for each fidelity
n_sims_hf = 100;   % High-fidelity (40×40×20)
n_sims_mf = 500;   % Medium-fidelity (20×20×20)
n_sims_lf = 2000;  % Low-fidelity (10×10×10)

% Grid configurations
grid_configs = struct();
grid_configs.hf = [40, 40, 20];
grid_configs.mf = [20, 20, 20];
grid_configs.lf = [10, 10, 10];

%% PARAMETER RANGES (same for all fidelities)
param_ranges = struct();
param_ranges.K_min = 50 * milli * darcy;
param_ranges.K_max = 500 * milli * darcy;
param_ranges.phi_min = 0.15;
param_ranges.phi_max = 0.35;
param_ranges.Q_mult_min = 0.5;
param_ranges.Q_mult_max = 1.5;
param_ranges.cycle_dur_min = 150;
param_ranges.cycle_dur_max = 210;
param_ranges.hyst_mult_min = 0.8;
param_ranges.hyst_mult_max = 1.2;

%% GENERATE PARAMETER SETS USING LHS
rng(42); % For reproducibility

% High-fidelity parameters
lhs_hf = lhsdesign(n_sims_hf, 5);
params_hf = scale_lhs_to_params(lhs_hf, param_ranges);

% Medium-fidelity parameters
lhs_mf = lhsdesign(n_sims_mf, 5);
params_mf = scale_lhs_to_params(lhs_mf, param_ranges);

% Low-fidelity parameters
lhs_lf = lhsdesign(n_sims_lf, 5);
params_lf = scale_lhs_to_params(lhs_lf, param_ranges);

%% SAVE PARAMETER CONFIGURATION
param_config = struct();
param_config.param_ranges = param_ranges;
param_config.params_hf = params_hf;
param_config.params_mf = params_mf;
param_config.params_lf = params_lf;
param_config.grid_configs = grid_configs;
param_config.n_sims = [n_sims_hf, n_sims_mf, n_sims_lf];

save(fullfile(base_dir, 'parameter_configuration.mat'), 'param_config', '-v7.3');
fprintf('Parameter configuration saved.\n\n');

%% CHECK FOR EXISTING PROGRESS
progress_file = fullfile(base_dir, 'simulation_progress.mat');
if exist(progress_file, 'file')
    load(progress_file, 'progress');
    fprintf('RESUMING FROM PREVIOUS RUN:\n');
    fprintf('  High-fidelity completed: %d/%d\n', length(progress.completed_hf), n_sims_hf);
    fprintf('  Medium-fidelity completed: %d/%d\n', length(progress.completed_mf), n_sims_mf);
    fprintf('  Low-fidelity completed: %d/%d\n\n', length(progress.completed_lf), n_sims_lf);
else
    progress = struct();
    progress.completed_hf = [];
    progress.completed_mf = [];
    progress.completed_lf = [];
    progress.failed_hf = [];
    progress.failed_mf = [];
    progress.failed_lf = [];
end

%% MAIN SIMULATION LOOP
total_start = tic;

% Run all fidelity levels
fidelity_levels = {'high', 'medium', 'low'};
n_sims_all = [n_sims_hf, n_sims_mf, n_sims_lf];
params_all = {params_hf, params_mf, params_lf};
dirs_all = {hf_dir, mf_dir, lf_dir};
completed_all = {progress.completed_hf, progress.completed_mf, progress.completed_lf};
failed_all = {progress.failed_hf, progress.failed_mf, progress.failed_lf};

for fid_idx = 1:3
    fidelity = fidelity_levels{fid_idx};
    n_sims = n_sims_all(fid_idx);
    params = params_all{fid_idx};
    save_dir = dirs_all{fid_idx};
    completed = completed_all{fid_idx};
    failed = failed_all{fid_idx};
    
    fprintf('\n================================================================================\n');
    fprintf('STARTING %s FIDELITY SIMULATIONS\n', upper(fidelity));
    fprintf('================================================================================\n\n');
    
    for sim_id = 1:n_sims
        % Skip if already completed or failed
        if ismember(sim_id, completed)
            continue;
        end
        if ismember(sim_id, failed)
            fprintf('Simulation %s_%04d previously failed, skipping.\n', fidelity, sim_id);
            continue;
        end
        
        % Individual simulation timing
        sim_start = tic;
        
        % Create simulation-specific log file
        sim_log_file = fullfile(log_dir, sprintf('%s_sim_%04d_log.txt', fidelity, sim_id));
        sim_error_file = fullfile(log_dir, sprintf('%s_sim_%04d_ERROR.txt', fidelity, sim_id));
        
        % Log simulation start
        fprintf('\n--- %s Simulation %d/%d (%.1f%% complete) ---\n', ...
                upper(fidelity), sim_id, n_sims, (length(completed)/n_sims)*100);
        fprintf('Started at: %s\n', datestr(now));
        
        % Extract parameters
        K = params(sim_id, 1);
        phi = params(sim_id, 2);
        Q_mult = params(sim_id, 3);
        cycle_dur = params(sim_id, 4);
        hyst_mult = params(sim_id, 5);
        
        fprintf('Parameters: K=%.1f mD, phi=%.2f, Q_mult=%.2f, cycle=%d d, hyst=%.2f\n', ...
                K/(milli*darcy), phi, Q_mult, cycle_dur, hyst_mult);
        
        % Set grid configuration
        if strcmp(fidelity, 'high')
            nx = 40; ny = 40; nz = 20;
        elseif strcmp(fidelity, 'medium')
            nx = 20; ny = 20; nz = 20;
        else
            nx = 10; ny = 10; nz = 10;
        end
        
        % Try to run simulation with comprehensive error handling
        try
            % Run the simulation
            sim_data = run_single_simulation_safe(nx, ny, nz, K, phi, Q_mult, ...
                                                 cycle_dur, hyst_mult, sim_id, ...
                                                 fidelity, sim_log_file);
            
            % CRITICAL: Save immediately after simulation
            save_filename = sprintf('%s_sim_%04d.mat', fidelity, sim_id);
            save_path = fullfile(save_dir, save_filename);
            
            % Save with verification
            save_success = false;
            save_attempts = 0;
            max_attempts = 5;
            
            while ~save_success && save_attempts < max_attempts
                save_attempts = save_attempts + 1;
                try
                    % Save the data - use v7 for compression compatibility
                    save(save_path, 'sim_data', '-v7');
                    
                    % Verify the save by loading it back
                    test_load = load(save_path, 'sim_data');
                    if isstruct(test_load.sim_data) && isfield(test_load.sim_data, 'sim_id')
                        save_success = true;
                        fprintf('SUCCESS: Data saved and verified at %s\n', save_path);
                        
                        % Get file size for Google Drive monitoring
                        file_info = dir(save_path);
                        file_size_mb = file_info.bytes / 1024^2;
                        fprintf('File size: %.1f MB\n', file_size_mb);
                        
                        % Update Google Drive sync status
                        fid_sync = fopen(sync_status_file, 'a');
                        fprintf(fid_sync, '%s: Saved %s_sim_%04d.mat (%.1f MB)\n', ...
                                datestr(now, 'HH:MM:SS'), fidelity, sim_id, file_size_mb);
                        fclose(fid_sync);
                        
                        % Create backup in local temp before Google Drive sync
                        backup_path = fullfile(backup_dir, sprintf('%s_sim_%04d_backup.mat', fidelity, sim_id));
                        copyfile(save_path, backup_path);
                        
                        % Check Google Drive sync status (optional)
                        pause(0.5); % Give Google Drive a moment to detect the new file
                    else
                        error('Saved data failed verification');
                    end
                catch save_err
                    fprintf('WARNING: Save attempt %d failed: %s\n', save_attempts, save_err.message);
                    
                    % If Google Drive is having issues, try local save first
                    if contains(save_err.message, 'Permission denied') || contains(save_err.message, 'cannot write')
                        temp_local_path = fullfile(tempdir, sprintf('%s_sim_%04d_temp.mat', fidelity, sim_id));
                        try
                            save(temp_local_path, 'sim_data', '-v7');
                            movefile(temp_local_path, save_path, 'f');
                            fprintf('Used temp directory workaround for Google Drive sync issues\n');
                        catch
                            pause(5); % Wait longer if Google Drive is syncing
                        end
                    else
                        pause(2); % Normal retry delay
                    end
                end
            end
            
            if ~save_success
                error('Failed to save simulation data after %d attempts', max_attempts);
            end
            
            % Update progress
            completed = [completed, sim_id];
            
            % Save progress immediately
            update_progress(progress_file, fidelity, completed, failed);
            
            % Log success
            sim_time = toc(sim_start);
            fprintf('Simulation completed in %.1f minutes\n', sim_time/60);
            fprintf('Recovery factor: %.1f%%\n', sim_data.results.recovery_factor);
            
            % Time estimate
            n_remaining = n_sims - length(completed);
            if length(completed) > 0
                avg_time = toc(total_start) / length(completed);
                est_remaining = avg_time * n_remaining / 3600;
                fprintf('Estimated time remaining for %s fidelity: %.1f hours\n', fidelity, est_remaining);
            end
            
        catch ME
            % Log error comprehensively
            fprintf('\nERROR in %s simulation %d!\n', fidelity, sim_id);
            fprintf('Error message: %s\n', ME.message);
            
            % Write detailed error file
            fid_err = fopen(sim_error_file, 'w');
            fprintf(fid_err, 'ERROR LOG FOR %s SIMULATION %04d\n', upper(fidelity), sim_id);
            fprintf(fid_err, '=====================================\n\n');
            fprintf(fid_err, 'Timestamp: %s\n\n', datestr(now));
            fprintf(fid_err, 'Parameters:\n');
            fprintf(fid_err, '  K = %.3e\n', K);
            fprintf(fid_err, '  phi = %.3f\n', phi);
            fprintf(fid_err, '  Q_mult = %.3f\n', Q_mult);
            fprintf(fid_err, '  cycle_dur = %d\n', cycle_dur);
            fprintf(fid_err, '  hyst_mult = %.3f\n', hyst_mult);
            fprintf(fid_err, '  Grid: %d x %d x %d\n\n', nx, ny, nz);
            fprintf(fid_err, 'Error Message: %s\n\n', ME.message);
            fprintf(fid_err, 'Stack Trace:\n');
            for i = 1:length(ME.stack)
                fprintf(fid_err, '  In %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
            end
            fclose(fid_err);
            
            % Update failed list
            failed = [failed, sim_id];
            update_progress(progress_file, fidelity, completed, failed);
            
            % Continue to next simulation
            fprintf('Error logged to: %s\n', sim_error_file);
            fprintf('Continuing to next simulation...\n');
        end
        
        % Force MATLAB to flush all buffers
        drawnow;
        diary off;
        diary(master_log_file);
    end
    
    % Summary for this fidelity level
    fprintf('\n%s FIDELITY SUMMARY:\n', upper(fidelity));
    fprintf('  Completed: %d/%d (%.1f%%)\n', length(completed), n_sims, length(completed)/n_sims*100);
    fprintf('  Failed: %d\n', length(failed));
end

%% FINAL SUMMARY
total_time = toc(total_start);
fprintf('\n================================================================================\n');
fprintf('MULTI-FIDELITY DATA GENERATION COMPLETE\n');
fprintf('================================================================================\n');
fprintf('Total time: %.1f hours\n', total_time/3600);
fprintf('High-fidelity: %d/%d completed\n', length(progress.completed_hf), n_sims_hf);
fprintf('Medium-fidelity: %d/%d completed\n', length(progress.completed_mf), n_sims_mf);
fprintf('Low-fidelity: %d/%d completed\n', length(progress.completed_lf), n_sims_lf);
fprintf('\nData saved in Google Drive at:\n');
fprintf('%s\n', base_dir);
fprintf('\nIMPORTANT: Please wait for Google Drive to finish syncing all files.\n');
fprintf('Check sync status in: %s\n', sync_status_file);

% Final sync status report
fid_sync = fopen(sync_status_file, 'a');
fprintf(fid_sync, '\n================================================================================\n');
fprintf(fid_sync, 'Generation Complete at %s\n', datestr(now));
fprintf(fid_sync, 'Total files created: %d\n', length(progress.completed_hf) + length(progress.completed_mf) + length(progress.completed_lf));
fprintf(fid_sync, 'Please ensure all files are synced to Google Drive before closing the computer.\n');
fclose(fid_sync);

diary off;

%% HELPER FUNCTIONS

function params = scale_lhs_to_params(lhs_samples, ranges)
    % Scale LHS samples to actual parameter ranges
    n_samples = size(lhs_samples, 1);
    params = zeros(n_samples, 5);
    
    params(:,1) = ranges.K_min + (ranges.K_max - ranges.K_min) * lhs_samples(:,1);
    params(:,2) = ranges.phi_min + (ranges.phi_max - ranges.phi_min) * lhs_samples(:,2);
    params(:,3) = ranges.Q_mult_min + (ranges.Q_mult_max - ranges.Q_mult_min) * lhs_samples(:,3);
    params(:,4) = round(ranges.cycle_dur_min + (ranges.cycle_dur_max - ranges.cycle_dur_min) * lhs_samples(:,4));
    params(:,5) = ranges.hyst_mult_min + (ranges.hyst_mult_max - ranges.hyst_mult_min) * lhs_samples(:,5);
end

function update_progress(progress_file, fidelity, completed, failed)
    % Safely update progress file
    try
        if exist(progress_file, 'file')
            load(progress_file, 'progress');
        else
            progress = struct();
        end
        
        if strcmp(fidelity, 'high')
            progress.completed_hf = completed;
            progress.failed_hf = failed;
        elseif strcmp(fidelity, 'medium')
            progress.completed_mf = completed;
            progress.failed_mf = failed;
        else
            progress.completed_lf = completed;
            progress.failed_lf = failed;
        end
        
        progress.last_update = datestr(now);
        
        % Save with temporary file to prevent corruption
        temp_file = [progress_file, '.tmp'];
        save(temp_file, 'progress', '-v7.3');
        movefile(temp_file, progress_file, 'f');
        
    catch
        fprintf('WARNING: Could not update progress file\n');
    end
end

function sim_data = run_single_simulation_safe(nx, ny, nz, K, phi, Q_mult, ...
                                              cycle_dur, hyst_mult, sim_id, ...
                                              fidelity, log_file)
    % Run simulation with all error handling and logging
    
    % Redirect output to individual log file
    diary(log_file);
    
    try
        % Add required MRST modules
        mrstModule clear;
        mrstModule add ad-core ad-props ad-blackoil incomp
        
        % Load fitted parameters
        if ~exist('brooks_corey_fitted_csv.mat', 'file')
            error('Required file brooks_corey_fitted_csv.mat not found');
        end
        load('brooks_corey_fitted_csv.mat', 'fitted_params');
        
        % Define functions
        krW_BC = @(sw, Swc, Sgr, n, krw_max) krw_max * max(0, min(1, (sw - Swc) / (1 - Swc - Sgr))).^n;
        krG_BC = @(sg, Swc, Sgr, n, krg_max) krg_max * max(0, min(1, (sg - Sgr) / (1 - Swc - Sgr))).^n;
        pcWG_BC = @(sw, Swc, Sgr, Pe, n) min(1e6, Pe * max(0.001, min(0.999, (sw - Swc) / (1 - Swc - Sgr))).^(-n));
        
        % Extract and scale parameters
        Swc = fitted_params.Swc;
        Sgr_dr = fitted_params.drainage.Sgr * hyst_mult;
        n_water_dr = fitted_params.drainage.n_water;
        krw_max_dr = fitted_params.drainage.krw_max;
        n_gas_dr = fitted_params.drainage.n_gas;
        krg_max_dr = fitted_params.drainage.krg_max;
        Pe_drainage = fitted_params.drainage.Pe * hyst_mult;
        n_drainage = fitted_params.drainage.n_pc;
        
        Sgr_imb = 0.18 * hyst_mult;
        n_water_imb = 3.5;
        krw_max_imb = fitted_params.imbibition.krw_max;
        n_gas_imb = fitted_params.imbibition.n_gas;
        krg_max_imb = fitted_params.imbibition.krg_max;
        Pe_imbibition = fitted_params.imbibition.Pe * hyst_mult;
        n_imbibition = fitted_params.imbibition.n_pc;
        
        % Fluid properties
        rho_brine = 1000;
        mu_brine = 0.80e-3;
        rho_H2 = 15;
        mu_H2 = 0.01e-3;
        sigma = 72e-3;
        P_ref = 14.8e6;
        
        % Aquifer properties
        aquifer_thickness = 100;
        aquifer_size = 1000;
        
        % Create grid
        G = cartGrid([nx, ny, nz], [aquifer_size, aquifer_size, aquifer_thickness]);
        G = computeGeometry(G);
        
        % Rock properties
        Kh = K;
        Kv = K * 0.1;
        rock = makeRock(G, [Kh, Kh, Kv], phi);
        
        % Calculate injection rate
        PV_total = sum(G.cells.volumes .* rock.poro);
        target_total_PV = 0.009;
        injection_days = cycle_dur * 1.5;
        Q_injection = (target_total_PV * PV_total) / (injection_days * day) * Q_mult;
        
        % Create fluid models
        fluid_drainage = initSimpleADIFluid('phases', 'WG', ...
                                           'mu', [mu_brine, mu_H2], ...
                                           'rho', [rho_brine, rho_H2]);
        fluid_drainage.krW = @(sw) krW_BC(sw, Swc, Sgr_dr, n_water_dr, krw_max_dr);
        fluid_drainage.krG = @(sg) krG_BC(sg, Swc, Sgr_dr, n_gas_dr, krg_max_dr);
        fluid_drainage.pcWG = @(sw) pcWG_BC(sw, Swc, Sgr_dr, Pe_drainage, n_drainage);
        
        fluid_imbibition = initSimpleADIFluid('phases', 'WG', ...
                                             'mu', [mu_brine, mu_H2], ...
                                             'rho', [rho_brine, rho_H2]);
        fluid_imbibition.krW = @(sw) krW_BC(sw, Swc, Sgr_imb, n_water_imb, krw_max_imb);
        fluid_imbibition.krG = @(sg) krG_BC(sg, Swc, Sgr_imb, n_gas_imb, krg_max_imb);
        fluid_imbibition.pcWG = @(sw) pcWG_BC(sw, Swc, Sgr_imb, Pe_imbibition, n_imbibition);
        
        % Initial conditions
        gravity on;
        g = norm(gravity);
        state = initResSol(G, P_ref);
        state.s = repmat([1, 0], G.cells.num, 1);
        
        % Hydrostatic pressure
        z_ref = aquifer_thickness/2;
        for i = 1:G.cells.num
            z = G.cells.centroids(i,3);
            dz = z_ref - z;
            state.pressure(i) = P_ref + rho_brine * g * dz;
        end
        
        % Well setup
        ix_center = round(nx/2);
        iy_center = round(ny/2);
        well_cells = [];
        n_layers = min(round(nz/2+8), nz);
        for iz = 1:n_layers
            well_cells = [well_cells; sub2ind([nx, ny, nz], ix_center, iy_center, iz)];
        end
        
        % Well parameters
        dx = aquifer_size / nx;
        dy = aquifer_size / ny;
        dz = aquifer_thickness / nz;
        rw = 0.2;
        re = 0.28 * sqrt(dx^2 + dy^2) / sqrt(2);
        skin = -5;
        WI_skin = 2 * pi * K * dz / (log(re/rw) + skin);
        
        W_inj = addWell([], G, rock, well_cells, ...
                       'Type', 'rate', 'Val', Q_injection, ...
                       'Comp_i', [0, 1], 'Name', 'Injector', ...
                       'WI', repmat(WI_skin, length(well_cells), 1));
        
        W_prod = addWell([], G, rock, well_cells, ...
                        'Type', 'rate', 'Val', -Q_injection, ...
                        'Comp_i', [0, 1], 'Name', 'Producer', ...
                        'lims', struct('bhp', P_ref - 20*barsa));
        
        % Boundary conditions
        bc = [];
        bottom_faces = find(G.faces.centroids(:,3) > aquifer_thickness - 0.1);
        P_bottom = P_ref + rho_brine * g * aquifer_thickness;
        bc = addBC(bc, bottom_faces, 'pressure', P_bottom, 'sat', [0.9, 0.1]);
        
        % Schedule
        durations = [cycle_dur, cycle_dur/2, cycle_dur/2, cycle_dur/2] * day;
        phase_names = {'Injection1', 'Production1', 'Injection2', 'Production2'};
        
        schedule = struct();
        schedule.step.val = [];
        schedule.step.control = [];
        
        % Adaptive timestep based on grid resolution
        dt_base = 1.5 * day;
        dt_factor = (40*40*20) / (nx*ny*nz); % Scale timestep with grid
        dt = dt_base * sqrt(dt_factor);
        dt = max(0.5*day, min(3*day, dt)); % Limit timestep range
        
        for phase = 1:4
            n_steps = round(durations(phase) / dt);
            dt_phase = repmat(dt, n_steps, 1);
            total = sum(dt_phase);
            if total ~= durations(phase)
                dt_phase(end) = dt_phase(end) + (durations(phase) - total);
            end
            schedule.step.val = [schedule.step.val; dt_phase];
            if mod(phase, 2) == 1
                schedule.step.control = [schedule.step.control; ones(size(dt_phase))];
            else
                schedule.step.control = [schedule.step.control; 2*ones(size(dt_phase))];
            end
        end
        
        schedule.control(1).W = W_inj;
        schedule.control(1).bc = bc;
        schedule.control(2).W = W_prod;
        schedule.control(2).bc = bc;
        
        % Solver settings
        nls = NonLinearSolver();
        nls.maxIterations = 25;
        nls.useRelaxation = true;
        nls.relaxationParameter = 1;
        
        % Run simulation
        fprintf('Starting %s fidelity simulation %d...\n', fidelity, sim_id);
        
        wellSols = {};
        states = {};
        timestep_data = [];
        
        % Frequency of saving states
        save_frequency = max(1, round(5 * (20*20*20) / (nx*ny*nz)));
        
        phase_starts = [1; find(diff(schedule.step.control) ~= 0) + 1];
        phase_ends = [find(diff(schedule.step.control) ~= 0); length(schedule.step.val)];
        
        Sg_max_reached = zeros(G.cells.num, 1);
        
        sim_timer = tic;
        for phase = 1:4
            fprintf('  Phase %d: %s...', phase, phase_names{phase});
            
            if mod(phase, 2) == 1
                model_phase = GenericBlackOilModel(G, rock, fluid_drainage, ...
                                                 'water', true, 'gas', true, 'oil', false);
                phase_type = 'drainage';
            else
                model_phase = GenericBlackOilModel(G, rock, fluid_imbibition, ...
                                                 'water', true, 'gas', true, 'oil', false);
                phase_type = 'imbibition';
            end
            
            model_phase.toleranceCNV = 1e-6;
            model_phase.toleranceMB = 1e-10;
            
            phase_schedule = schedule;
            phase_schedule.step.val = schedule.step.val(phase_starts(phase):phase_ends(phase));
            phase_schedule.step.control = schedule.step.control(phase_starts(phase):phase_ends(phase));
            
            [ws, sts, ~] = simulateScheduleAD(state, model_phase, phase_schedule, ...
                                              'NonLinearSolver', nls, 'Verbose', false);
            
            % Collect timestep data
            for i = 1:save_frequency:length(sts)
                ts_data = struct();
                ts_data.phase = phase;
                ts_data.phase_type = phase_type;
                ts_data.timestep_global = length(timestep_data) + 1;
                ts_data.time = sum(schedule.step.val(1:(phase_starts(phase)+i-1))) / day;
                ts_data.Sw = reshape(sts{i}.s(:,1), nx, ny, nz);
                ts_data.Sg = reshape(sts{i}.s(:,2), nx, ny, nz);
                ts_data.P = reshape(sts{i}.pressure, nx, ny, nz);
                
                Sg_current = sts{i}.s(:,2);
                Sg_max_reached = max(Sg_max_reached, Sg_current);
                ts_data.Sg_max = reshape(Sg_max_reached, nx, ny, nz);
                
                if mod(phase, 2) == 1
                    ts_data.Q = Q_injection;
                    ts_data.operation = 'injection';
                else
                    ts_data.Q = -Q_injection;
                    ts_data.operation = 'production';
                end
                
                timestep_data = [timestep_data, ts_data];
            end
            
            wellSols = [wellSols; ws];
            states = [states; sts];
            state = sts{end};
            
            fprintf(' done.\n');
        end
        
        sim_time = toc(sim_timer);
        fprintf('  Simulation core completed in %.1f seconds.\n', sim_time);
        
        % Calculate results
        dt_all = schedule.step.val;
        injected_ts = zeros(length(wellSols), 1);
        produced_ts = zeros(length(wellSols), 1);
        
        for i = 1:length(wellSols)
            if ~isempty(wellSols{i})
                qGs = abs(wellSols{i}(1).qGs);
                if schedule.step.control(i) == 1
                    injected_ts(i) = qGs * dt_all(i);
                else
                    produced_ts(i) = qGs * dt_all(i);
                end
            end
        end
        
        total_injected_PV = sum(injected_ts) / PV_total;
        total_produced_PV = sum(produced_ts) / PV_total;
        recovery = total_produced_PV / total_injected_PV * 100;
        
        % Compile all data
        sim_data = struct();
        sim_data.sim_id = sim_id;
        sim_data.fidelity = fidelity;
        sim_data.grid_size = [nx, ny, nz];
        
        sim_data.params.K = K;
        sim_data.params.phi = phi;
        sim_data.params.Q_mult = Q_mult;
        sim_data.params.cycle_dur = cycle_dur;
        sim_data.params.hyst_mult = hyst_mult;
        
        sim_data.hyst_params.Swc = Swc;
        sim_data.hyst_params.Sgr_dr = Sgr_dr;
        sim_data.hyst_params.Sgr_imb = Sgr_imb;
        sim_data.hyst_params.all_drainage = struct('n_water', n_water_dr, 'krw_max', krw_max_dr, ...
                                                   'n_gas', n_gas_dr, 'krg_max', krg_max_dr, ...
                                                   'Pe', Pe_drainage, 'n_pc', n_drainage);
        sim_data.hyst_params.all_imbibition = struct('n_water', n_water_imb, 'krw_max', krw_max_imb, ...
                                                     'n_gas', n_gas_imb, 'krg_max', krg_max_imb, ...
                                                     'Pe', Pe_imbibition, 'n_pc', n_imbibition);
        
        sim_data.grid.nx = nx;
        sim_data.grid.ny = ny;
        sim_data.grid.nz = nz;
        sim_data.grid.PV_total = PV_total;
        
        sim_data.timesteps = timestep_data;
        sim_data.n_timesteps = length(timestep_data);
        
        bhp = zeros(length(wellSols), 1);
        rates = zeros(length(wellSols), 1);
        for i = 1:length(wellSols)
            if ~isempty(wellSols{i})
                bhp(i) = wellSols{i}(1).bhp;
                rates(i) = wellSols{i}(1).qGs;
            end
        end
        
        sim_data.well_data.bhp = bhp;
        sim_data.well_data.rates = rates;
        
        sim_data.results.recovery_factor = recovery;
        sim_data.results.total_injected_PV = total_injected_PV;
        sim_data.results.total_produced_PV = total_produced_PV;
        sim_data.results.max_bhp = max(bhp) / 1e6;
        sim_data.results.min_bhp = min(bhp) / 1e6;
        sim_data.results.computation_time = sim_time;
        
        sg_final = states{end}.s(:,2);
        sim_data.final_state.avg_sg_plume = mean(sg_final(sg_final > 0.05));
        sim_data.final_state.max_sg = max(sg_final);
        
        fprintf('  Results: Recovery = %.1f%%\n', recovery);
        
        diary off;
        
    catch ME
        diary off;
        rethrow(ME);
    end
end