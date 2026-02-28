function [salida, frecs, angulos] = postprocesado_isar(filename, sigma, varargin)
% POSTPROCESADO_ISAR  Parse a NewFASANT RCS output file and generate an ISAR image.
%
%   salida = POSTPROCESADO_ISAR(filename, sigma)
%   [salida, frecs, angulos] = POSTPROCESADO_ISAR(filename, sigma)
%   [...] = POSTPROCESADO_ISAR(..., 'Name', Value)
%
%   Inputs
%   ------
%   filename : Path to the NewFASANT RcsFieldRP.out file.
%   sigma    : Standard deviation of additive complex Gaussian noise.
%              Use 0 for a clean image.
%
%   Name-Value Parameters
%   ---------------------
%   'Nfrecs'     : Expected number of frequency points (default: 32)
%   'Nangulos'   : Expected number of angular points   (default: 32)
%   'Nfft'       : FFT size passed to isar_fft          (default: 32)
%   'PlotResult' : Display the ISAR image               (default: false)
%
%   Outputs
%   -------
%   salida  : ISAR image matrix (from isar_fft)
%   frecs   : Parsed frequency vector (Hz)
%   angulos : Parsed angle vector (degrees)
%
%   Author: Carlos Delgado

    % --- Parse optional arguments -------------------------------------------
    p = inputParser;
    addParameter(p, 'Nfrecs',     32,    @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'Nangulos',   32,    @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'Nfft',       32,    @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'PlotResult', false, @(x) islogical(x) || isnumeric(x));
    parse(p, varargin{:});

    Nfrecs     = p.Results.Nfrecs;
    Nang       = p.Results.Nangulos;
    Nfft       = p.Results.Nfft;
    PlotResult = logical(p.Results.PlotResult);

    % --- Pre-allocate -------------------------------------------------------
    frecs        = zeros(1, Nfrecs);
    G_cells      = cell(Nfrecs, 1);
    angulos      = zeros(Nang, 1);
    current_real = zeros(Nang, 1);
    current_imag = zeros(Nang, 1);
    current_theta = zeros(Nang, 1);

    % --- Open file ----------------------------------------------------------
    fid = fopen(filename, 'r');
    if fid == -1
        error('postprocesado_isar:fileNotFound', ...
              'Cannot open file: %s', filename);
    end
    cleanupObj = onCleanup(@() fclose(fid));  % Guarantee file closure

    freq_count = 0;
    line_idx   = 0;

    % --- Parse line by line -------------------------------------------------
    while ~feof(fid)
        tline = strtrim(fgetl(fid));

        if startsWith(tline, '#FREQUENCY')
            % Store previous frequency block
            if freq_count > 0
                G_cells{freq_count} = complex(current_real, current_imag);
                if freq_count == 1
                    angulos = current_theta;
                end
            end

            % Reset buffers
            current_real(:)  = 0;
            current_imag(:)  = 0;
            current_theta(:) = 0;
            line_idx = 0;

            freq_count = freq_count + 1;

            tokens = regexp(tline, '#FREQUENCY\s*=\s*([\d\.E\+\-]+)', 'tokens');
            if ~isempty(tokens)
                frecs(freq_count) = str2double(tokens{1}{1});
            end

        elseif startsWith(tline, 'THETA')
            continue;

        elseif ~isempty(tline) && ...
               (isstrprop(tline(1), 'digit') || tline(1) == '-' || tline(1) == '+')

            line_idx = line_idx + 1;
            data = sscanf(tline, '%f');
            if numel(data) < 4
                error('postprocesado_isar:incompleteData', ...
                      'Incomplete data at frequency #%d, line #%d.', ...
                      freq_count, line_idx);
            end

            current_theta(line_idx) = data(1);  % Theta angle
            current_real(line_idx)  = data(3);   % E_VV real
            current_imag(line_idx)  = data(4);   % E_VV imaginary
        end
    end

    % --- Store last frequency block -----------------------------------------
    if freq_count > 0
        G_cells{freq_count} = complex(current_real, current_imag);
        if freq_count == 1
            angulos = current_theta;
        end
    end

    % --- Validate -----------------------------------------------------------
    if freq_count ~= Nfrecs
        error('postprocesado_isar:freqMismatch', ...
              'Expected %d frequencies but parsed %d.', Nfrecs, freq_count);
    end
    if line_idx ~= Nang
        error('postprocesado_isar:angleMismatch', ...
              'Expected %d angles per frequency but parsed %d.', Nang, line_idx);
    end

    % --- Build scattering matrix G(freq, angle) -----------------------------
    G = zeros(Nfrecs, Nang);
    for k = 1:Nfrecs
        G(k, :) = G_cells{k}.';
    end

    % --- Add complex Gaussian noise -----------------------------------------
    if sigma > 0
        noise = sigma * complex(randn(size(G)), randn(size(G)));
        G = G + noise;
    end

    % --- Generate ISAR image ------------------------------------------------
    salida = isar_fft(G, frecs, angulos, 'Nfft', Nfft, 'PlotResult', PlotResult);
end
