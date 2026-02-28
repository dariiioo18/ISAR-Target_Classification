function [image_out, freqs, angles] = parse_newfasant(filename, sigma, varargin)
% PARSE_NEWFASANT  Parse a NewFASANT RCS output file and generate an ISAR image.
%
%   image_out = PARSE_NEWFASANT(filename, sigma)
%   [image_out, freqs, angles] = PARSE_NEWFASANT(filename, sigma)
%   [...] = PARSE_NEWFASANT(..., 'Name', Value)
%
%   Inputs
%   ------
%   filename : Path to the NewFASANT RcsFieldRP.out file.
%   sigma    : Standard deviation of additive complex Gaussian noise.
%              Use 0 for a clean image.
%
%   Name-Value Parameters
%   ---------------------
%   'Nfreqs'     : Expected number of frequency points (default: 32)
%   'Nangles'    : Expected number of angular points   (default: 32)
%   'Nfft'       : FFT size passed to isar_fft          (default: 32)
%   'PlotResult' : Display the ISAR image               (default: false)
%
%   Outputs
%   -------
%   image_out : ISAR image matrix (from isar_fft)
%   freqs     : Parsed frequency vector (Hz)
%   angles    : Parsed angle vector (degrees)
%
%   Author: Dario del Saz

    % --- Parse optional arguments -------------------------------------------
    p = inputParser;
    addParameter(p, 'Nfreqs',     32,    @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'Nangles',    32,    @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'Nfft',       32,    @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'PlotResult', false, @(x) islogical(x) || isnumeric(x));
    parse(p, varargin{:});

    Nfreqs     = p.Results.Nfreqs;
    Nangles    = p.Results.Nangles;
    Nfft       = p.Results.Nfft;
    PlotResult = logical(p.Results.PlotResult);

    % --- Pre-allocate -------------------------------------------------------
    freqs        = zeros(1, Nfreqs);
    G_cells      = cell(Nfreqs, 1);
    angles       = zeros(Nangles, 1);
    current_real = zeros(Nangles, 1);
    current_imag = zeros(Nangles, 1);
    current_theta = zeros(Nangles, 1);

    % --- Open file ----------------------------------------------------------
    fid = fopen(filename, 'r');
    if fid == -1
        error('parse_newfasant:fileNotFound', ...
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
                    angles = current_theta;
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
                freqs(freq_count) = str2double(tokens{1}{1});
            end

        elseif startsWith(tline, 'THETA')
            continue;

        elseif ~isempty(tline) && ...
               (isstrprop(tline(1), 'digit') || tline(1) == '-' || tline(1) == '+')

            line_idx = line_idx + 1;
            data = sscanf(tline, '%f');
            if numel(data) < 4
                error('parse_newfasant:incompleteData', ...
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
            angles = current_theta;
        end
    end

    % --- Validate -----------------------------------------------------------
    if freq_count ~= Nfreqs
        error('parse_newfasant:freqMismatch', ...
              'Expected %d frequencies but parsed %d.', Nfreqs, freq_count);
    end
    if line_idx ~= Nangles
        error('parse_newfasant:angleMismatch', ...
              'Expected %d angles per frequency but parsed %d.', Nangles, line_idx);
    end

    % --- Build scattering matrix G(freq, angle) -----------------------------
    G = zeros(Nfreqs, Nangles);
    for k = 1:Nfreqs
        G(k, :) = G_cells{k}.';
    end

    % --- Add complex Gaussian noise -----------------------------------------
    if sigma > 0
        noise = sigma * complex(randn(size(G)), randn(size(G)));
        G = G + noise;
    end

    % --- Generate ISAR image ------------------------------------------------
    image_out = isar_fft(G, freqs, angles, 'Nfft', Nfft, 'PlotResult', PlotResult);
end
