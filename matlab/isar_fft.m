function [salida, ejexf, ejeyf] = isar_fft(G, frecs, angulos, varargin)
% ISAR_FFT  Generate an ISAR image from scattering data using 2-D FFT.
%
%   salida = ISAR_FFT(G, frecs, angulos)
%   [salida, ejexf, ejeyf] = ISAR_FFT(G, frecs, angulos)
%   [...] = ISAR_FFT(..., 'Name', Value)
%
%   Inputs
%   ------
%   G       : Complex scattering matrix  [Nfrecs x Nangulos]
%   frecs   : Frequency vector in Hz     [1 x Nfrecs]
%   angulos : Angle vector in degrees    [Nangulos x 1] or [1 x Nangulos]
%
%   Name-Value Parameters
%   ---------------------
%   'Nfft'        : FFT size for both dimensions (default: 32)
%   'PlotResult'  : Logical flag to display the surface plot (default: false)
%
%   Outputs
%   -------
%   salida : ISAR image matrix (complex, FFT-shifted)
%   ejexf  : Cross-range axis in metres
%   ejeyf  : Range axis in metres
%
%   Author: Carlos Delgado

    % --- Parse optional arguments -------------------------------------------
    p = inputParser;
    addParameter(p, 'Nfft', 32, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'PlotResult', false, @(x) islogical(x) || isnumeric(x));
    parse(p, varargin{:});

    Nfft       = p.Results.Nfft;
    PlotResult = logical(p.Results.PlotResult);

    % --- Physical constants -------------------------------------------------
    c = 3e8;  % Speed of light (m/s)

    % --- Convert angles to radians ------------------------------------------
    angulos_rad = angulos(:).' * pi / 180;

    % --- Centre frequency ---------------------------------------------------
    f0 = (frecs(1) + frecs(end)) / 2;

    % --- 2-D FFT with zero-padding -----------------------------------------
    salida = fftshift(fft2(G, Nfft, Nfft));

    % --- Build spatial axes -------------------------------------------------
    deltaf = frecs(2) - frecs(1);
    deltat = angulos_rad(2) - angulos_rad(1);

    ejeyf = linspace(-c / (4 * deltaf),  c / (4 * deltaf),  Nfft);   % Range
    ejexf = linspace(-c / (4 * f0 * deltat), c / (4 * f0 * deltat), Nfft); % Cross-range

    % --- Optional visualisation ---------------------------------------------
    if PlotResult
        figure;
        surf(ejexf, ejeyf, abs(salida) ./ max(abs(salida(:))));
        colormap(gray);
        brighten(0.7);
        xlabel('Cross-range (m)');
        ylabel('Range (m)');
        zlabel('Normalised reflectivity');
        title('ISAR Image');
    end
end
