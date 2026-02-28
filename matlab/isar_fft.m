function [image_out, cross_range_axis, range_axis] = isar_fft(G, freqs, angles, varargin)
% ISAR_FFT  Generate an ISAR image from scattering data using 2-D FFT.
%
%   image_out = ISAR_FFT(G, freqs, angles)
%   [image_out, cross_range_axis, range_axis] = ISAR_FFT(G, freqs, angles)
%   [...] = ISAR_FFT(..., 'Name', Value)
%
%   Inputs
%   ------
%   G      : Complex scattering matrix  [Nfreqs x Nangles]
%   freqs  : Frequency vector in Hz     [1 x Nfreqs]
%   angles : Angle vector in degrees    [Nangles x 1] or [1 x Nangles]
%
%   Name-Value Parameters
%   ---------------------
%   'Nfft'        : FFT size for both dimensions (default: 32)
%   'PlotResult'  : Logical flag to display the surface plot (default: false)
%
%   Outputs
%   -------
%   image_out        : ISAR image matrix (complex, FFT-shifted)
%   cross_range_axis : Cross-range axis in metres
%   range_axis       : Range axis in metres
%
%   Author: Dario del Saz

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
    angles_rad = angles(:).' * pi / 180;

    % --- Centre frequency ---------------------------------------------------
    f0 = (freqs(1) + freqs(end)) / 2;

    % --- 2-D FFT with zero-padding -----------------------------------------
    image_out = fftshift(fft2(G, Nfft, Nfft));

    % --- Build spatial axes -------------------------------------------------
    deltaf = freqs(2) - freqs(1);
    deltat = angles_rad(2) - angles_rad(1);

    range_axis       = linspace(-c / (4 * deltaf),  c / (4 * deltaf),  Nfft);
    cross_range_axis = linspace(-c / (4 * f0 * deltat), c / (4 * f0 * deltat), Nfft);

    % --- Optional visualisation ---------------------------------------------
    if PlotResult
        figure;
        surf(cross_range_axis, range_axis, abs(image_out) ./ max(abs(image_out(:))));
        colormap(gray);
        brighten(0.7);
        xlabel('Cross-range (m)');
        ylabel('Range (m)');
        zlabel('Normalised reflectivity');
        title('ISAR Image');
    end
end
