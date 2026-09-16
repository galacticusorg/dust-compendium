#!/usr/bin/env perl
use strict;
use warnings;
use Cwd;
use lib $ENV{'GALACTICUS_EXEC_PATH'         }."/perl";
use lib $ENV{'GALACTICUS_ANALYSIS_PERL_PATH'}."/perl";
use PDL;
use PDL::NiceSlice;
use PDL::IO::HDF5;
use Galacticus::Options;
use GnuPlot::PrettyPlots;
use GnuPlot::LaTeX;
use Galacticus::DustAttenuation;

# Get options.
my %options = (
    plotFile           => "plot.pdf",
    attenuationFile    => "data/attenuations.hdf5",
    opticalDepthIndex  => 1,
    inclinationIndex   => 1
    );
&Galacticus::Options::Parse_Options(\@ARGV,\%options);
# Load the attenuation data.
my $attenuationFile        = new PDL::IO::HDF5($options{'attenuationFile'});
my $inclination            = $attenuationFile->dataset('inclination'                   )->get();
my $wavelength             = $attenuationFile->dataset('wavelength'                    )->get();
my $opticalDepth           = $attenuationFile->dataset('opticalDepth'                  )->get();
my $spheroidRadius         = $attenuationFile->dataset('spheroidRadius'                )->get();
my $attenuation            = $attenuationFile->dataset('attenuationSpheroid'           )->get();
my $attenuationUncertainty = $attenuationFile->dataset('attenuationUncertaintySpheroid')->get();    

# Convert attenuation to extinction.
my $extinction            = -2.5*log10($attenuation);
my $extinctionUncertainty = +2.5*$attenuationUncertainty/$attenuation/log(10.0);

# Extract extinction at V-band (0.55um).
my $wavelengthIndices          = pdl sequence(nelem($wavelength));
my $wavelengthReversed         = $wavelength       ->(-1:0)->copy();
my $wavelengthIndicesReversed  = $wavelengthIndices->(-1:0)->copy();
my $wavelengthV                = pdl 0.55;
(my $wavelengthIndex)          = interpolate($wavelengthV,$wavelengthReversed,$wavelengthIndicesReversed);
my $indices                    = pdl zeros(4,nelem($spheroidRadius));
$indices->((0),:)             .= sequence(nelem($spheroidRadius));
$indices->((1),:)             .= $options{'opticalDepthIndex'};
$indices->((2),:)             .= $options{'inclinationIndex' };
$indices->((3),:)             .= $wavelengthIndex;
my $extinctionV                = $extinction           ->interpND($indices);
my $extinctionVUncertainty     = $extinctionUncertainty->interpND($indices);

# Determine ranges to plot.
my $xMinimum = $spheroidRadius->minimum();
my $xMaximum = $spheroidRadius->maximum();
my $yMinimum = $extinctionV   ->minimum();
my $yMaximum = $extinctionV   ->maximum();
$yMinimum .= -0.02
    if ( $yMinimum > -0.02 );

# Make the plot.
my $plot;
my $gnuPlot;
(my $plotFileTeX = $options{'plotFile'}) =~ s/\.pdf$/.tex/;
open($gnuPlot,"|gnuplot");
print $gnuPlot "set terminal cairolatex pdf standalone color lw 2\n";
print $gnuPlot "set output '".$plotFileTeX."'\n";
print $gnuPlot "set title offset 0,-1  'Extinction at inclination \$i=".sprintf("%4.1f",$inclination->(($options{'inclinationIndex'})))."^\\circ\$ and face-on optical depth \$\\tau_\\mathrm{V,0}=".sprintf("%6.2f",$opticalDepth->(($options{'opticalDepthIndex'})))."\$'\n";
print $gnuPlot "set xlabel 'Spheroid radius; \$r_\\mathrm{s}/r_\\mathrm{d}\$'\n";
print $gnuPlot "set ylabel 'Extinction; \$A_\\mathrm{V}\$'\n";
print $gnuPlot "set lmargin screen 0.15\n";
print $gnuPlot "set rmargin screen 0.95\n";
print $gnuPlot "set bmargin screen 0.15\n";
print $gnuPlot "set tmargin screen 0.95\n";
print $gnuPlot "set key spacing 1.2\n";
print $gnuPlot "set key at screen 0.25,0.40\n";
print $gnuPlot "set key left\n";
print $gnuPlot "set key bottom\n";
print $gnuPlot "set xrange [".$xMinimum.":".$xMaximum."]\n";
print $gnuPlot "set yrange [".$yMinimum.":".$yMaximum."]\n";
print $gnuPlot "set pointsize 1.0\n";
&GnuPlot::PrettyPlots::Prepare_Dataset(
    \$plot,
    $spheroidRadius,
    $extinctionV+$extinctionVUncertainty,
    y2 => $extinctionV-$extinctionVUncertainty,
    style      => "filledCurve",
    weight     => [2,1],
    transparency => 0.8,
    color      => $GnuPlot::PrettyPlots::colorPairs{'mediumSeaGreen'}
    );
&GnuPlot::PrettyPlots::Prepare_Dataset(
    \$plot,
    $spheroidRadius,
    $extinctionV,
    style      => "line",
    weight     => [2,1],
    color      => $GnuPlot::PrettyPlots::colorPairs{'redYellow'}
    );
&GnuPlot::PrettyPlots::Plot_Datasets($gnuPlot,\$plot);
close($gnuPlot);
&GnuPlot::LaTeX::GnuPlot2PDF($plotFileTeX);

exit 0;
