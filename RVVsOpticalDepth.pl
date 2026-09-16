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
    inclinationIndex   =>  1,
    spheroidScaleIndex => -1,
    showAtlas          =>  0
    );
&Galacticus::Options::Parse_Options(\@ARGV,\%options);
# Load the attenuation data.
my $attenuationFile = new PDL::IO::HDF5($options{'attenuationFile'});
my $inclination     = $attenuationFile->dataset('inclination' )->get();
my $wavelength      = $attenuationFile->dataset('wavelength'  )->get();
my $opticalDepth    = $attenuationFile->dataset('opticalDepth')->get();
my $attenuation;
my $attenuationUncertainty;
if ( $options{'spheroidScaleIndex'} < 0 ) {
    # Read disk attenuations.
    $attenuation                 = $attenuationFile->dataset('attenuationDisk'               )->get();
    $attenuationUncertainty      = $attenuationFile->dataset('attenuationUncertaintyDisk'    )->get();
} else {
    # Read spheroid attenuations.
    my $attenuations             = $attenuationFile->dataset('attenuationSpheroid'           )->get();
    my $attenuationUncertainties = $attenuationFile->dataset('attenuationUncertaintySpheroid')->get();    
    $attenuation                 = $attenuations            ->(($options{'spheroidScaleIndex'}),:,:,:);
    $attenuationUncertainty      = $attenuationUncertainties->(($options{'spheroidScaleIndex'}),:,:,:);
}

# Load the Ferrara et al. dust atlas.
my $dustAtlas;
&Galacticus::DustAttenuation::Load_Dust_Atlas($dustAtlas);

# Convert attenuation to extinction.
my $extinction = -2.5*log10($attenuation);

# Extract extinction at V-band (0.55um).
my $wavelengthIndices          = pdl sequence(nelem($wavelength));
my $wavelengthReversed         = $wavelength       ->(-1:0)->copy();
my $wavelengthIndicesReversed  = $wavelengthIndices->(-1:0)->copy();
my $wavelengthV                = pdl 0.55;
my $wavelengthB                = pdl 0.44;
(my $wavelengthIndexV)         = interpolate($wavelengthV,$wavelengthReversed,$wavelengthIndicesReversed);
(my $wavelengthIndexB)         = interpolate($wavelengthB,$wavelengthReversed,$wavelengthIndicesReversed);
my $indices                    = pdl zeros(3,nelem($opticalDepth));
$indices->((0),:)             .= sequence(nelem($opticalDepth));
$indices->((1),:)             .= $options{'inclinationIndex'};
$indices->((2),:)             .= $wavelengthIndexV;
my $extinctionV                = $extinction->interpND($indices);
$indices->((2),:)             .= $wavelengthIndexB;
my $extinctionB                = $extinction->interpND($indices);
my $RV                         = $extinctionV/($extinctionB-$extinctionV);
my $determinable               = which($extinctionV != 0.0);

# Interpolate in the Ferrara et al. dust atlas.
my $wavelengthIndicesFerrara    = pdl sequence(nelem($Galacticus::DustAttenuation::wavelengths));
(my $wavelengthVIndexFerrara)    = interpolate($wavelengthV*1.0e4,$Galacticus::DustAttenuation::wavelengths,$wavelengthIndicesFerrara);
(my $wavelengthBIndexFerrara)    = interpolate($wavelengthB*1.0e4,$Galacticus::DustAttenuation::wavelengths,$wavelengthIndicesFerrara);
my $inclinationIndicesFerrara  = pdl sequence(nelem($Galacticus::DustAttenuation::diskOpticalDepths));
(my $inclinationIndexFerrara)  = interpolate($inclination->(($options{'inclinationIndex'})),$Galacticus::DustAttenuation::diskOpticalDepths,$inclinationIndicesFerrara);
my $indicesFerrara              = pdl zeros(3,nelem($Galacticus::DustAttenuation::diskOpticalDepths));
$indicesFerrara->((0),:)       .= $inclinationIndexFerrara;
$indicesFerrara->((1),:)       .= sequence(nelem($Galacticus::DustAttenuation::diskOpticalDepths));
$indicesFerrara->((2),:)       .= $wavelengthVIndexFerrara;
my $attenuationVFerrara         = $Galacticus::DustAttenuation::diskAttenuations->interpND($indicesFerrara);
my $extinctionVFerrara          = -2.5*log10($attenuationVFerrara);
$indicesFerrara->((2),:)       .= $wavelengthBIndexFerrara;
my $attenuationBFerrara         = $Galacticus::DustAttenuation::diskAttenuations->interpND($indicesFerrara);
my $extinctionBFerrara          = -2.5*log10($attenuationBFerrara);
my $RVFerrara                   = $extinctionVFerrara/($extinctionBFerrara-$extinctionVFerrara);

# Determine ranges to plot.
my $xMinimum = $opticalDepth->(1:-1)->minimum();
my $xMaximum = $opticalDepth->(1:-1)->maximum();
my $yMinimum = $RV          ->($determinable)->minimum();
my $yMaximum = $RV          ->($determinable)->maximum();

# Make the plot.
my $plot;
my $gnuPlot;
(my $plotFileTeX = $options{'plotFile'}) =~ s/\.pdf$/.tex/;
open($gnuPlot,"|gnuplot");
print $gnuPlot "set terminal cairolatex pdf standalone color lw 2\n";
print $gnuPlot "set output '".$plotFileTeX."'\n";
print $gnuPlot "set title offset 0,-1 'Reddening at inclination \$i=".sprintf("%4.1f",$inclination->(($options{'inclinationIndex'})))."^\\circ\$'\n";
print $gnuPlot "set xlabel 'Optical depth; \$\\tau_\\mathrm{V,0}\$'\n";
print $gnuPlot "set ylabel 'Reddening; \$R_\\mathrm{V}\$'\n";
print $gnuPlot "set lmargin screen 0.15\n";
print $gnuPlot "set rmargin screen 0.95\n";
print $gnuPlot "set bmargin screen 0.15\n";
print $gnuPlot "set tmargin screen 0.95\n";
print $gnuPlot "set key spacing 1.2\n";
print $gnuPlot "set key at screen 0.25,0.40\n";
print $gnuPlot "set key left\n";
print $gnuPlot "set key bottom\n";
print $gnuPlot "set logscale x\n";
print $gnuPlot "set mxtics 10\n";
print $gnuPlot "set format x '\$10^{\%L}\$'\n";
print $gnuPlot "set xrange [".$xMinimum.":".$xMaximum."]\n";
print $gnuPlot "set yrange [".$yMinimum.":".$yMaximum."]\n";
print $gnuPlot "set pointsize 1.0\n";
&GnuPlot::PrettyPlots::Prepare_Dataset(
    \$plot,
    $Galacticus::DustAttenuation::diskOpticalDepths,
    $RVFerrara,
    style      => "point",
    symbol     => [6,7],
    weight     => [2,1],
    pointSize  => 0.25,
    color      => $GnuPlot::PrettyPlots::colorPairs{'cornflowerBlue'}
    );
&GnuPlot::PrettyPlots::Prepare_Dataset(
    \$plot,
    $opticalDepth->($determinable),
    $RV          ->($determinable),
    style      => "line",
    weight     => [2,1],
    color      => $GnuPlot::PrettyPlots::colorPairs{'redYellow'}
    );
&GnuPlot::PrettyPlots::Plot_Datasets($gnuPlot,\$plot);
close($gnuPlot);
&GnuPlot::LaTeX::GnuPlot2PDF($plotFileTeX);

exit 0;
