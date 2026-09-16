#!/usr/bin/env perl
use strict;
use warnings;
use Cwd;
use lib $ENV{'GALACTICUS_EXEC_PATH'         }."/perl";
use lib $ENV{'GALACTICUS_ANALYSIS_PERL_PATH'}."/perl";
use PDL;
use PDL::NiceSlice;
use PDL::IO::HDF5;
use PDL::Fit::Polynomial;
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
    showAtlas          =>  0,
    showExtrapolation  =>  1
    );
&Galacticus::Options::Parse_Options(\@ARGV,\%options);
# Load the attenuation data.
my $attenuationFile = new PDL::IO::HDF5($options{'attenuationFile'});
my $inclination     = $attenuationFile->dataset('inclination'        )->get();
my $wavelength      = $attenuationFile->dataset('wavelength'         )->get();
my $opticalDepth    = $attenuationFile->dataset('opticalDepth'       )->get();
my $spheroidRadius  = $attenuationFile->dataset('spheroidScaleRadial')->get();
my $attenuation;
my $attenuationUncertainty;
my $extrapolationCoefficients;
if ( $options{'spheroidScaleIndex'} < 0 ) {
    # Read disk attenuations.
    $attenuation                 = $attenuationFile->dataset('attenuationDisk'               )->get();
    $attenuationUncertainty      = $attenuationFile->dataset('attenuationUncertaintyDisk'    )->get();
    $extrapolationCoefficients   = $attenuationFile->dataset('extrapolationCoefficientsDisk' )->get();
} else {
    # Read spheroid attenuations.
    my $attenuations             = $attenuationFile->dataset('attenuationSpheroid'              )->get();
    my $attenuationUncertainties = $attenuationFile->dataset('attenuationUncertaintySpheroid'   )->get();    
    my $extrapolation            = $attenuationFile->dataset('extrapolationCoefficientsSpheroid')->get();
    $attenuation                 = $attenuations            ->(($options{'spheroidScaleIndex'}),:,:,:);
    $attenuationUncertainty      = $attenuationUncertainties->(($options{'spheroidScaleIndex'}),:,:,:);
    $extrapolationCoefficients   = $extrapolation           ->(($options{'spheroidScaleIndex'}),:,:,:);
}

# Load the Ferrara et al. dust atlas.
my $dustAtlas;
&Galacticus::DustAttenuation::Load_Dust_Atlas($dustAtlas);

# Convert attenuation to extinction.
my $extinction = -2.5*log10($attenuation);

# Extract extinction at V-band (0.55um).
my $wavelengthIndices           = pdl sequence(nelem($wavelength));
my $wavelengthReversed          = $wavelength       ->(-1:0)->copy();
my $wavelengthIndicesReversed   = $wavelengthIndices->(-1:0)->copy();
my $wavelengthV                 = pdl 0.55;
(my $wavelengthIndex)           = interpolate($wavelengthV,$wavelengthReversed,$wavelengthIndicesReversed);
my $indices                     = pdl zeros(3,nelem($opticalDepth));
$indices->((0),:)              .= sequence(nelem($opticalDepth));
$indices->((1),:)              .= $options{'inclinationIndex'};
$indices->((2),:)              .= $wavelengthIndex;
my $extinctionV                 = $extinction->interpND($indices);
my $coefficients                = pdl zeroes(2);
my $indicesExtrapolation        = pdl zeros(2,nelem($opticalDepth));
$indicesExtrapolation->((0),:) .= $options{'inclinationIndex'};
$indicesExtrapolation->((1),:) .= $wavelengthIndex;
$coefficients->((0))          .= $extrapolationCoefficients->(:,:,(0))->interpND($indicesExtrapolation);
$coefficients->((1))          .= $extrapolationCoefficients->(:,:,(1))->interpND($indicesExtrapolation);

# Interpolate in the Ferrara et al. dust atlas.
my $wavelengthIndicesFerrara   = pdl sequence(nelem($Galacticus::DustAttenuation::wavelengths));
(my $wavelengthIndexFerrara)   = interpolate($wavelengthV*1.0e4,$Galacticus::DustAttenuation::wavelengths,$wavelengthIndicesFerrara);
my $inclinationIndicesFerrara  = pdl sequence(nelem($Galacticus::DustAttenuation::diskInclinations));
(my $inclinationIndexFerrara)  = interpolate($inclination->(($options{'inclinationIndex'})),$Galacticus::DustAttenuation::diskInclinations,$inclinationIndicesFerrara);
my $sizeIndicesFerrara  = pdl sequence(nelem($Galacticus::DustAttenuation::spheroidSizes));
my $sizeFerrara         = $spheroidRadius->(($options{'spheroidScaleIndex'}))/1.16;
(my $sizeIndexFerrara)  = interpolate($sizeFerrara,$Galacticus::DustAttenuation::spheroidSizes,$sizeIndicesFerrara);
my $indicesFerrara;
my $attenuationVFerrara;
my $opticalDepthsFerrara;
if ( $options{'spheroidScaleIndex'} < 0 ) {
    $opticalDepthsFerrara     = $Galacticus::DustAttenuation::diskOpticalDepths;
    $indicesFerrara           = pdl zeros(3,nelem($Galacticus::DustAttenuation::diskOpticalDepths));
    $indicesFerrara->((0),:) .= $inclinationIndexFerrara;
    $indicesFerrara->((1),:) .= sequence(nelem($Galacticus::DustAttenuation::diskOpticalDepths));
    $indicesFerrara->((2),:) .= $wavelengthIndexFerrara; 
    $attenuationVFerrara      = $Galacticus::DustAttenuation::diskAttenuations    ->interpND($indicesFerrara);
} else {
    $opticalDepthsFerrara     = $Galacticus::DustAttenuation::spheroidOpticalDepths;
    $indicesFerrara           = pdl zeros(4,nelem($Galacticus::DustAttenuation::spheroidOpticalDepths));
    $indicesFerrara->((0),:) .= $sizeIndexFerrara;
    $indicesFerrara->((1),:) .= $inclinationIndexFerrara;
    $indicesFerrara->((2),:) .= sequence(nelem($Galacticus::DustAttenuation::spheroidOpticalDepths));
    $indicesFerrara->((3),:) .= $wavelengthIndexFerrara; 
    $attenuationVFerrara      = $Galacticus::DustAttenuation::spheroidAttenuations->interpND($indicesFerrara);
}
my $extinctionVFerrara         = -2.5*log10($attenuationVFerrara);

# Determine ranges to plot.
my $xMinimum = $opticalDepth->(1:-1)->minimum()*0.9;
my $xMaximum = $opticalDepth->(1:-1)->maximum()*1.1;
my $extinctionVRange = $extinctionV;
$extinctionVRange = $extinctionVRange->append($extinctionVFerrara)
    if ( $options{'showAtlas'} ); 
my $yMinimum = $extinctionVRange->minimum();
my $yMaximum = $extinctionVRange->maximum()*1.1;
$yMinimum .= -0.05
    if ( $yMinimum > -0.05 );

# Find an extrapolation.
my $extrapolation = -2.5*($coefficients->((0))+$coefficients->((1))*$opticalDepth->log())/log(10.0);

# Make the plot.
my $plot;
my $gnuPlot;
(my $plotFileTeX = $options{'plotFile'}) =~ s/\.pdf$/.tex/;
open($gnuPlot,"|gnuplot");
print $gnuPlot "set terminal cairolatex pdf standalone color lw 2\n";
print $gnuPlot "set output '".$plotFileTeX."'\n";
print $gnuPlot "set title offset 0,-1  'Extinction at inclination \$i=".sprintf("%4.1f",$inclination->(($options{'inclinationIndex'})))."^\\circ\$".($options{'spheroidScaleIndex'} < 0 ? "" : " and \$r_\\mathrm{s}/r_\\mathrm{d}=".sprintf("%7.3f",$spheroidRadius->(($options{'spheroidScaleIndex'})))."\$")."'\n";
print $gnuPlot "set xlabel 'Optical depth; \$\\tau_\\mathrm{V,0}\$'\n";
print $gnuPlot "set ylabel 'Extinction; \$A_\\mathrm{V}\$'\n";
print $gnuPlot "set lmargin screen 0.15\n";
print $gnuPlot "set rmargin screen 0.95\n";
print $gnuPlot "set bmargin screen 0.15\n";
print $gnuPlot "set tmargin screen 0.95\n";
print $gnuPlot "set logscale x\n";
print $gnuPlot "set mxtics 10\n";
print $gnuPlot "set format x '\$10^{\%L}\$'\n";
print $gnuPlot "set key spacing 1.2\n";
print $gnuPlot "set key at screen 0.25,0.40\n";
print $gnuPlot "set key left\n";
print $gnuPlot "set key bottom\n";
print $gnuPlot "set xrange [".$xMinimum.":".$xMaximum."]\n";
print $gnuPlot "set yrange [".$yMinimum.":".$yMaximum."]\n";
print $gnuPlot "set pointsize 1.0\n";
&GnuPlot::PrettyPlots::Prepare_Dataset(
    \$plot,
    $opticalDepthsFerrara,
    $extinctionVFerrara,
    style      => "point",
    symbol     => [6,7],
    weight     => [2,1],
    pointSize  => 0.25,
    color      => $GnuPlot::PrettyPlots::colorPairs{'cornflowerBlue'}
    )
    if ( $options{'showAtlas'} );
&GnuPlot::PrettyPlots::Prepare_Dataset(
    \$plot,
    $opticalDepth,
    $extinctionV,
    style      => "line",
    weight     => [2,1],
    color      => $GnuPlot::PrettyPlots::colorPairs{'redYellow'}
    );
&GnuPlot::PrettyPlots::Prepare_Dataset(
	\$plot,
	$opticalDepth,
	$extrapolation,
	style      => "line",
	weight     => [2,1],
	transparency => 0.7,
	color      => $GnuPlot::PrettyPlots::colorPairs{'mediumSeaGreen'}
    )
    if ( $options{'showExtrapolation'} );
&GnuPlot::PrettyPlots::Plot_Datasets($gnuPlot,\$plot);
close($gnuPlot);
&GnuPlot::LaTeX::GnuPlot2PDF($plotFileTeX);

exit 0;
