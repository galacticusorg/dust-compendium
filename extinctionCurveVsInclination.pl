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

# Get options.
my %options = (
    attenuationFile    => "data/attenuations.hdf5",
    plotFile           => "plot.pdf",
    opticalDepthIndex  =>  1,
    spheroidScaleIndex => -1
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

# Convert attenuation to extinction.
my $extinction = -2.5*log10($attenuation);

# Determine ranges to plot.
my $xMinimum = $wavelength->minimum();
my $xMaximum = $wavelength->maximum();
my $yMinimum = $extinction->(($options{'opticalDepthIndex'}),:,:)->flat()->minimum();
my $yMaximum = $extinction->(($options{'opticalDepthIndex'}),:,:)->flat()->maximum();

# Make the plot.
my $plot;
my $gnuPlot;
(my $plotFileTeX = $options{'plotFile'}) =~ s/\.pdf$/.tex/;
open($gnuPlot,"|gnuplot");
print $gnuPlot "set terminal cairolatex pdf standalone color lw 2\n";
print $gnuPlot "set output '".$plotFileTeX."'\n";
print $gnuPlot "set title offset 0,-1  'Extinction at face-on \$\\tau_\\mathrm{V,0}=".sprintf("%6.2f",$opticalDepth->(($options{'opticalDepthIndex'}))).($options{'spheroidScaleIndex'} < 0 ? "" : " and \$r_\\mathrm{s}/r_\\mathrm{d}=".sprintf("%6.2f",$spheroidRadius->(($options{'spheroidScaleIndex'})))."\$")."\$'\n";
print $gnuPlot "set xlabel 'Wavelength; \$\\lambda\\, [\\mu\\mathrm{m}]\$'\n";
print $gnuPlot "set ylabel 'Extinction; \$A_\\lambda\$'\n";
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
for(my $i=0;$i<nelem($inclination);++$i) {
    my $fraction = (nelem($inclination)-1-$i)/(nelem($inclination)-1);
    my $color = &GnuPlot::PrettyPlots::Color_Gradient($fraction,[0.0,1.0,0.5],[240.0,1.0,0.5]);
    &GnuPlot::PrettyPlots::Prepare_Dataset(
	\$plot,
	$wavelength,
	$extinction->(($options{'opticalDepthIndex'}),($i),:),
	style      => "line",
	weight     => [2,1],
	color      => [$color,$color]
	);
}
&GnuPlot::PrettyPlots::Plot_Datasets($gnuPlot,\$plot);
close($gnuPlot);
&GnuPlot::LaTeX::GnuPlot2PDF($plotFileTeX);

exit 0;
