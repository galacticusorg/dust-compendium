#!/usr/bin/env perl
use strict;
use warnings;
use lib $ENV{'GALACTICUS_EXEC_PATH'}."/perl";
use Cwd;
use PDL;
use PDL::NiceSlice;
use PDL::IO::HDF5;
use PDL::Fit::Polynomial;
use Galacticus::Launch::PBS;
use Galacticus::Options;
use DateTime;
use Clone 'clone';
use Data::Dumper;

# Construct and run Hyperion models to compute radiative transfer through simple galaxy geometries.
# Andrew Benson (16-July-2018)

# Read options.
my %options = (
    forceRun   => "no"
    );
&Galacticus::Options::Parse_Options(\@ARGV,\%options);

# Describe models to run.
my @models;

# Base description of the Compendium models.
my $compendiumBaseModel =
     {
     	 diskStructureVertical => "sechSquared"                                                    ,
     	 diskScaleVertical     => 1.37e-1                                                          , # Vertical scale height (Kregel et al., 2002; MNRAS; 334; 646).
     	 dustScaleVertical     => 1.37e-1                                                          ,
     	 diskCutOff            => 10.0                                                             ,
     	 spheroidStructure     => "hernquist"                                                      ,
     	 # spheroidScaleRadial   => [ 0.0010,   0.0013,   0.0016,   0.0020,   0.0025,   0.0032,   0.0040,   0.0050,   0.0063,   0.0079,   0.0100,   0.0126,   0.0158,   0.0200,   0.0251,   0.0316,   0.0398,   0.0501,   0.0631,   0.0794,   0.1000,   0.1259,   0.1585,   0.1995,   0.2512,   0.3162,   0.3981,   0.5012,   0.6310,   0.7943,   1.0000,   1.2589,   1.5849,   1.9953,   2.5119,   3.1623,   3.9811,   5.0119,   6.3096,   7.9433,  10.0000,  12.5893,  15.8489,  19.9526,  25.1189,  31.6228,  39.8107,  50.1187,  63.0957,  79.4328, 100.0000 ],
     	 spheroidScaleRadial   => [  1.0000 ],
     	 spheroidCutOff        => 10.0
     };

# Dust models for Compendium.
my @dustModels =
    (
     # Draine dust with R_V=3.1.
     {
	 label       => "dustD03Rv3.1"                                                                                              ,
	 fileName    => "hyperion-dust-0.1.0/dust_files/d03_3.1_6.0_A.hdf5"                                                         ,
	 description => "Weingartner & Draine (2001; ApJ; 548; 296; http://adsabs.harvard.edu/abs/2001ApJ...548..296W) with R_V=3.1"
     },
     # # Draine dust with R_V=4.0.
     # {
     # 	 label       => "dustD03Rv4.0"                                                                                              ,
     # 	 fileName    => "hyperion-dust-0.1.0/dust_files/d03_4.0_4.0_A.hdf5"                                                         ,
     # 	 description => "Weingartner & Draine (2001; ApJ; 548; 296; http://adsabs.harvard.edu/abs/2001ApJ...548..296W) with R_V=4.0"
     # },
     # # Draine dust with R_V=5.5.
     # {
     # 	 label       => "dustD03Rv5.5"                                                                                              ,
     # 	 fileName    => "hyperion-dust-0.1.0/dust_files/d03_5.5_3.0_A.hdf5"                                                         ,
     # 	 description => "Weingartner & Draine (2001; ApJ; 548; 296; http://adsabs.harvard.edu/abs/2001ApJ...548..296W) with R_V=5.5"
     # },
     # # Kim, Martin, and Hendry (1994) dust with Henyey-Greenstein scattering
     # {
     # 	 label       => "dustKMH94HGRv3.1"                                                                                                                              ,
     # 	 fileName    => "hyperion-dust-0.1.0/dust_files/kmh94_3.1_hg.hdf5"                                                                                              ,
     # 	 description => "Kim, Martin, and Hendry (1994; ApJ; 422; 164; http://adsabs.harvard.edu/abs/1994ApJ...422..164K) with R_V=3.1 and Henyey-Greenstein scattering"
     # },
     # # Kim, Martin, and Hendry (1994) dust with full scattering
     # {
     # 	 label       => "dustKMH94FullRv3.1"                                                                                                                            ,
     # 	 fileName    => "hyperion-dust-0.1.0/dust_files/kmh94_3.1_full.hdf5"                                                                                            ,
     # 	 description => "Kim, Martin, and Hendry (1994; ApJ; 422; 164; http://adsabs.harvard.edu/abs/1994ApJ...422..164K) with R_V=3.1 and full scattering"
     # },
    );

# Generate variations on the Compendium model.
## Iterate over dust models.
foreach my $dustModel ( @dustModels ) {
    # Clone the base model.
    my $compendiumModel = clone($compendiumBaseModel);
    # Set attributes of the model.
    $compendiumModel->{'label'          } = "compendium:exp:sech:Hernquist:hd0.137:hz0.137:".$dustModel->{'label'      };
    $compendiumModel->{'dustFileName'   } =                                                  $dustModel->{'fileName'   };
    $compendiumModel->{'dustDescription'} =                                                  $dustModel->{'description'};
    # Push model into the queue.
    push(@models,$compendiumModel);
}

# # Base description of the Ferrara et al. (1999) models.
# my $ferraraBaseModel = 
# {
#     diskStructureVertical => "exponential"                                      ,
#     diskScaleVertical     => 8.75e-2                                            ,
#     diskCutOff            => 6.0                                                ,
#     spheroidStructure     => "jaffe"                                            ,
#     spheroidScaleRadial   => [ 0.116, 0.464, 1.856 ]                            , # Ferrara et al. use r_s=1.16Re, and they tabulated at Re=[0.1,0.4,1.6]
#     spheroidCutOff        => 5.8                                                , # Ferrara et al. use r_s=1.16Re, and they truncate at 5Re
#     inclinations          => [ 9.3, 22.9, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0 ],
#     diskOpticalDepths     => [ 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0 ],
#     spheroidOpticalDepths => [ 0.0 ],
#     wavelengths           => [ 0.1250, 0.1515, 0.1775, 0.1995, 0.2215, 0.2480, 0.2895, 0.3605, 0.4413, 0.5512, 0.6594, 0.8059, 1.2369, 2.1578 ]
# };

# # Generate variations on the Ferrara et al. (1999) models.
# ## Iterate over dust scale heights.
# foreach my $scaleHeightDustRelative ( 0.4, 1.0, 2.5 ) {
#     ## Iterate over dust models.
#     foreach my $dustModel ( "MW", "SMC" ) {
# 	## Iterate over grid resolution.
# 	foreach my $resolution ( "original", "highRes" ) {
# 	# Clone the base model.
# 	my $ferraraModel = clone($ferraraBaseModel);
# 	# Set attributes of the model.
# 	my $scaleHeightDustRelativeLabel = sprintf("%3.1f",$scaleHeightDustRelative);
# 	$ferraraModel->{'label'            } = "Ferrara1999_".$dustModel."_hz".$scaleHeightDustRelativeLabel;
# 	$ferraraModel->{'dustScaleVertical'} = $scaleHeightDustRelative*$ferraraModel->{'diskScaleVertical'};
# 	$ferraraModel->{'dustFileName'     } = "data/dustFerrara_".$dustModel.".hdf5";
# 	$ferraraModel->{'dustDescription'  } = "Dust matched to Ferrara et al. (1999; ApJ; 123; 437; http://adsabs.harvard.edu/abs/1999ApJS..123..437F) based on ".$dustModel." model of Gordon et al. (1997; ApJ; 487; 625; http://adsabs.harvard.edu/abs/1997ApJ...487..625G), using Henyey-Greenstein scattering.";
# 	# For high-res models change the tabulation points.
# 	if ( $resolution eq "highRes" ) {
# 	    # Add a suffix to the model label.
# 	    $ferraraModel->{'label'} .= "_highRes";
# 	    # Set a wide range of spheroid radii.
# 	    @{$ferraraModel->{'spheroidScaleRadial'}} = ( 0.0010,   0.0013,   0.0016,   0.0020,   0.0025,   0.0032,   0.0040,   0.0050,   0.0063,   0.0079,   0.0100,   0.0126,   0.0158,   0.0200,   0.0251,   0.0316,   0.0398,   0.0501,   0.0631,   0.0794,   0.1000,   0.1259,   0.1585,   0.1995,   0.2512,   0.3162,   0.3981,   0.5012,   0.6310,   0.7943,   1.0000,   1.2589,   1.5849,   1.9953,   2.5119,   3.1623,   3.9811,   5.0119,   6.3096,   7.9433,  10.0000,  12.5893,  15.8489,  19.9526,  25.1189,  31.6228,  39.8107,  50.1187,  63.0957,  79.4328, 100.0000 );
# 	    # Remove custom grid points so that the defaults will be used instead.
# 	    delete($ferraraModel->{$_})
# 		foreach ( "inclinations", "diskOpticalDepths", "spheroidOpticalDepths", "wavelengths" );
# 	}
# 	# Push model into the queue.
# 	push(@models,$ferraraModel);
# 	}
#     }
# }

# # Generate variations on the Ferrara et al. (1999) models matched to GRASIL assumptions.
# ## Iterate over dust scale heights.
# foreach my $scaleHeightDustRelative ( 1.0 ) {
#     ## Iterate over dust models.
#     foreach my $dustModel ( @dustModels ) {
# 	next
# 	    unless (
# 		$dustModel->{'label'} eq "dustD03Rv3.1"
# 		||
# 		$dustModel->{'label'} eq "dustD03Rv5.5"
# 	    );
# 	## Iterate over scale-heights.
# 	foreach my $scaleHeight ( 0.1, 0.5 ) {
# 	    # Clone the base model.
# 	    my $ferraraModel = clone($ferraraBaseModel);
# 	    # Set attributes of the model.
# 	    my $scaleHeightLabel             = sprintf("%3.1f",$scaleHeight            );
# 	    my $scaleHeightDustRelativeLabel = sprintf("%3.1f",$scaleHeightDustRelative);
# 	    $ferraraModel->{'label'            } = "Ferrara1999_GRASIL_".$dustModel->{'label'}."_hzStars".$scaleHeightLabel."_hzDust".$scaleHeightDustRelativeLabel;
# 	    $ferraraModel->{'diskScaleVertical'} = $scaleHeight;
# 	    $ferraraModel->{'dustScaleVertical'} = $scaleHeightDustRelative*$ferraraModel->{'diskScaleVertical'};
# 	    $ferraraModel->{'dustFileName'     } = $dustModel->{'fileName'   };
# 	    $ferraraModel->{'dustDescription'  } = $dustModel->{'description'};
# 	    # Add a suffix to the model label.
# 	    $ferraraModel->{'label'} .= "_highRes";
# 	    # Set a wide range of spheroid radii.
# 	    @{$ferraraModel->{'spheroidScaleRadial'}} = ( 0.0010,   0.0013,   0.0016,   0.0020,   0.0025,   0.0032,   0.0040,   0.0050,   0.0063,   0.0079,   0.0100,   0.0126,   0.0158,   0.0200,   0.0251,   0.0316,   0.0398,   0.0501,   0.0631,   0.0794,   0.1000,   0.1259,   0.1585,   0.1995,   0.2512,   0.3162,   0.3981,   0.5012,   0.6310,   0.7943,   1.0000,   1.2589,   1.5849,   1.9953,   2.5119,   3.1623,   3.9811,   5.0119,   6.3096,   7.9433,  10.0000,  12.5893,  15.8489,  19.9526,  25.1189,  31.6228,  39.8107,  50.1187,  63.0957,  79.4328, 100.0000 );
# 	    # Remove custom grid points so that the defaults will be used instead.
# 	    delete($ferraraModel->{$_})
# 		foreach ( "inclinations", "diskOpticalDepths", "spheroidOpticalDepths", "wavelengths" );
# 	    # Push model into the queue.
# 	    push(@models,$ferraraModel);
# 	}
#     }
# }

# Set initial random number seed.
my $seed = -653;

# Iterate over models.
foreach my $model ( @models ) {

    # Construct output file name.
    my $outputFileName = "data/".$model->{'label'}.":AttenuationsNew.hdf5";

    # If output file exists, skip this model.
    if ( $options{'forceRun'} ne "yes" && -e $outputFileName ) {
	print "Skipping model ".$model->{'label'}." - already exists...\n";
	next;
    } else {
	print "Running model ".$model->{'label'}."...\n";
    }
    
    # Specify disk structure.
    my $diskStructureVertical = $model->{'diskStructureVertical'};
    
    # Specify spheroid structure.
    my $spheroidStructure     = $model->{'spheroidStructure'    };
    
    # Specify the disk dimensions and cut off.
    my $diskScaleRadial       = 1.0e-3; # Radial scale length in Mpc - results should be independent of this choice.
    my $diskScaleVertical     = $model->{'diskScaleVertical'}*$diskScaleRadial;
    my $dustScaleVertical     = $model->{'dustScaleVertical'}*$diskScaleRadial;
    my $diskCutOff            = $model->{'diskCutOff'};

    # Specify number of inclination angles (will be distributed linearly between 0 and 90 degrees).
    my $inclinationDefinition = 1;
    my $inclinationCount      = $inclinationDefinition;
    if ( exists($model->{'inclinations'}) ) {
	$inclinationDefinition = join  (":",@{$model->{'inclinations'}});
	$inclinationCount      = scalar(    @{$model->{'inclinations'}});
    }

    # Specify wavelength range and count.
    my $wavelengthMinimum     =   0.01;
    my $wavelengthMaximum     =   3.00;
    my $wavelengthDefinition  = 250   ;
    my $wavelengthCount       = $wavelengthDefinition;
    if ( exists($model->{'wavelengths'}) ) {
	$wavelengthMinimum    = -1.0;
	$wavelengthMaximum    = -1.0;
	$wavelengthDefinition = join  (":",@{$model->{'wavelengths'}});
	$wavelengthCount      = scalar(    @{$model->{'wavelengths'}});
    }

    # Construct a set of disk optical depths at which to generate models. This is the optical depth through the center of the
    # galaxy from -infinity to +infinity, when viewed face-on.
    my $diskOpticalDepths = pdl [ 0.0 ];
    my $diskOpticalDepthCount = 1;
    # my $diskOpticalDepthMinimum = pdl     0.01;
    # my $diskOpticalDepthMaximum = pdl 10000.00;
    # my $diskOpticalDepthCount   =        60   ;
    # my $diskOpticalDepths;
    # if ( exists($model->{'diskOpticalDepths'}) ) {
    # 	$diskOpticalDepths      = pdl @{$model->{'diskOpticalDepths'}};
    # 	$diskOpticalDepths      = $diskOpticalDepths->append(0.0) 
    # 	    unless ( any($diskOpticalDepths == 0.0) );
    # 	$diskOpticalDepths     .= $diskOpticalDepths->qsort();
    # 	$diskOpticalDepthCount  = nelem($diskOpticalDepths);
    # } else {
    # 	$diskOpticalDepths      = pdl sequence($diskOpticalDepthCount-1)*log($diskOpticalDepthMaximum/$diskOpticalDepthMinimum)/($diskOpticalDepthCount-2)+log($diskOpticalDepthMinimum);
    # 	$diskOpticalDepths     .= exp($diskOpticalDepths);
    # 	$diskOpticalDepths      = $diskOpticalDepths->append(0.0);
    # 	$diskOpticalDepths     .= $diskOpticalDepths->qsort();
    # }

    # Construct a set of spheroid optical depths at which to generate models. This is the optical depth through the galaxy from
    # -infinity to +infinity at the scale radius.
    my $spheroidOpticalDepthMinimum = pdl     0.01;
    my $spheroidOpticalDepthMaximum = pdl 10000.00;
    my $spheroidOpticalDepthCount   =         6   ;
    my $spheroidOpticalDepths;
    if ( exists($model->{'spheroidOpticalDepths'}) ) {
	$spheroidOpticalDepths      = pdl @{$model->{'spheroidOpticalDepths'}};
	$spheroidOpticalDepths      = $spheroidOpticalDepths->append(0.0) 
	    unless ( any($spheroidOpticalDepths == 0.0) );
	$spheroidOpticalDepths     .= $spheroidOpticalDepths->qsort();
	$spheroidOpticalDepthCount  = nelem($spheroidOpticalDepths);
    } else {
	$spheroidOpticalDepths      = pdl sequence($spheroidOpticalDepthCount-1)*log($spheroidOpticalDepthMaximum/$spheroidOpticalDepthMinimum)/($spheroidOpticalDepthCount-2)+log($spheroidOpticalDepthMinimum);
	$spheroidOpticalDepths     .= exp($spheroidOpticalDepths);
	$spheroidOpticalDepths      = $spheroidOpticalDepths->append(0.0);
	$spheroidOpticalDepths     .= $spheroidOpticalDepths->qsort();
    }
    
    # Specify the dust file.
    my $dustFileName          = $model->{'dustFileName'   };
    my $dustDescription       = $model->{'dustDescription'};
    
    # Decrement the random number seed.
    --$seed;
    
    # Initialize array which will store the full dataset.
    my $attenuations;
    my $attenuationsUncertainty;
    $attenuations           ->{'disk'    } = pdl zeros(                                           $diskOpticalDepthCount,$spheroidOpticalDepthCount,$inclinationCount,$wavelengthCount);
    $attenuations           ->{'spheroid'} = pdl zeros(scalar(@{$model->{'spheroidScaleRadial'}}),$diskOpticalDepthCount,$spheroidOpticalDepthCount,$inclinationCount,$wavelengthCount);
    $attenuationsUncertainty->{'disk'    } = pdl zeros(                                           $diskOpticalDepthCount,$spheroidOpticalDepthCount,$inclinationCount,$wavelengthCount);
    $attenuationsUncertainty->{'spheroid'} = pdl zeros(scalar(@{$model->{'spheroidScaleRadial'}}),$diskOpticalDepthCount,$spheroidOpticalDepthCount,$inclinationCount,$wavelengthCount);

    # Set stellar components for which to compute the attenuation.
    # my @stellarComponents = ( "disk" );
    # push(@stellarComponents,"spheroid")
    # 	if ( exists($model->{'spheroidScaleRadial'}) );
    my @stellarComponents = ( "spheroid" );
    
    # Stack to be used for PBS jobs.
    my @jobStack;

    # Set options for PBS launch.
    my %pbsOptions =
	(
	 pbsJobMaximum       =>   4,
	 limitQueuedOnly     =>   1,
	 submitSleepDuration =>   1,
	 waitSleepDuration   =>  10
	);

    # Begin generating models.
    print "Generating dust and source distributions...\n";
    # Iterate over optical depths.
    for(my $iOpticalDepth=0;$iOpticalDepth<nelem($diskOpticalDepths);++$iOpticalDepth) {
	for(my $jOpticalDepth=0;$jOpticalDepth<nelem($spheroidOpticalDepths);++$jOpticalDepth) {
	    
	    # Iterate over components.
	    foreach my $stellarComponent ( @stellarComponents ) {
		
		# Specify the spheroid dimensions and cut off.
		my $spheroidScalesRadial;
		my $spheroidCutOff      ;
		if ( $stellarComponent eq "disk"          ) {
		    $spheroidScalesRadial  = pdl [ 0.0 ];
		    $spheroidCutOff        = 0.0;
		} elsif ( $stellarComponent eq "spheroid" ) {
		    $spheroidScalesRadial  = pdl $model->{'spheroidScaleRadial'};
		    $spheroidCutOff        =     $model->{'spheroidCutOff'     };
		    $spheroidScalesRadial *= $diskScaleRadial;
		} else {
		    die("unknown component");
		}

		# Iterate over spheroid scales.
		for(my $iSpheroidScaleRadial=0;$iSpheroidScaleRadial<nelem($spheroidScalesRadial);++$iSpheroidScaleRadial) {
		    my $spheroidScaleRadial = $spheroidScalesRadial->(($iSpheroidScaleRadial));
		    
		    # Construct file names.
		    my $inputFileName  = "data/".$model->{'label'}."Input_" .$stellarComponent.($stellarComponent eq "spheroid" ? "_".$iSpheroidScaleRadial : "")."_".$iOpticalDepth."_".$jOpticalDepth.".hdf5";
		    my $outputFileName = "data/".$model->{'label'}."Output_".$stellarComponent.($stellarComponent eq "spheroid" ? "_".$iSpheroidScaleRadial : "")."_".$iOpticalDepth."_".$jOpticalDepth.".hdf5";
		    
		    if ( -e $outputFileName && $options{'forceRun'} ne "yes" ) {
			&postProcess($stellarComponent,$iSpheroidScaleRadial,$iOpticalDepth,$jOpticalDepth,$outputFileName,$attenuations,$attenuationsUncertainty,$inclinationCount);
		    } else {
			my $rootName = $model->{'label'}."_".$stellarComponent.($stellarComponent eq "spheroid" ? "_".$iSpheroidScaleRadial : "")."_".$iOpticalDepth."_".$jOpticalDepth;
			$rootName =~ s/:/_/g;
			system(
			    "python hyperionBuildModel.py"            ." ".
			    $dustFileName                             ." ".
			    $diskStructureVertical                    ." ".
			    $diskScaleRadial                          ." ".
			    $diskScaleVertical                        ." ".
			    $dustScaleVertical                        ." ".
			    $diskCutOff                               ." ".
			    $spheroidStructure                        ." ".
			    $spheroidScaleRadial                      ." ".
			    $spheroidCutOff                           ." ".
			    $stellarComponent                         ." ".
			    $inclinationDefinition                    ." ".
			    $wavelengthMinimum                        ." ".
			    $wavelengthMaximum                        ." ".
			    $wavelengthDefinition                     ." ".
			    $diskOpticalDepths    ->(($iOpticalDepth))." ".
			    $spheroidOpticalDepths->(($jOpticalDepth))." ".
			    $seed                                     ." ".
			    $inputFileName
			    );
			my %hyperionJob =
			    (
			     launchFile   => "data/".$rootName.".pbs",
			     label        =>         $rootName,
			     logFile      => "data/".$rootName.".log",
			     command      => "cd ".cwd()."; rm -f ".$outputFileName."; mpirun -np 64 --mca btl ^openib --mca mpi_preconnect_mpi 1 -hostfile \$PBS_NODEFILE hyperion_cyl_mpi ".$inputFileName." ".$outputFileName,
			     nodes        => 4,
			     ppn          => 16,
			     onCompletion => 
			     {
				 function  => \&postProcess,
				 arguments => [ $stellarComponent, $iSpheroidScaleRadial, $iOpticalDepth, $jOpticalDepth, $outputFileName, $attenuations, $attenuationsUncertainty, $inclinationCount ]
			     }
			    );
			push(@jobStack,\%hyperionJob);
		    }

		}

	    }

	}
	
    }
    print "   ...done\n";
    
    # Submit jobs to the queue.
    if ( scalar(@jobStack) > 0 ) {
	print "Running Hyperion jobs...\n";
	&Galacticus::Launch::PBS::SubmitJobs(\%pbsOptions,@jobStack);
	print "   ...done\n";
    }

    # Extract opacity array.
    my $opacityText = `python hyperionExtractOpacity.py $dustFileName`;
    chomp($opacityText);
    my $opacity = pdl $opacityText;

    # Extract wavelengths array.
    my $wavelengthsFileName;
    if      ( -e "data/$model->{'label'}Output_disk_0_0.hdf5" ) {
	$wavelengthsFileName = "data/".$model->{'label'}."Output_disk_0_0.hdf5";
    } elsif ( -e "data/$model->{'label'}Output_spheroid_0_0_0.hdf5" ) {
	$wavelengthsFileName = "data/".$model->{'label'}."Output_spheroid_0_0_0.hdf5";
    } else {
	die("Unable to find a file from which to extract wavelengths");
    }
    my $wavelengthsJoined = `python hyperionExtractWavelengths.py $wavelengthsFileName`;
    my $wavelengths       = pdl split(" ",$wavelengthsJoined);

    # Construct the inclinations array.
    my $inclinations;
    if ( $inclinationCount == 1 ) {
	$inclinations = pdl [ 90.0 ];
    } else {
	$inclinations = pdl sequence($inclinationCount)/($inclinationCount-1)*90.0;
    }
    
    # Normalize by the unattenuated spectrum to get attenuations.
    my $unattenuatedDiskSED            = $attenuations           ->{'disk'}->((0),(0),:,:)->copy();
    my $unattenuatedDiskSEDUncertainty = $attenuationsUncertainty->{'disk'}->((0),(0),:,:)->copy();
    for(my $i=0;$i<nelem($diskOpticalDepths);++$i) {
	for(my $j=0;$j<nelem($spheroidOpticalDepths);++$j) {
	    $attenuationsUncertainty->{'disk'}->(($i),($j),:,:) /= $unattenuatedDiskSED;
	    $attenuations           ->{'disk'}->(($i),($j),:,:) /= $unattenuatedDiskSED;
	}
    }
    if ( exists($model->{'spheroidScaleRadial'}) ) {
	for(my $iSpheroidScaleRadial=0;$iSpheroidScaleRadial<scalar(@{$model->{'spheroidScaleRadial'}});++$iSpheroidScaleRadial) {
	    my $unattenuatedSpheroidSED = $attenuations->{'spheroid'}->(($iSpheroidScaleRadial),(0),(0),:,:)->copy();
	    for(my $i=0;$i<nelem($diskOpticalDepths);++$i) {
		for(my $j=0;$j<nelem($spheroidOpticalDepths);++$j) {
		    $attenuationsUncertainty->{'spheroid'}->(($iSpheroidScaleRadial),($i),($j),:,:) /= $unattenuatedSpheroidSED;
		    $attenuations           ->{'spheroid'}->(($iSpheroidScaleRadial),($i),($j),:,:) /= $unattenuatedSpheroidSED;
		}
	    }
	}
    }

    # Determine interpolation coefficients to high optical depth.
    my $coefficients;
    if ( $inclinationCount > 1 ) {
	print "Computing extrapolation coefficients....\n";
	my $opticalDepthRange = 10.0; # Factor below maximum tabulated optical depth to which extrapolation should be fit.
	my $fitSubset         = which($opticalDepthRange*$diskOpticalDepths >= $diskOpticalDepths->((-1)));
	$coefficients->{'disk'    } = pdl zeroes(                                           nelem($spheroidOpticalDepths),$inclinationCount,$wavelengthCount,2);
	$coefficients->{'spheroid'} = pdl zeroes(scalar(@{$model->{'spheroidScaleRadial'}}),nelem($spheroidOpticalDepths),$inclinationCount,$wavelengthCount,2);
	for(my $i=0;$i<$inclinationCount;++$i) {
	    print "  Inclination ".$i." of ".$inclinationCount."\n";
	    for(my $l=0;$l<nelem($spheroidOpticalDepths);++$l) {
		for(my $j=0;$j<$wavelengthCount;++$j) {
		    (my $opticalDepthFit, my $coefficientsFit) = fitpoly1d($diskOpticalDepths->($fitSubset)->log(),$attenuations->{'disk'}->($fitSubset,($l),($i),($j))->log(),2);
		    $coefficients->{'disk'}->(($l),($i),($j),:) .= $coefficientsFit;
		    for(my $k=0;$k<scalar(@{$model->{'spheroidScaleRadial'}});++$k) {
			(my $opticalDepthFit, my $coefficientsFit) = fitpoly1d($diskOpticalDepths->($fitSubset)->log(),$attenuations->{'spheroid'}->(($k),$fitSubset,($l),($i),($j))->log(),2);
			$coefficients->{'spheroid'}->(($k),($l),($i),($j),:) .= $coefficientsFit;
		    }
		}
	    }
	}
	print "   ...done\n";
    }
    
    # Construct the output file.
    my $outputFile = new PDL::IO::HDF5(">".$outputFileName);
    ## Store primary datasets.
    my $spheroidScalesRadial  = pdl $model->{'spheroidScaleRadial'};
    $outputFile->dataset('inclination'                      )->set($inclinations                         );
    $outputFile->dataset('wavelength'                       )->set($wavelengths                          );
    $outputFile->dataset('diskOpticalDepth'                 )->set($diskOpticalDepths                    );
    $outputFile->dataset('spheroidOpticalDepth'             )->set($spheroidOpticalDepths                );
    $outputFile->dataset('spheroidScaleRadial'              )->set($spheroidScalesRadial                 );
    $outputFile->dataset('attenuationDisk'                  )->set($attenuations           ->{'disk'    })
	if (                           grep {$_ eq "disk"    } @stellarComponents );
    $outputFile->dataset('attenuationUncertaintyDisk'       )->set($attenuationsUncertainty->{'disk'    })
	if (                           grep {$_ eq "disk"    } @stellarComponents );
    $outputFile->dataset('extrapolationCoefficientsDisk'    )->set($coefficients           ->{'disk'    })
	if ( defined($coefficients) && grep {$_ eq "disk"    } @stellarComponents );
    $outputFile->dataset('attenuationSpheroid'              )->set($attenuations           ->{'spheroid'})
	if (                           grep {$_ eq "spheroid"} @stellarComponents );
    $outputFile->dataset('attenuationUncertaintySpheroid'   )->set($attenuationsUncertainty->{'spheroid'})
	if (                           grep {$_ eq "spheroid"} @stellarComponents );
    $outputFile->dataset('extrapolationCoefficientsSpheroid')->set($coefficients           ->{'spheroid'})
	if ( defined($coefficients) && grep {$_ eq "spheroid"} @stellarComponents );
    # Add metadata.
    my $dt = DateTime->now->set_time_zone('local');
    (my $tz = $dt->format_cldr("ZZZ")) =~ s/(\d{2})(\d{2})/$1:$2/;
    my $now = $dt->ymd."T".$dt->hms.".".$dt->format_cldr("SSS").$tz;
    $diskScaleVertical /= $diskScaleRadial;
    $dustScaleVertical /= $diskScaleRadial;
    $outputFile->attrSet(timeStamp             => $now                      );
    $outputFile->attrSet(opacity               => $opacity                  );
    $outputFile->attrSet(diskStructureVertical => $diskStructureVertical    );
    $outputFile->attrSet(diskScaleVertical     => $diskScaleVertical        );
    $outputFile->attrSet(dustScaleVertical     => $dustScaleVertical        );
    $outputFile->attrSet(diskCutOff            => $diskCutOff               );
    $outputFile->attrSet(spheroidCutOff        => $model->{'spheroidCutOff'})
	if ( exists($model->{'spheroidScaleRadial'}) );
    $outputFile->attrSet(dustDescription       => $dustDescription          );
}

exit 0;

sub postProcess {
    # Callback function which handles postprocessing of the Hyperion models.
    my $stellarComponent               = shift();
    my $iSpheroidScaleRadial           = shift();
    my $iOpticalDepth                  = shift();
    my $jOpticalDepth                  = shift();
    my $fileName                       = shift();
    my $attenuations                   = shift();
    my $attenuationsUncertainty        = shift();
    my $inclinationCount               = shift();
    my $jobID                          = shift();
    my $errorStatus                    = shift();
    # Open the file.
    my $file                           = new PDL::IO::HDF5($fileName);
    # Extract SEDs.
    my $SEDs                           = $file->group('Peeled/group_00001')->dataset('seds'    )->get();
    my $SEDsUncertainty                = $file->group('Peeled/group_00001')->dataset('seds_unc')->get();
    my $attenuation                    = $SEDs           ->(:,(0),:,(0),(0))->xchg(0,1);
    my $attenuationUncertainty         = $SEDsUncertainty->(:,(0),:,(0),(0))->xchg(0,1);
    my $attenuationAveraged            =     ($attenuation           ->(0:$inclinationCount-1,:)   +$attenuation           ->($inclinationCount:-1,:)   )/2.0;
    my $attenuationUncertaintyAveraged = sqrt($attenuationUncertainty->(0:$inclinationCount-1,:)**2+$attenuationUncertainty->($inclinationCount:-1,:)**2)/2.0;
    if ( $stellarComponent      eq "disk"     ) {
	$attenuations           ->{$stellarComponent}->(                        ($iOpticalDepth),($jOpticalDepth),:,:) .= $attenuationAveraged           ;
	$attenuationsUncertainty->{$stellarComponent}->(                        ($iOpticalDepth),($jOpticalDepth),:,:) .= $attenuationUncertaintyAveraged;
    } elsif ( $stellarComponent eq "spheroid" ) {
	$attenuations           ->{$stellarComponent}->(($iSpheroidScaleRadial),($iOpticalDepth),($jOpticalDepth),:,:) .= $attenuationAveraged           ;
	$attenuationsUncertainty->{$stellarComponent}->(($iSpheroidScaleRadial),($iOpticalDepth),($jOpticalDepth),:,:) .= $attenuationUncertaintyAveraged;
    }
}
