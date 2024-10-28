# Workflow for focal mechanism and stress inversion

## Description

This project provides a complete workflow for processing seismic data, from raw waveforms to full moment tensor solutions. It includes automated phase detection, amplitude estimation, and two different approaches for moment tensor inversion.

## Installation and Setup

### Step 1: Install Required Packages
Install the required environments from the conda_envs directory. For example:
```bash
conda env create -f .\conda_envs\eqnet.yml
conda activate eqnet
```

For the EQNet and SKHASH refer to their documnets for the installation.

## Processing

### Step 2: Build Waveforms and Prepare Catalog

1. Place your raw seismic data in the `waveforms` directory.
the pattern for the waveforms directory must follow this:
waveforms/{event_id}/{event_id}_{station_id}.mseed
2. Place the station xml files in the stationxml directory with the following pattern:  
stationxml/{station_id}.xml  
3. Here is an example:  
wavefroms/eq02387/eq02387_UU.FORK.01.GH.mseed  
wavefroms/eq02387/eq02387_CP.C1608..GP.mseed  
wavefroms/eq02387/eq02387_UU.FSB2.01.HH.mseed  
stationxml/UU.FORK.01.GH.xml  
stationxml/CP.C1608..GP.xml  
stationxml/UU.FSB2.01.HH.xml  

### Step 3: Phase Detection and Amplitude Estimation

1. Run 1_phase_detection.ipynb to prepare the list of mseed files to do the phase detection. Then run the following lines to start the phase detection.
```bash
  python /home/ahmadfervo/EQNet/predict.py \
  --model phasenet_plus \
  --data_list /home/ahmadfervo/cape/results_eqnet/mseed.txt \
  --response_path /home/ahmadfervo/cape/stationxml \
  --result_path /home/ahmadfervo/cape/results_eqnet \
  --sampling_rate 500 --highpass_filter 5 --batch_size=1 --format mseed
```
2. After the phase detection step run the rest of the 1_phase_detection.ipynb notebook to merge the picks and generate a single csv file.
3. Perform amplitude estimation:
```bash
   conda activate seisbench
   python 2_eqnet_fix_amplitude.py
```

### Step 4: Focal Mechanism Inversion (SKHASH)

Run the SKHASH inversion to obtain initial focal mechanism solutions:

```bash
conda activate skhash
python skhash/1_run_inversion_psh_febMarch.py
```

### Step 5: Full Moment Tensor Inversion (MTfit)

Perform the final moment tensor inversion using MTfit:
1. Prepare the inputs for the inversion using the mtfit/1_create_input.ipynb notebook
2. Run the moment tensor inversion
```bash
conda activate mtfit
python mtfit/2_run_inversion_iterate.py
```
### Step 6: Stress inversion (MSATSI)
The stress inversion need the Msatsi package and Matlab.
Run the stress_inversion/stress_febMarch.ipynb to do the clustring using spatial and temporal information and then stress inversion.
The notebook will generate the stress inversion results for the orientation and relative amplitude of the principal stress components.

## Output Files

The pipeline generates the following outputs:  
- Phase picks: `/home/ahmadfervo/cape/result_eqnet`  
- Merged picks: `/home/ahmadfervo/cape/result_eqnet/picks.csv`  
- Amplitude measurements: `/home/ahmadfervo/cape/picks_amp.csv`  
- SKHASH focal mechanisms: `/home/ahmadfervo/cape/skhash`  
- SKHASH faulting regime: `/home/ahmadfervo/cape/skhash/focal_mechanisms_regime.csv`  
- MTfit moment tensor solutions: `/home/ahmadfervo/cape/mtfit`  
- MTfit faulting regime: `/home/ahmadfervo/cape/mtfit/focal_mechanisms_regime.csv`  
- Stress inversion result: `/home/ahmadfervo/cape/stress_inversion`  
