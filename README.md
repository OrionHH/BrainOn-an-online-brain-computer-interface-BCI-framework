# BrainOn: an online brain-computer interface (BCI) framework for feature modulation and processing

## **Welcome!**
Hello！First and foremost, thank you for taking your time to visit the BrainOn repository. Hope it can help or inspire you.

This BrainOn project, which was developed in python with multi-threading concurrency programming, is aimed to create an online brain-computer interface (BCI) framework for feature modulation and processing, allowing researchers to develop their own online experiment programs easily and quickly. The BrainOn supports real-time data stream reading, data processing (e.g., re-reference, down-sampling, filter, data segmentation, and so on), real-time result feedback,pattern recognition (e.g., machine learning, deep learning), as well as other functions. Moreover, The BrainOn can be used to control devices, such as classic brain-controlled typing (see <https://www.youtube.com/watch?v=EW2Q08oHSBo>), brain-controlled robotic arms (see <https://www.youtube.com/watch?v=A1w-e2dBGl0>).

The BrainOn is a part of MetaBCI that is meant to provide a python platform for BCI users to design paradigm, collect data, process signals, present feedbacks and drive robots. Please see <https://github.com/TBC-TJU/brainda> for more information on Brainda, which is the other part of MetaBCI.

This document is to give you more information about the BrainOn. Click the section name to jump to corresponding location.

[TOC]


## **Filelist**

- **BaseFramework.py**  
  A BCI online baseframework with multi-thread concurrency programming was proposed, which included three modules, i.e. *class BaseReadData* for reading data from EEG device in real time., *class BaseProcessingRecog* for signal processing and pattern recognition, *class SendMessageUdp* for message transmission (e.g., real-time classification result).  

- **BasePreProcessing.py**  
  A preprocessing baseframework is mainly used for building offline model for the further online experiment with the training problems.  

- **BaseStimulation.py**  
  A stimulus presentation framework of BCI paradigms that supplying three BCI paradigms (i.e., P300, SSVEP, MI) as well as new paradigms.  
  Under development. It will be uploaded later.  

- **demo_stimulation.py**  
  A demo of SSVEP stimulus derived from the *class BaseStimulation*.  
  Under development. It will be uploaded later. 

- **ReadNeuroscan.py**  
  A demo of reading live data steaming from the Neuroscan Synamps2 amplifier (a EEG acquisition device, Compumedics Ltd., Melbourne Australia) was implemented by deriving from parent *class BaseFramework*. Although the demo is based on Neuroscan device, it's worth noting that the framework is theoretically compatible for most of EEG devices.  

- **simulate_serve_new_packet.py**  
  Simulate EEG amplifier acting as a server and Distribute data in the fixed time interval.
  The software USR-TCP232-TEST as a client can be used to debug.
  The download link is https://www.pusr.com/support/downloads/usr-tcp232-test-V13.  

- **simulate_online_CNT.py**  
  Simulate the online processing flow for program validity that not need connect amplifier or bulid hardware platform. It is convenient for the program validation of the new online experiment without the need for a hardware platform.

- **demo_offline_model.py**  
  A demo of building offline model was developed based on the hybrid P300-SSVEP BCI by deriving from parent *class BasePreProcessing*.  
  When the manuscript is published online, the corresponding sample data for this .py will be uploaded to github.  

- **demo_online_hybridBCI.py**  
  A demo of BCI online modulation and processing framework was implemented based on the hybrid P300-SSVEP BCI system using above base frameworks. Notably, this program can run correctly if only the hardware platform  is connected, which including an EEG amplifier (Neuroscan in this case), ethernet network cable, and stimulation program.  
  When the manuscript is published online, the corresponding show video will be uploaded to youtube.

- **algorithms-folder**  
  Some BCI algorithms to complement Brainda repository, such as Spatial-Temporal Discriminant Analysis (STDA), Shrinkage Linear discriminant analysis (SKLDA), and so on. Hierarchical discriminant component analysis (HDCA), Sliding-HDCA will uploaded in few days.

  The interface form is consistent with scikit-learn for convenient.

---
## **Contributor**
The project was developed and completed in the Lab of Neural Engineering and Rehabilitation, Tianjin University, China.

Dr.Jin Han is the main contributor to the BrainOn project.

---
## **Dependency**
- joblib>=1.0.1  
- mne==0.23.4  
- numpy==1.21.3  
- python>=3.8  
- scikit-learn==1.0.2  
- scipy==1.7.1  

---
## **Abbreviation**
- BCI: brain-computer interface
- EEG: electroencephalogram
- P300: P300 waveform
- SSVEP: steady-state visual evoked potential
- MI: motor imagery

---
## **Troubleshooting**
Please feel free to contact us ( jinhan9165@gmail.com ) if there are any questions. 