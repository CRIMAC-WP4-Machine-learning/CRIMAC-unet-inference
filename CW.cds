<?xml version="1.0" encoding="UTF-8"?>

<ModuleContainer version="3">
   <modules>
      <module name="NetcdfWriterModule">
         <parameters>
            <parameter name="Active">true</parameter>
            <parameter name="DirName">sv</parameter>
            <parameter name="MainFrequency">38</parameter>
            <parameter name="DeltaRange">0.2</parameter>
            <parameter name="MaxRange">500</parameter>
            <parameter name="OutputType">SV_AND_ANGLES</parameter>
            <parameter name="WriteAngels">true</parameter>
            <parameter name="FftWindowSize">10</parameter>
            <parameter name="DeltaFrequency">1</parameter>
         </parameters>
      </module>
      <module name="ChannelDataRemovalModule">
         <parameters>
            <parameter name="Active">true</parameter>
            <parameter name="Channels"/>
            <parameter name="ChannelsFromEnd"/>
            <parameter name="Frequencies"/>
            <parameter name="KeepSpecified">true</parameter>
         </parameters>
      </module>
   </modules>
</ModuleContainer>
