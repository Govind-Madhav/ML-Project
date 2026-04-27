package com.cloudopt.model;

public class MetricsResponse {
    private String  machineId;
    private double  currentCpu;
    private double  predictedCpu;       // kept for backward compat (holds failure prob)
    private double  assignedMemory;
    private String  timestamp;
    private double  failureProbability;
    private boolean failed;
    private String  confidence;

    public String  getMachineId()           { return machineId; }
    public void    setMachineId(String v)   { this.machineId = v; }

    public double  getCurrentCpu()          { return currentCpu; }
    public void    setCurrentCpu(double v)  { this.currentCpu = v; }

    public double  getPredictedCpu()        { return predictedCpu; }
    public void    setPredictedCpu(double v){ this.predictedCpu = v; }

    public double  getAssignedMemory()         { return assignedMemory; }
    public void    setAssignedMemory(double v) { this.assignedMemory = v; }

    public String  getTimestamp()           { return timestamp; }
    public void    setTimestamp(String v)   { this.timestamp = v; }

    public double  getFailureProbability()         { return failureProbability; }
    public void    setFailureProbability(double v) { this.failureProbability = v; }

    public boolean isFailed()              { return failed; }
    public void    setFailed(boolean v)    { this.failed = v; }

    public String  getConfidence()          { return confidence; }
    public void    setConfidence(String v)  { this.confidence = v; }
}
