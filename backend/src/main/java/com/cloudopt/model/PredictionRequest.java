package com.cloudopt.model;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Maps to the TaskFeatures Pydantic model in the FastAPI ML service.
 * All 30 features must match exactly (in the same order the model was trained on).
 */
public class PredictionRequest {

    // CPU distribution bins
    @JsonProperty("cpu_dist_p0")  public double cpuDistP0  = 0.0;
    @JsonProperty("cpu_dist_p1")  public double cpuDistP1  = 0.0;
    @JsonProperty("cpu_dist_p2")  public double cpuDistP2  = 0.0;
    @JsonProperty("cpu_dist_p3")  public double cpuDistP3  = 0.0;
    @JsonProperty("cpu_dist_p4")  public double cpuDistP4  = 0.0;
    @JsonProperty("cpu_dist_p5")  public double cpuDistP5  = 0.0;
    @JsonProperty("cpu_dist_p6")  public double cpuDistP6  = 0.0;
    @JsonProperty("cpu_dist_p7")  public double cpuDistP7  = 0.0;
    @JsonProperty("cpu_dist_p8")  public double cpuDistP8  = 0.0;
    @JsonProperty("cpu_dist_p9")  public double cpuDistP9  = 0.0;
    @JsonProperty("cpu_dist_p10") public double cpuDistP10 = 0.0;

    // Summary stats
    @JsonProperty("cpu_dist_mean") public double cpuDistMean = 0.0;
    @JsonProperty("cpu_dist_max")  public double cpuDistMax  = 0.0;
    @JsonProperty("cpu_dist_std")  public double cpuDistStd  = 0.05;
    @JsonProperty("cpu_dist_p95")  public double cpuDistP95  = 0.0;

    // Temporal
    @JsonProperty("cpu_lag1")      public double cpuLag1     = 0.0;
    @JsonProperty("cpu_lag2")      public double cpuLag2     = 0.0;
    @JsonProperty("cpu_roll_mean") public double cpuRollMean = 0.0;

    // Resource allocation
    @JsonProperty("assigned_memory")  public double assignedMemory  = 4.0;
    @JsonProperty("scheduling_class") public double schedulingClass = 0.0;
    @JsonProperty("priority")         public double priority        = 0.0;

    // Optional usage metrics
    @JsonProperty("avg_cpu")    public double avgCpu    = 0.0;
    @JsonProperty("avg_memory") public double avgMemory = 0.0;
    @JsonProperty("max_u_cpu")  public double maxUCpu   = 0.0;
    @JsonProperty("max_u_memory") public double maxUMemory = 0.0;
    @JsonProperty("sample_cpu")   public double sampleCpu  = 0.0;
    @JsonProperty("sample_memory") public double sampleMemory = 0.0;

    // Tail distribution
    @JsonProperty("tail_cpu_dist_mean") public double tailCpuDistMean = 0.0;
    @JsonProperty("tail_cpu_dist_max")  public double tailCpuDistMax  = 0.0;
    @JsonProperty("tail_cpu_dist_p95")  public double tailCpuDistP95  = 0.0;

    // Duration
    @JsonProperty("task_duration") public double taskDuration = 0.0;
}
