package com.cloudopt.model;

import com.fasterxml.jackson.annotation.JsonProperty;

/** Response from FastAPI /predict endpoint */
public class MlPredictResponse {
    @JsonProperty("failed")
    public boolean failed;

    @JsonProperty("failure_probability")
    public double failureProbability;

    @JsonProperty("confidence")
    public String confidence;
}
