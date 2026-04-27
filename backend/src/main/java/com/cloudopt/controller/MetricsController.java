package com.cloudopt.controller;

import org.springframework.web.bind.annotation.*;
import org.springframework.beans.factory.annotation.Autowired;
import com.cloudopt.service.MetricsService;
import com.cloudopt.model.MetricsResponse;
import com.cloudopt.model.DecisionResponse;
import java.time.Instant;

@RestController
@RequestMapping("/api")
public class MetricsController {
    @Autowired
    private MetricsService metricsService;

    @GetMapping("/metrics")
    public MetricsResponse getMetrics() {
        return metricsService.getMetrics();
    }

    @PostMapping("/predict")
    public MetricsResponse predict(@RequestBody MetricsResponse request) {
        return metricsService.predict(request);
    }

    @GetMapping("/decision")
    public DecisionResponse getDecision() {
        MetricsResponse metrics = metricsService.getMetrics();
        String decision = metricsService.getDecision();
        DecisionResponse resp = new DecisionResponse();
        resp.setMachineId(metrics.getMachineId());
        resp.setDecision(decision);
        resp.setTimestamp(Instant.now().toString());
        resp.setReason("Based on predicted CPU: " + metrics.getPredictedCpu());
        return resp;
    }
}
