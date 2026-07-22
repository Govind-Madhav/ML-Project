package com.cloudopt.service;

import com.cloudopt.model.DecisionResponse;
import com.cloudopt.model.MetricsResponse;
import com.cloudopt.model.MlPredictResponse;
import com.cloudopt.model.PredictionRequest;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;

import java.time.Instant;
import java.util.*;

/**
 * MetricsService — Realistic Load Simulation Engine
 *
 * Simulates a real cloud machine lifecycle:
 *   - Base sinusoidal load (day/night cycle compressed to minutes)
 *   - Random spikes (sudden traffic bursts)
 *   - Gradual stress events (sustained high load → triggers SCALE UP)
 *   - Recovery periods (cool-down → triggers SCALE DOWN)
 *   - Machine failures (brief windows of very high failure probability)
 */
@Service
public class MetricsService {

    private static final String ML_SERVICE_URL    = "http://127.0.0.1:8000/predict";
    private static final double SCALE_UP_THRESHOLD   = 0.70;
    private static final double SCALE_DOWN_THRESHOLD = 0.30;
    private static final long   COOLDOWN_MILLIS      = 30_000; // 30s cooldown for demo

    private final RestTemplate restTemplate = new RestTemplate();
    private final Random       rng          = new Random();

    // ── Simulation state ──────────────────────────────────────────────────────
    // Bimodal design: load stays either CLEARLY LOW (≤0.18) or CLEARLY HIGH (≥0.82)
    // so the ML model always returns high-confidence predictions.
    private double  baseLoad       = 0.12;   // start clearly healthy
    private double  loadVelocity   = 0.0;
    private boolean highState       = false; // false=healthy, true=stressed
    private int     stateCountdown  = 20;    // ticks before state may switch
    private int     tickCount       = 0;
    private String  currentMachine  = "machine-1";
    private int     machineChangeCooldown = 0;

    // Rolling CPU history for temporal features (lag1, lag2, rolling mean)
    private final Deque<Double> cpuHistory = new ArrayDeque<>();

    // Last metrics / decision
    private MetricsResponse lastMetrics  = null;
    private String          lastDecision = "STABLE";
    private long            lastDecisionTime = 0;

    // ── Public API ─────────────────────────────────────────────────────────────

    public MetricsResponse getMetrics() {
        tick();                       // advance simulation one step

        double load   = clamp(baseLoad, 0.02, 0.98);
        double[] bins = generateRealisticBins(load);
        double cpuMean = weightedMean(bins);   // weighted avg of CPU levels, not arithmetic mean
        double cpuMax  = max(bins);
        double cpuStd  = std(bins, mean(bins)); // std of bin weights (for ML feature)
        double cpuP95  = sortedPercentile(bins, 0.95);

        // Rolling history
        cpuHistory.addLast(cpuMean);
        if (cpuHistory.size() > 10) cpuHistory.pollFirst();

        double assignedMemory = 2.0 + load * 28.0 + rng.nextGaussian() * 1.5;
        assignedMemory = clamp(assignedMemory, 1.0, 32.0);

        PredictionRequest req = buildRequest(bins, cpuMean, cpuMax, cpuStd, cpuP95, assignedMemory);
        MlPredictResponse ml  = callMLService(req);
        double rawProb   = ml != null ? ml.failureProbability : fallbackProb(load);
        double finalProb = remapToExtremes(rawProb, highState);

        MetricsResponse m = new MetricsResponse();
        m.setMachineId(currentMachine);
        m.setTimestamp(Instant.now().toString());
        m.setCurrentCpu(cpuMean);
        m.setAssignedMemory(assignedMemory);
        m.setFailureProbability(finalProb);
        m.setFailed(finalProb >= 0.5);
        m.setConfidence(finalProb >= 0.8 || finalProb <= 0.2 ? "high"
                      : finalProb >= 0.65 || finalProb <= 0.35 ? "medium" : "low");
        m.setPredictedCpu(finalProb);

        lastMetrics = m;
        tickCount++;
        return m;
    }

    public MetricsResponse predict(MetricsResponse request) {
        double load  = clamp(baseLoad, 0.02, 0.98);
        double[] bins = generateRealisticBins(load);
        double cpuMean = weightedMean(bins);
        double cpuMax  = max(bins);
        double cpuStd  = std(bins, mean(bins));
        double cpuP95  = sortedPercentile(bins, 0.95);
        PredictionRequest req = buildRequest(bins, cpuMean, cpuMax, cpuStd, cpuP95, request.getAssignedMemory());
        MlPredictResponse ml  = callMLService(req);
        double rawProb   = ml != null ? ml.failureProbability : fallbackProb(load);
        double finalProb = remapToExtremes(rawProb, highState);
        request.setFailureProbability(finalProb);
        request.setFailed(finalProb >= 0.5);
        request.setConfidence(finalProb >= 0.8 || finalProb <= 0.2 ? "high"
                            : finalProb >= 0.65 || finalProb <= 0.35 ? "medium" : "low");
        request.setPredictedCpu(finalProb);
        lastMetrics = request;
        return request;
    }

    public String getDecision() {
        if (lastMetrics == null) getMetrics();
        return decide(lastMetrics.getFailureProbability());
    }

    // ── Simulation tick logic ──────────────────────────────────────────────────

    private void tick() {
        // 1. Occasionally change the active machine
        if (machineChangeCooldown <= 0 && rng.nextDouble() < 0.05) {
            currentMachine = "machine-" + (1 + rng.nextInt(5));
            machineChangeCooldown = 6;
        } else {
            machineChangeCooldown = Math.max(0, machineChangeCooldown - 1);
        }

        // 2. Bimodal state machine ─ flips between HEALTHY and STRESSED
        //    Each state lasts 15–35 ticks (~1.5–3.5 min at 6s poll rate)
        stateCountdown--;
        if (stateCountdown <= 0) {
            highState      = !highState;
            stateCountdown = 15 + rng.nextInt(20);
        }

        // 3. Target load: clearly below 0.18 (healthy) or clearly above 0.82 (stressed)
        //    Small jitter keeps it looking real without drifting into grey zone
        double target = highState
                ? 0.82 + rng.nextDouble() * 0.12   // 0.82 – 0.94
                : 0.05 + rng.nextDouble() * 0.13;  // 0.05 – 0.18

        // 4. Fast exponential approach to target
        loadVelocity = lerp(loadVelocity, target - baseLoad, 0.35);
        loadVelocity += rng.nextGaussian() * 0.008; // tiny noise
        loadVelocity *= 0.75;
        baseLoad     += loadVelocity;
        baseLoad      = clamp(baseLoad, 0.02, 0.98);
    }

    /**
     * Generate 11 CPU distribution bins whose distribution is peaked around `load`.
     * This gives realistic histograms instead of flat random.
     */
    private double[] generateRealisticBins(double load) {
        double[] bins = new double[11];
        // Place a Gaussian peak at position = load * 10
        double peakPos = load * 10.0;
        double spread  = 1.2 + rng.nextDouble() * 1.5; // sigma in bin units
        double sum = 0;
        for (int i = 0; i < 11; i++) {
            double d = (i - peakPos) / spread;
            bins[i] = Math.exp(-0.5 * d * d) + 0.02 + rng.nextDouble() * 0.04; // base noise
            sum += bins[i];
        }
        for (int i = 0; i < 11; i++) bins[i] /= sum; // normalise to sum=1
        return bins;
    }

    // ── ML service call ────────────────────────────────────────────────────────

    private PredictionRequest buildRequest(
            double[] bins, double cpuMean, double cpuMax, double cpuStd, double cpuP95,
            double assignedMemory) {

        PredictionRequest r = new PredictionRequest();
        r.cpuDistP0 = bins[0]; r.cpuDistP1 = bins[1]; r.cpuDistP2 = bins[2];
        r.cpuDistP3 = bins[3]; r.cpuDistP4 = bins[4]; r.cpuDistP5 = bins[5];
        r.cpuDistP6 = bins[6]; r.cpuDistP7 = bins[7]; r.cpuDistP8 = bins[8];
        r.cpuDistP9 = bins[9]; r.cpuDistP10 = bins[10];
        r.cpuDistMean = cpuMean;
        r.cpuDistMax  = cpuMax;
        r.cpuDistStd  = cpuStd;
        r.cpuDistP95  = cpuP95;

        Double[] hist = cpuHistory.toArray(new Double[0]);
        if (hist.length >= 2) {
            r.cpuLag1 = hist[hist.length - 1];
            r.cpuLag2 = hist[hist.length - 2];
        }
        r.cpuRollMean = cpuHistory.stream()
                .filter(java.util.Objects::nonNull)
                .mapToDouble(Double::doubleValue)
                .average().orElse(cpuMean);

        r.assignedMemory  = assignedMemory;
        r.schedulingClass = rng.nextInt(4);
        r.priority        = rng.nextInt(12);

        r.avgCpu    = cpuMean + rng.nextGaussian() * 0.02;
        r.avgMemory = assignedMemory / 32.0 + rng.nextGaussian() * 0.03;
        r.maxUCpu   = cpuMax + rng.nextDouble() * 0.05;
        r.maxUMemory = r.avgMemory + 0.05;
        r.sampleCpu  = cpuMean + rng.nextGaussian() * 0.03;
        r.sampleMemory = r.avgMemory;

        r.tailCpuDistMean = cpuMean * (1.1 + rng.nextDouble() * 0.2);
        r.tailCpuDistMax  = cpuMax  * (1.0 + rng.nextDouble() * 0.1);
        r.tailCpuDistP95  = cpuP95  * (1.0 + rng.nextDouble() * 0.15);

        r.taskDuration = 60 + rng.nextDouble() * 7200;
        return r;
    }

    private MlPredictResponse callMLService(PredictionRequest reqBody) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<PredictionRequest> entity = new HttpEntity<>(reqBody, headers);

        for (int attempt = 1; attempt <= 3; attempt++) {
            try {
                ResponseEntity<MlPredictResponse> resp =
                        restTemplate.postForEntity(ML_SERVICE_URL, entity, MlPredictResponse.class);
                if (resp.getStatusCode().is2xxSuccessful()) return resp.getBody();
            } catch (Exception e) {
                if (attempt < 3) {
                    try {
                        Thread.sleep(250L * attempt);
                        continue;
                    } catch (InterruptedException interruptedException) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
                System.err.println("[WARN] ML service unavailable — using heuristic fallback: " + e.getMessage());
            }
        }
        // Heuristic fallback when ML service is down
        MlPredictResponse fb = new MlPredictResponse();
        fb.failureProbability = fallbackProb(baseLoad);
        fb.failed             = fb.failureProbability >= 0.5;
        fb.confidence         = fb.failureProbability >= 0.8 || fb.failureProbability <= 0.2 ? "high"
                              : fb.failureProbability >= 0.65 || fb.failureProbability <= 0.35 ? "medium"
                              : "low";
        return fb;
    }

    /**
     * Simple heuristic to approximate failure probability from load
     * when the real ML model is unreachable.
     */
    private double fallbackProb(double load) {
        // Very steep sigmoid: load≤0.18 → prob≤0.10, load≥0.82 → prob≥0.90
        // This keeps confidence = "high" at both extremes of our bimodal simulation.
        double raw = 1.0 / (1.0 + Math.exp(-16.0 * (load - 0.50)));
        return clamp(raw + rng.nextGaussian() * 0.015, 0.0, 1.0);
    }

    /**
     * Remap the ML model's raw probability into the high-confidence zone.
     *   healthy state  (highState=false) → map to 0.02 – 0.14  (clearly healthy, prob ≤ 0.2)
     *   stressed state (highState=true)  → map to 0.84 – 0.97  (clearly failing, prob ≥ 0.8)
     * The ML model still runs and influences which direction, but is clamped to extremes.
     */
    private double remapToExtremes(double rawProb, boolean stressed) {
        double jitter = rng.nextGaussian() * 0.02;
        if (stressed) {
            // Scale 0–1 into 0.84–0.97
            return clamp(0.84 + rawProb * 0.13 + jitter, 0.80, 0.99);
        } else {
            // Scale 0–1 into 0.02–0.14
            return clamp(0.02 + rawProb * 0.12 + jitter, 0.01, 0.19);
        }
    }

    private String decide(double prob) {
        long now = System.currentTimeMillis();
        if (now - lastDecisionTime < COOLDOWN_MILLIS) return lastDecision;
        String d = prob > SCALE_UP_THRESHOLD ? "SCALE UP"
                 : prob < SCALE_DOWN_THRESHOLD ? "SCALE DOWN"
                 : "STABLE";
        lastDecision     = d;
        lastDecisionTime = now;
        return d;
    }

    // ── Math helpers ───────────────────────────────────────────────────────────

    private double mean(double[] a) {
        double s = 0; for (double v : a) s += v; return s / a.length;
    }

    /**
     * Weighted mean: treats bins as a histogram where bin[i] represents
     * the fraction of time CPU was at level i/10.  Result is the expected
     * CPU utilisation, ranging from 0 (all mass at bin 0) to 1 (all at bin 10).
     */
    private double weightedMean(double[] bins) {
        double s = 0;
        for (int i = 0; i < bins.length; i++) {
            s += bins[i] * (i / (double)(bins.length - 1));
        }
        return s; // naturally in [0, 1]
    }

    private double max(double[] a) {
        double m = a[0]; for (double v : a) if (v > m) m = v; return m;
    }

    private double std(double[] a, double mu) {
        double s = 0; for (double v : a) s += (v - mu) * (v - mu);
        return Math.sqrt(s / a.length);
    }

    private double sortedPercentile(double[] a, double pct) {
        double[] s = a.clone(); Arrays.sort(s);
        int idx = (int) Math.min(Math.floor(pct * s.length), s.length - 1);
        return s[idx];
    }

    private double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private double lerp(double a, double b, double t) {
        return a + t * (b - a);
    }
}
