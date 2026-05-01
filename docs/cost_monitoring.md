# Cost monitoring

Written for someone deploying this to a real budget, not for someone
running a one-off build. The numbers are operational ranges, not
build-test ones. They include the three Vertex AI architectures so
the operational user can choose with eyes open.

All figures in GBP at 2026 Google Cloud rates. Spot-check before
committing to a budget; cloud prices drift.

## TL;DR for an operational deployment

| Workload | Build / dissertation | Yearly operational |
| --- | ---: | ---: |
| Earth Engine compute | £0 | £0 |
| GCS storage (training + predictions) | < £0.05 / month | £1-3 / month |
| GCS egress (downloads) | minimal | minimal |
| Vertex AI inference (CPU) | n/a (build) | £40-120 / year |
| Vertex AI inference (GPU, on demand) | n/a (build) | £150-400 / year |
| Vertex AI inference (GPU, always-on) | n/a (build) | £1,500-3,500 / year |

Realistic operational pattern for a wildlife protection officer
processing fire events as they happen: GCS storage steady-state plus
on-demand Vertex AI CPU inference when a fire happens. Roughly **£1-10
per month at typical use** if you remember to deprovision the endpoint
between events.

## Earth Engine

Free under non-commercial registration. There is no per-compute charge
and no egress charge for exports to GCS. The only practical cost is
your time waiting for export tasks to clear the queue.

Quotas: Earth Engine throttles concurrent exports per project. The
default of 4 concurrent tasks (`export.max_concurrent_tasks` in the
config) is well within the free-tier limits and gives a full v2
equivalent harvest a 30-90 minute wall-clock time.

If Sonia later registers an Earth Engine **commercial** project for
operational use beyond the BSc dissertation, that pricing changes.
Commercial Earth Engine has a per-task and per-storage component but
the typical heath-fire-mapping workload (a few fires per year, a few
hundred patches each) sits comfortably inside the lowest tier. Expect
under £200 / year for typical operational use, possibly free if Sonia
qualifies for the Earth Engine for non-profits programme.

## Cloud Storage

Standard tier in europe-west2 (London) is roughly £0.018 per GB per
month. The full v2-equivalent harvest is roughly 1.5 GB. So:

* Storage: £0.03 / month for the harvest output.
* Predictions: a few hundred KB per prediction GeoTIFF. Negligible
  unless you keep many years of run archives.
* Egress: free within Google Cloud (Vertex AI reads bucket contents
  in-region for free); £0.06 / GB if you download to your laptop.
  Downloading the full harvest once costs around £0.10.

Operational steady state: under £1 / month, dominated by the harvest
output sitting in the bucket. Trivially within any personal budget.

## Vertex AI inference: three architectures

Phase 3 of this build will pick one of these. The choice is operational,
not technical, and depends on how often Sonia needs predictions.

### A. GEE-only inference (cheapest)

Run the trained model from inside Earth Engine using
`ee.Model.fromVertexAi` style hosting, or — simpler — re-use the GEE
harvesting pipeline to apply the model directly to S1 acquisitions and
write predicted reflectance back to GCS.

* Cost: £0 / month, no Vertex AI involvement.
* Latency: as long as the export queue (minutes to an hour).
* Suitability: when prediction is rare (a few times per year per fire
  event), latency is not critical, and Sonia is comfortable running a
  single CLI command.

### B. Vertex AI CPU-hosted endpoint, on demand

Deploy the SavedModel to a Vertex AI Endpoint with a CPU-only machine
type (e.g. `n1-standard-4`). Provision the endpoint when a fire event
needs prediction, run inference, deprovision when done.

* Provisioning + inference + deprovisioning cycle: roughly 5-10
  minutes of wall-clock for one fire event.
* Cost per cycle: £0.80 (machine cost roughly £0.20/hour for 5-10
  minutes plus a small Vertex AI per-prediction fee).
* Yearly: 50-150 fire events × £0.80 = **£40-120**.
* Suitability: Sonia checks inference within 10 minutes of a fire
  event. Most operational use cases. Recommended default unless the
  use case demands sub-minute latency.

### C. Vertex AI GPU-hosted endpoint, always on

Deploy with a GPU (e.g. `n1-standard-4` + `nvidia-tesla-t4`) and leave
the endpoint provisioned 24/7 for sub-second inference. This is the
expensive option.

* Always-on cost: roughly £4-9 / day for the smallest GPU; **£1,500-3,500
  / year**.
* Suitability: only when sub-second inference is genuinely required
  (e.g. live monitoring during a heatwave with continuous Sentinel-1
  drops; Sonia is unlikely to need this for individual heath fire
  events).

### Switching between architectures

The Phase 3 build will provide all three deployment scripts, with
sensible defaults pointing at architecture B (CPU on-demand). Switching
from B to A is a config change; switching from B to C is a different
deployment script. None of these decisions need to be made before
Phase 1 review.

## Quotas and free-tier ceilings

* Earth Engine non-commercial: 250 concurrent compute requests, a few
  thousand exports per month. Comfortably within typical use.
* GCS Standard: no relevant ceilings at this scale.
* Vertex AI: free trial includes £230 of credit; new Google Cloud
  projects also receive £230 in trial credits. Either covers the build
  and a few months of operational use without spending any real money.

## Teardown — when Sonia is done with the v2 build-test artefacts

This is operationally important: the build was run against Marcus's
Google account. To reproduce under Sonia's own account she'll repeat
the Setup section with her own credentials. To clear out the build-test
artefacts to avoid orphaned-cost surprises:

```bash
# 1. Delete the bucket contents and the bucket itself.
gcloud storage rm -r gs://<your-bucket>/gee_s1s2_translator/
gcloud storage buckets delete gs://<your-bucket>/

# 2. Delete the Vertex AI endpoint (Phase 3).
gcloud ai endpoints list --region=europe-west2
gcloud ai endpoints delete <ENDPOINT_ID> --region=europe-west2

# 3. Delete the Vertex AI model registry entries.
gcloud ai models list --region=europe-west2
gcloud ai models delete <MODEL_ID> --region=europe-west2

# 4. Optional: delete the entire Cloud project for a clean wipe.
gcloud projects delete <project-id>
```

For the **v2 PyTorch artefacts** (which lived locally on Marcus's
laptop, not on Google), the teardown is simply removing the
`s1s2-translator/data/` directory; nothing in the cloud needs cleaning
up.

## Practical advice

* Set up a billing alert at £10 / month to catch any unexpected charge
  before it accumulates. Cloud Console → Billing → Budgets & alerts.
* Remember to deprovision Vertex AI endpoints (architecture B or C)
  when you're not using them. The biggest accidental-cost risk is a
  GPU endpoint left running over a holiday weekend.
* The harvest is idempotent: re-running it never duplicates patches or
  manifest rows. So running `gee_s1s2 harvest` repeatedly during
  development is safe and cheap.
* Earth Engine's commercial tier becomes relevant only if Sonia is
  publishing model outputs commercially or processing data for a
  paying client. For a wildlife protection officer's professional
  practice, non-commercial registration almost certainly applies; if
  in doubt, the Earth Engine team will respond to a written question.
