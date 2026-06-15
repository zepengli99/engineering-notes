# Kubernetes

Personal notes from working through k8s concepts. Each section starts from a real pain point, then introduces the k8s primitive that solves it.

Local cluster: kind (`k8s-learn`), single node, k8s v1.36.1.

---

## Setup

```powershell
# Start Docker Desktop first, then:
kind create cluster --name k8s-learn
kubectl get nodes  # should show k8s-learn-control-plane Ready
```

---

## Sections

---

### 1. Pod & Deployment — what happens when a container crashes?

With plain Docker, a killed container stays dead. Nobody brings it back.

k8s doesn't manage containers directly — it manages **Pods**. A Pod is a wrapper around one or more containers, sharing a network namespace and storage. In practice, almost always one container per Pod.

```yaml
# manifests/01-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-nginx
spec:
  containers:
    - name: nginx
      image: nginx:alpine
      ports:
        - containerPort: 80
```

```powershell
kubectl apply -f manifests/01-pod.yaml
kubectl get pods
```

But a raw Pod has the same problem as Docker: delete it and it's gone. Nobody restores it.

**Deployment** solves this. It's a declaration: *"keep N copies of this Pod alive at all times."* The Deployment controller runs a continuous **reconciliation loop** — comparing desired state (replicas: 2) against actual state, and correcting any drift.

```yaml
# manifests/02-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nginx          # owns all Pods with this label
  template:               # blueprint for creating new Pods
    metadata:
      labels:
        app: nginx
    spec:
      containers:
        - name: nginx
          image: nginx:alpine
          ports:
            - containerPort: 80
```

```powershell
kubectl apply -f manifests/02-deployment.yaml
kubectl get pods
# delete one pod — Deployment immediately creates a replacement
kubectl delete pod <name>
kubectl get pods
```

Kill a Pod and a replacement appears within seconds. The Pod name changes (it's a new instance), but the Deployment's intent is preserved.

**Key insight:** you declare *what you want*, not *how to get there*. This declarative + reconciliation pattern is the core idea behind every k8s primitive.

**Note on state:** Deployment Pods are designed to be stateless and interchangeable — the replacement Pod starts with an empty local disk. For stateful workloads (databases), you need PersistentVolume + StatefulSet (section 5).

---

### 2. Service — stable access to a moving target

Each Pod gets its own IP, but that IP changes every time the Pod is replaced. With multiple replicas, there's also no built-in load balancing. A caller can't hardcode a Pod IP.

**Node vs Pod:** a Node is the actual machine (physical server, VM, or in kind's case a Docker container) that Pods run on. A single Node hosts many Pods. kube-proxy runs as a system process on each Node and manages iptables rules at the OS kernel level.

```
Node (machine)
├── kube-proxy  (system process)
├── Pod A       (your app)
└── Pod B       (your app)
```

**Service** solves this with a stable virtual IP (ClusterIP) and automatic load balancing across all matching Pods.

```yaml
# manifests/03-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
spec:
  type: NodePort
  selector:
    app: nginx           # forward traffic to all Pods with this label
  ports:
    - port: 80           # Service's virtual port (cluster-internal)
      targetPort: 80     # port on the Pod container to forward to
      nodePort: 30080    # port on the Node for external access (30000-32767)
```

**The three ports:**
- `nodePort` — opened on the Node itself; external traffic enters here
- `port` — the Service's virtual port; other Pods in the cluster use `<ClusterIP>:port`
- `targetPort` — the port the container actually listens on; must match what's in the app

These three can all be different. Example for a Node.js app on port 3000:
```yaml
port: 80        # internal callers use <ClusterIP>:80
targetPort: 3000  # container listens on 3000
nodePort: 31000   # external traffic hits <NodeIP>:31000
```

**How the label selector connects Service to Pods:**

Labels are key-value pairs attached to any k8s resource. Both Deployment and Service use selectors to find Pods, but for different purposes:
- Deployment's `selector`: "I own and manage these Pods — replace them if they die"
- Service's `selector`: "I forward traffic to these Pods"

The Service has no knowledge of the Deployment. It just watches for Pods with matching labels. This is loose coupling — you could have one Service routing to Pods from two different Deployments, as long as the labels match.

```
Deployment ──creates──► Pod [app: nginx, ip: 10.244.0.7]
                         Pod [app: nginx, ip: 10.244.0.9]
                              ▲
Service ─────selector────────┘  (app: nginx)
```

**What ClusterIP actually is:**

ClusterIP is a virtual IP — no process listens on it. It only exists in iptables rules written by kube-proxy on each Node. k8s automatically maintains an **Endpoints** object that tracks the live Pod IPs behind a Service. When a Pod dies and gets replaced, Endpoints updates, kube-proxy rewrites iptables, and traffic routes to the new Pod — all without the ClusterIP changing.

```powershell
kubectl get endpoints nginx-service
# ENDPOINTS: 10.244.0.7:80,10.244.0.9:80
```

Traffic flow when you curl the ClusterIP:
```
curl <ClusterIP>:80
   │
   ▼
iptables intercepts (kube-proxy wrote these rules)
   ├── 50% → DNAT → 10.244.0.7 (Pod A)
   └── 50% → DNAT → 10.244.0.9 (Pod B)
```

DNAT rewrites the destination IP from ClusterIP to the actual Pod IP. Load balancing happens in the kernel — no proxy process involved.

**k8s vs Docker:** k8s doesn't replace Docker — they solve different problems. Docker builds images and runs containers locally. k8s orchestrates containers at scale across many machines. k8s itself never builds images; it pulls pre-built ones from a registry. Typical split: Docker / Docker Compose for local dev, k8s for production.

Also: inside a kind node, Pods are not run by Docker. k8s talks to container runtimes via **CRI (Container Runtime Interface)**. kind uses **containerd** directly — Docker was removed as a k8s runtime in v1.24 (the dockershim layer was dropped as unnecessary indirection, since Docker itself uses containerd underneath).

---

### 3. ConfigMap & Secret — decouple config from the image

Hardcoding config inside an image is wrong: you'd need a separate build per environment (dev/staging/prod), and secrets would be baked into the image layer permanently.

k8s solves this by injecting configuration at runtime, keeping the image environment-agnostic.

Two primitives:
- **ConfigMap** — non-sensitive config (hostnames, ports, feature flags)
- **Secret** — sensitive data (passwords, API keys, TLS certs); values are base64-encoded and get stricter RBAC and storage controls

**ConfigMap:**

```yaml
# manifests/04-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  DB_HOST: "postgres.internal"
  DB_PORT: "5432"
  APP_ENV: "production"
```

Reference it in a Deployment with `envFrom` to load all keys as env vars:

```yaml
# manifests/05-deployment-with-config.yaml
containers:
  - name: myapp
    image: busybox
    envFrom:
      - configMapRef:
          name: app-config    # all keys become environment variables
```

```powershell
kubectl apply -f manifests/04-configmap.yaml
kubectl apply -f manifests/05-deployment-with-config.yaml
kubectl logs -l app=myapp
# DB_HOST=postgres.internal
# DB_PORT=5432
# APP_ENV=production
```

To update config without rebuilding the image: edit the ConfigMap and do a rolling restart.

```powershell
# edit manifests/04-configmap.yaml, then:
kubectl apply -f manifests/04-configmap.yaml
kubectl rollout restart deployment/app-deployment
kubectl logs -l app=myapp
# APP_ENV=staging  ← new value picked up
```

**Secret:**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secret
type: Opaque
data:
  DB_PASSWORD: cGFzc3dvcmQxMjM=   # base64("password123")
```

```powershell
# encode a value:  [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("password123"))
```

Reference alongside a ConfigMap — they merge into the same env namespace:

```yaml
envFrom:
  - configMapRef:
      name: app-config
  - secretRef:
      name: app-secret
```

**Two injection modes:**

`envFrom` (used above) loads everything as environment variables — simple, but the app reads them via `os.getenv`. The alternative is mounting as files, which suits TLS certs or large config files:

```yaml
volumeMounts:
  - name: config-vol
    mountPath: /etc/config     # each key becomes a file at this path
volumes:
  - name: config-vol
    configMap:
      name: app-config
```

**Key insight:** the image is now a pure artifact — the same image runs in dev, staging, and prod. Only the ConfigMap/Secret differs per environment. Rotating a password means updating a Secret and restarting Pods, not rebuilding anything.

---

### 4. Ingress — one entry point for everything

NodePort gives each Service its own port (`:30080`, `:30081`, ...). In production this doesn't scale: users can't access arbitrary ports, and managing dozens of port numbers is chaos. What you want is:

```
example.com/api   → api-service
example.com/web   → frontend-service
admin.example.com → admin-service
```

**Ingress** is a k8s resource that defines these routing rules. It needs an **Ingress Controller** to actually enforce them — the most common is the nginx Ingress Controller.

**The Ingress Controller is itself a Pod** running nginx as a reverse proxy. What makes it special: it has a sidecar process that watches the k8s API server. Every time you apply an Ingress resource, the controller auto-generates a new `nginx.conf` and reloads — no manual config editing. This is reconciliation loop again: the controller keeps nginx's actual routing config in sync with whatever Ingress resources exist in the cluster.

```powershell
# install nginx ingress controller (kind-specific manifest)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
kubectl wait --namespace ingress-nginx --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller --timeout=90s

kubectl get pods -n ingress-nginx
# ingress-nginx-controller-xxx   1/1   Running
```

Deploy two backends to demonstrate path routing:

```yaml
# manifests/06-ingress-backends.yaml
# app-a: responds "hello from app-a"
# app-b: responds "hello from app-b"
# each has its own Deployment + ClusterIP Service
```

Define the routing rules:

```yaml
# manifests/07-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  rules:
    - http:
        paths:
          - path: /app-a
            pathType: Prefix
            backend:
              service:
                name: service-a
                port:
                  number: 80
          - path: /app-b
            pathType: Prefix
            backend:
              service:
                name: service-b
                port:
                  number: 80
```

```powershell
kubectl apply -f manifests/06-ingress-backends.yaml
kubectl apply -f manifests/07-ingress.yaml

# test (kind on Windows needs port-forward since NodePort isn't directly reachable)
kubectl port-forward -n ingress-nginx svc/ingress-nginx-controller 8888:80
curl http://localhost:8888/app-a   # hello from app-a
curl http://localhost:8888/app-b   # hello from app-b
```

**Full request path:**

```
GET /app-a
   │
   ▼
Ingress Controller Pod (nginx)
— matched path /app-a → service-a
   │
   ▼
Service A (ClusterIP + iptables)
   │
   ▼
Pod: app-a
```

**Ingress vs Service:** Service solves stable access to a Pod group — it's an internal cluster primitive. Ingress solves how external traffic is routed to different Services at the cluster boundary. They work at different layers and complement each other.

---

### 5. StatefulSet + PersistentVolume — stateful workloads

Deployment Pods are interchangeable by design: any replica can be killed and replaced, and the caller doesn't notice. This works for stateless apps (API servers, web frontends) but breaks for databases, which have three hard requirements:

- **Data must survive Pod restarts** — a replacement Pod starts with an empty disk
- **Replicas are not equivalent** — a primary and replica have different roles; you can't swap them
- **Startup order matters** — a replica can't join before the primary is ready

**PersistentVolume (PV)** decouples storage from the Pod lifecycle. Three concepts:

- **PV** — the actual storage (local disk, NFS, cloud volume — EBS, GCP PD, etc.)
- **PVC** — a Pod's storage request ("I need 1Gi, ReadWriteOnce")
- **StorageClass** — a policy for auto-provisioning PVs on demand; kind ships with one

The path to the data:
```
Pod container's /data  (mountPath — the container's view)
      ↕ mounted via PVC
PersistentVolumeClaim  (the request)
      ↕ bound to
PersistentVolume       (the actual storage)
      ↕ backed by
kind node filesystem / AWS EBS / GCP PD / NFS / ...
```

`mountPath` is a path *inside the container*, not on the host machine. The Pod is unaware of what backs the PV — swapping cloud providers means changing StorageClass, not Pod YAML.

**StatefulSet** provides stable identity and ordered lifecycle on top of PVs:

```yaml
# manifests/09-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: db
spec:
  serviceName: db
  replicas: 2
  selector:
    matchLabels:
      app: db
  template:
    metadata:
      labels:
        app: db
    spec:
      containers:
        - name: db
          image: busybox
          command: ["sh", "-c", "mkdir -p /data && sleep 3600"]
          volumeMounts:
            - name: data
              mountPath: /data
  volumeClaimTemplates:         # each Pod gets its own PVC automatically
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 100Mi
```

```powershell
kubectl apply -f manifests/09-statefulset.yaml
kubectl get pods -l app=db
# db-0   1/1   Running   — NOT a random hash
# db-1   1/1   Running

kubectl get pvc
# data-db-0   Bound   ...   100Mi    # db-0's dedicated disk
# data-db-1   Bound   ...   100Mi    # db-1's dedicated disk

# write data, kill the pod, verify it survives
kubectl exec db-0 -- sh -c "echo 'persistent data' > /data/record.txt"
kubectl delete pod db-0
# db-0 comes back with the same name and same PVC
kubectl exec db-0 -- cat /data/record.txt
# persistent data  ✓
```

**Deployment vs StatefulSet:**

```
                  Deployment             StatefulSet
Pod name          random hash            db-0, db-1, db-2 (stable)
Replicas          interchangeable        identity-bound
Startup order     all at once            sequential (0 → 1 → 2)
Storage           shared or none         one PVC per Pod
Use case          API servers, web       databases, queues, caches
```

---

### 6. Engineering practice — migrating .env to k8s

Local Docker dev typically uses `.env` files (gitignored) with `docker-compose`. Moving to k8s means splitting that file by sensitivity and managing per-environment drift.

**Step 1: classify every key**

```
# non-sensitive → ConfigMap (safe to commit)
DB_HOST=postgres.internal
DB_PORT=5432
APP_ENV=production
LOG_LEVEL=info

# sensitive → Secret (never commit plaintext)
DB_PASSWORD=xxx
JWT_SECRET=xxx
AWS_SECRET_KEY=xxx
```

**Step 2: manage secrets without committing plaintext**

Three options in order of maturity:

- **kubectl create secret** (manual, no version control — dev/test only)
  ```powershell
  kubectl create secret generic app-secret \
    --from-literal=DB_PASSWORD=xxx --from-literal=JWT_SECRET=xxx
  ```

- **Sealed Secrets** (small teams — encrypt with cluster public key, commit the ciphertext)
  ```powershell
  kubeseal --format yaml < secret.yaml > sealed-secret.yaml
  # sealed-secret.yaml is safe to commit; only the cluster can decrypt it
  ```

- **External Secrets Operator** (larger teams — source of truth lives in AWS Secrets Manager / Vault / GCP SM; k8s only holds a reference)
  ```yaml
  kind: ExternalSecret
  spec:
    secretStoreRef:
      name: aws-secrets-manager
    data:
      - secretKey: DB_PASSWORD
        remoteRef:
          key: prod/myapp/db-password  # real value stays in AWS
  ```

**Step 3: manage per-environment config with Kustomize**

```
k8s/
├── base/                  # shared across all environments
│   ├── deployment.yaml
│   └── kustomization.yaml
└── overlays/
    ├── dev/               # override only what differs
    │   ├── configmap.yaml
    │   └── kustomization.yaml
    ├── staging/
    └── prod/
```

```powershell
kubectl apply -k k8s/overlays/prod/
```

**Common pitfalls:**

- **ConfigMap changes don't hot-reload env vars** — `envFrom` injects at Pod start time. Updating a ConfigMap requires `kubectl rollout restart` to take effect.
- **Quoted values and comments in .env get encoded literally** — `DB_PASSWORD="my pass"` base64-encodes the quotes too. Strip them before migrating.
- **Don't blindly use `envFrom` for everything** — after six months nobody knows what env vars a Pod has. Prefer explicit `env[].valueFrom` in production so the dependency is visible in the YAML.

**Migration path:**
```
1. audit .env — split into sensitive / non-sensitive
2. non-sensitive → ConfigMap, one overlay per environment
3. sensitive → Sealed Secrets or External Secrets, never plaintext in git
4. keep docker-compose for local dev, align key names with k8s ConfigMaps
5. CI/CD pipeline applies the right overlay per environment
```

---

## The big picture

After working through these five primitives, the mental model settles into place.

**The one idea behind everything:** you declare desired state, k8s reconciles reality to match it. Every controller (Deployment, StatefulSet, Service, Ingress) runs this same loop continuously.

**What each primitive solves:**

```
Pod          unit of scheduling — one or more containers sharing network + storage
Deployment   keeps N stateless Pods alive, handles rolling updates
Service      stable virtual IP + load balancing across a Pod group (via iptables)
Ingress      routes external traffic to Services by path or hostname
ConfigMap    injects non-sensitive config at runtime, decoupled from the image
Secret       same as ConfigMap, with stricter access controls
StatefulSet  stable Pod identity + one PVC per Pod, for stateful workloads
PVC / PV     storage that outlives the Pod that created it
```

**Docker's role after k8s:**

Docker becomes a build tool, not a runtime. The image is the artifact — built once, pushed to a registry, pulled by k8s. The split in practice:

```
Local dev          docker compose up        fast feedback loop
Build              docker build + push      in CI, produces the artifact
Deploy             kubectl apply            k8s owns everything after this
Config             ConfigMap + Secret       no more per-environment image builds
Stateful storage   PVC + StorageClass       no more bind-mounting host directories
```

You stop thinking about "which container is running on which machine" — that's k8s's problem. You think about desired state: how many replicas, what image version, what config. The cluster figures out the rest.
