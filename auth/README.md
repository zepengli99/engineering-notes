# Auth & Identity

IAM, OAuth 2.0, OIDC, Keycloak, SPIFFE/SPIRE.

---

## The IAM Landscape

IAM (Identity and Access Management) answers two distinct questions:

```
Authentication (AuthN)  →  Who are you? Prove it.
Authorization  (AuthZ)  →  What are you allowed to do?
```

These are different problems with different solutions. Everything in this document sits somewhere on this map:

```
IAM
├── Human Identity
│   ├── Protocol:       OAuth 2.0 (AuthZ) + OIDC (AuthN)
│   ├── Implementation: Keycloak / Auth0 / Okta / AWS Cognito
│   └── Token format:   JWT
│
└── Workload Identity (service-to-service, no human involved)
    ├── Spec:           SPIFFE
    └── Implementation: SPIRE
```

**OAuth 2.0 is an authorization protocol, not an authentication protocol.** The "Log in with Google" feeling comes from OIDC layered on top of OAuth — OAuth itself only grants access tokens, it doesn't tell you who the user is. That distinction matters when reading specs and debugging token flows.

---

## OAuth 2.0

### Why it exists

The naive approach to third-party access: give the third-party app your password. Problems:

```
1. The app gets full access — same as your credentials
2. Revoking access means changing your password, which invalidates everything
3. You can't selectively revoke one app's access
```

OAuth solves this: a third-party app can access your resources without your password, with limited scope, and the authorization can be revoked independently.

---

### The four roles

```
Resource Owner       →  You (the user who owns the data)
Client               →  The third-party app that wants access
Authorization Server →  Issues tokens (e.g. GitHub's OAuth server)
Resource Server      →  Holds the actual data (e.g. GitHub API)
```

The Authorization Server and Resource Server are often run by the same company but are logically separate services.

---

### Authorization Code Flow

The most common and most secure flow, for web apps with a backend:

```
1. User clicks "Log in with GitHub"
   Client redirects user to GitHub:

   GET https://github.com/login/oauth/authorize
     ?client_id=abc123
     &redirect_uri=https://myapp.com/callback
     &scope=read:user,repo
     &response_type=code
     &state=random-csrf-token

2. User logs in on GitHub and clicks "Authorize"
   GitHub redirects back to the Client:

   GET https://myapp.com/callback?code=xyz789&state=random-csrf-token

3. Client exchanges code for token (backend request, never seen by user):

   POST https://github.com/login/oauth/access_token
     client_id=abc123
     client_secret=secret
     code=xyz789

4. GitHub returns:

   {
     "access_token":  "gho_xxx",
     "token_type":    "bearer",
     "scope":         "read:user,repo",
     "refresh_token": "ghr_yyy",
     "expires_in":    3600
   }

5. Client uses token to call GitHub API:

   GET https://api.github.com/user
   Authorization: Bearer gho_xxx
```

---

### Why two steps (code → token) instead of returning the token directly?

Step 2 happens through a browser redirect — the URL appears in browser history, server access logs, and Referer headers. If the token were in that URL, it would leak into all of those.

```
Browser history      →  full URL stored
Server access logs   →  GET /callback?code=xyz789 logged
Referer header       →  full URL sent when navigating away
CDN / proxy logs     →  may log URLs
```

The code is short-lived, single-use, and can only be exchanged for a token by the Client backend (which has the `client_secret`). POST body doesn't appear in any of these places.

> **HTTPS protects the network, not the logs.** HTTPS encrypts everything — URL path, query string, and body — from interception on the wire. The reason to prefer POST body over URL parameters is not about network security; it's about where URLs end up after they arrive.

---

### HTTPS in 30 seconds

```
1. Client connects; Server sends its certificate (public key + CA signature)
2. Client verifies the certificate: Do I trust this CA? Does the domain match? Is it expired?
3. Client encrypts a random value with Server's public key, sends it over
4. Both sides derive a symmetric key from that random value
5. All subsequent traffic encrypted with that symmetric key
```

```
Asymmetric encryption (public/private key)  →  key exchange only, slow
Symmetric encryption (AES etc.)             →  actual data, fast
Certificate + CA                            →  proves "this public key really belongs to github.com"
```

**CA is a trusted third party** — like a government-issued ID. The browser trusts the CA, so it trusts the certificate. Browsers and operating systems ship with a pre-installed list of trusted root CAs; that list is the root of trust for the entire system.

CA validation levels:
```
DV (Domain Validation)    →  only verifies you control the domain (most common, Let's Encrypt)
OV (Organization)         →  also verifies the organization exists
EV (Extended Validation)  →  strictest; browsers used to show a green company name
```

---

### CSRF and the `state` parameter

**What CSRF is:**

Cross-Site Request Forgery. The browser automatically attaches cookies to requests. An attacker can exploit this:

```
User is logged into bank.com (browser holds the session cookie)
User visits evil.com
evil.com contains: <img src="https://bank.com/transfer?to=attacker&amount=1000" />
Browser loads the "image", sends the bank.com cookie automatically
bank.com validates the cookie, executes the transfer
```

**CSRF in the OAuth context:**

```
Attacker initiates OAuth flow, gets a code
Attacker doesn't complete the flow — instead sends the callback URL to the victim
Victim clicks it; Client exchanges attacker's code for a token
Victim's account gets bound to attacker's identity
```

**How `state` prevents this:**

```
Client generates a random state value, stores it in the user's session
GitHub echoes state back in the callback
Client checks: callback state == session state?

Attacker's callback URL carries a state that doesn't match the victim's session
→ request rejected
```

---

### `scope`

Declared permissions the user can see and consent to. `read:user` reads profile info only; `repo` reads and writes repositories. The Authorization Server encodes scope into the token; the Resource Server checks it on every request.

---

### access_token vs refresh_token

```
access_token   →  short-lived (minutes to hours), sent on every request
refresh_token  →  long-lived (days to months), used only to get a new access_token
```

When access_token expires:

```
POST /token
  grant_type=refresh_token
  refresh_token=ghr_yyy
  client_id=abc123
  client_secret=secret

→ returns new access_token (sometimes a new refresh_token too)
→ user never has to re-authorize
```

The user perceives "you need to log in again" only when the refresh_token itself expires or is revoked. Well-designed apps set refresh_token TTL long enough that users almost never see it.

**Where tokens live:** refresh_tokens only belong in a backend server. SPAs have no secure storage (localStorage is readable by XSS), so they typically don't hold refresh_tokens — they either use silent refresh (hidden iframe re-running the auth flow) or PKCE, covered below.

---

### Token revocation: JWT vs opaque

This matters when you want to immediately cut off access.

**Opaque access_token:**

```
Resource Server receives token → calls Authorization Server to validate
  POST /introspect  token=gho_xxx

Authorization Server checks its database: valid? banned?
→ Revocation takes effect immediately
```

**JWT access_token:**

```
Resource Server validates locally: check signature + check exp field
No call to Authorization Server

Authorization Server bans the token?
→ Resource Server doesn't know; keeps accepting it
→ Token continues working until it naturally expires
```

JWTs cannot be immediately revoked. The trade-off is latency vs correctness: local validation is fast and scales well, but revocation has a lag equal to the token's remaining lifetime. This is why JWT access_tokens are kept short-lived (minutes to one hour).

**The common pattern:**

```
access_token   →  JWT, short-lived (fast validation, acceptable revocation lag)
refresh_token  →  opaque, stored in DB (can be immediately revoked)
```

Revoking a refresh_token stops the user from getting new access_tokens. The current access_token keeps working until expiry — a known, accepted window.

---

## PKCE — Proof Key for Code Exchange

### The problem: public clients can't keep secrets

The standard Authorization Code Flow relies on `client_secret` to prove the Client is legitimate. But SPAs and mobile apps are **public clients** — their code is fully visible:

```
SPA         →  JavaScript in the browser; anyone can F12 and read it
Mobile app  →  APK can be decompiled; any string can be extracted
```

Putting a `client_secret` in either is the same as having no secret.

### The attack: Authorization Code interception

Mobile apps use custom URL schemes for OAuth callbacks (e.g. `myapp://callback`). Any app can register the same scheme:

```
Legitimate app registers myapp://callback
Malicious app also registers myapp://callback

User completes authorization; GitHub redirects:
  myapp://callback?code=xyz789

OS doesn't know which app to pick — may send it to the malicious one
Malicious app has the code, and there's no client_secret to stop it from exchanging it
```

### How PKCE works

Instead of a static secret, the Client generates a fresh cryptographic proof per authorization request:

```
code_verifier   = random high-entropy string (43–128 chars), stored in memory
code_challenge  = BASE64URL(SHA256(code_verifier))
```

The challenge goes in the authorization request; the verifier goes in the token exchange:

```
1. Client generates code_verifier + code_challenge

   GET /authorize
     ?code_challenge=E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM
     &code_challenge_method=S256
     ...
   Authorization Server stores code_challenge alongside the issued code

2. Callback (same as standard flow)

3. Client exchanges code for token — no client_secret, sends code_verifier instead:

   POST /token
     code=xyz789
     code_verifier=dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk

   Authorization Server verifies: SHA256(code_verifier) == stored code_challenge?
   → match: issue token
   → mismatch: reject
```

An attacker who intercepts the code still can't exchange it — SHA256 is one-way, so knowing `code_challenge` gives you nothing to compute `code_verifier` from.

### Diff against standard Authorization Code Flow

```
Standard flow                    PKCE flow
─────────────────────────────────────────────────────────
Step 1: nothing extra            generate code_verifier / code_challenge
Step 1: no extra params          add code_challenge + code_challenge_method
Step 3: send client_secret       send code_verifier; no client_secret
```

Everything else is identical. PKCE is a minimal extension to the existing flow, not a separate protocol. RFC 9700 now recommends PKCE for all OAuth clients, not just public ones.

---

## OIDC — OpenID Connect

### The problem: OAuth doesn't know who the user is

After a full OAuth flow, you have an `access_token`. That token tells the Resource Server "this request has permission" — it says nothing about which user it belongs to.

```
access_token = "gho_xxx"

Client asks: whose token is this?
OAuth spec: not my concern
```

OIDC solves this: an identity layer built on top of OAuth 2.0 that returns an `id_token` — a JWT containing standardized user identity claims.

### What OIDC adds

Three additions on top of the OAuth Authorization Code Flow:

**1. `openid` in scope** — signals to the Authorization Server that identity is needed:

```
scope=openid profile email
       ↑
       without this it's plain OAuth, no id_token
```

**2. `id_token` in the response:**

```json
{
  "access_token": "gho_xxx",
  "token_type":   "bearer",
  "expires_in":   3600,
  "id_token":     "eyJhbGc..."
}
```

**3. Standard claims inside `id_token`:**

```json
{
  "iss":   "https://accounts.google.com",
  "sub":   "1234567890",
  "aud":   "abc123",
  "exp":   1735689600,
  "iat":   1735686000,
  "email": "user@example.com",
  "name":  "Zhang San"
}
```

| Claim | Meaning |
|---|---|
| `iss` | Issuer — who signed this token |
| `sub` | Subject — permanent unique user ID, never changes |
| `aud` | Audience — which Client this token is intended for |
| `exp` / `iat` | Expiry / issued-at timestamps |
| `email`, `name` | Optional; returned when the corresponding scope is requested |

### access_token vs id_token

The most commonly confused distinction in OIDC:

```
access_token  →  for the Resource Server
                 proves "this request has permission"
                 Resource Server validates it and decides whether to respond

id_token      →  for the Client
                 tells the Client who the user is
                 Client reads sub / email to identify the user
                 should NOT be forwarded to the Resource Server
```

Use `sub` (not `email`) as the stable user identifier in your database — email can change if the user updates their account.

### nonce — preventing id_token replay

`state` defends against CSRF on the authorization request. `nonce` defends against replay of the `id_token` itself:

```
Client generates random nonce, includes it in the authorization request
Authorization Server writes the nonce into the id_token
Client verifies: nonce in id_token == nonce it generated?

→ prevents an attacker from reusing a captured id_token
```

### OIDC flow diff against OAuth

```
1. Authorization request — add openid scope + nonce
   GET /authorize
     ?scope=openid profile email
     &nonce=random-nonce
     ...

2. Callback (same as OAuth)

3. Token exchange (same as OAuth)

4. Response — id_token added:
   {
     "access_token": "xxx",
     "id_token":     "eyJ..."
   }

5. Client validates id_token locally (no network call):
   - verify signature (using Authorization Server's public key)
   - check iss / aud / exp
   - check nonce
   → read sub → user is authenticated
```

### "Sign up with Google" — how third-party registration works

This pattern is exactly OIDC:

**First visit (registration):**

```
App completes OIDC flow, receives id_token
Reads: sub = "google-user-123456"

Query DB: any user with external_id = "google-user-123456"?
→ no → new user, INSERT:
    { id: 1, external_id: "google-user-123456", provider: "google", email: "..." }
→ create session, user is logged in
```

**Subsequent logins:**

```
App completes OIDC flow, receives id_token
Reads: sub = "google-user-123456"

Query DB: any user with external_id = "google-user-123456"?
→ found → log in directly, no registration needed
```

The app's own database holds no password — only `sub` as the external identity anchor. Authentication is fully delegated to Google / GitHub / etc.

**Multiple providers linked to one account:**

```
users table:      id=1, email="user@gmail.com"
identities table: user_id=1, provider="google", sub="google-123"
                  user_id=1, provider="github", sub="github-456"
```

Logging in with either provider maps to the same `user_id=1`.

---

## Keycloak

### What it is

In OAuth/OIDC vocabulary: **Keycloak is an Authorization Server you deploy yourself.**

It implements OAuth 2.0, OIDC, and SAML. Your applications (Clients) redirect users to Keycloak for login and receive tokens back — identical to how you'd use Google as an Authorization Server, except you control the server.

```
OAuth/OIDC concept          Keycloak concrete
────────────────────────────────────────────
Authorization Server   →   Keycloak itself
Resource Owner         →   users registered in Keycloak
Client                 →   applications you configure in Keycloak
access_token/id_token  →   JWTs issued and signed by Keycloak
```

### When you need your own Authorization Server

"Sign in with Google" requires users to have Google accounts and you to trust Google. That doesn't always apply:

```
Internal enterprise systems
  Employees log into internal tools; their identities live in company LDAP/AD
  You need to revoke access the moment someone leaves — Google won't do that for you

Your own user base
  Users signed up with email + password, not Google accounts
  You need something to manage those accounts and issue tokens

Multiple login methods unified
  Users can log in with Google OR email/password
  Backend only wants to handle one token format
  → Keycloak sits in the middle, issues its own tokens regardless of how the user logged in
```

### Core concepts

**Realm** — the most important concept. A fully isolated namespace with its own users, clients, roles, signing keys, and login page configuration.

```
Keycloak instance
├── master realm       ← admin use only, never put business logic here
├── myapp-prod realm
└── myapp-dev realm
```

Different realms are completely isolated — users don't cross over, tokens don't work across realms.

**Client** — a registered application. Configured with: `client_id`, `client_secret`, `redirect_uri`, allowed grant types.

**User** — end users with credentials, assigned roles.

**Role** — permission labels encoded directly into the access token JWT:
```
Realm Role   →  cross-application (e.g. admin, user)
Client Role  →  specific to one application (e.g. myapp:editor)
```

### Running it

```bash
docker run -p 8080:8080 \
  -e KC_BOOTSTRAP_ADMIN_USERNAME=admin \
  -e KC_BOOTSTRAP_ADMIN_PASSWORD=admin \
  quay.io/keycloak/keycloak:latest start-dev
```

Admin UI at `http://localhost:8080`.

### Standard endpoints

OIDC Discovery — everything your application needs is in one place:

```
GET http://localhost:8080/realms/myrealm/.well-known/openid-configuration
```

Returns authorization_endpoint, token_endpoint, userinfo_endpoint, jwks_uri, etc. Your application only needs this one URL; all other endpoints are discovered from it. **This is the practical value of OIDC standardization** — switching from Keycloak to Auth0 or Google only requires changing this URL; application code doesn't change.

### What a Keycloak token looks like

```json
{
  "iss": "http://localhost:8080/realms/myrealm",
  "sub": "user-uuid-123",
  "aud": "myapp",
  "exp": 1735689600,
  "realm_access": {
    "roles": ["user", "admin"]
  },
  "resource_access": {
    "myapp": {
      "roles": ["editor"]
    }
  },
  "email": "user@example.com"
}
```

Your Resource Server reads roles directly from the JWT — no extra database query needed for authorization.

### SSO — how single sign-on actually works

"Log in once, access multiple applications" comes from where the session lives:

```
User logs into Keycloak → Keycloak creates its own session (cookie on the Keycloak domain)
User opens App A → redirected to Keycloak → session found → token issued, no password prompt
User opens App B → redirected to Keycloak → same session → token issued, no password prompt
```

The session lives at the Keycloak layer, not inside each application. Applications only hold short-lived tokens; Keycloak holds the long-lived session. Logging out of Keycloak invalidates the session and all applications lose their ability to silently renew tokens.

### Authorization Server options

```
Use Google / GitHub directly   →  users log in with their existing accounts
Self-hosted                    →  Keycloak (open source, you operate it)
Managed, enterprise-focused    →  Auth0, Okta (full-featured, expensive, compliance-ready)
Managed, developer-focused     →  Clerk, Firebase Auth (fast setup, great DX for React/Next.js)
```

Switching between any of these only changes the discovery URL — the OAuth/OIDC protocol and your application's token validation logic stay the same.

---

## SPIFFE / SPIRE

### A different problem

Everything above solves **human identity**: user logs in, gets a token, accesses a resource.

SPIFFE solves a different problem: **in a microservices environment, when Service A calls Service B, how does Service B know the caller is really Service A and not an attacker?**

The naive approach — static API keys in environment variables — has a fundamental weakness:

```
Static key leaked → valid forever
Key has no semantic binding to a specific service → anyone with the key can impersonate anything
Keys stored in env vars / config files / repos → many ways to leak
```

### The SPIFFE approach

Give every service a **cryptographic identity** — like an HTTPS certificate, but:

- Automatically issued, no manual process
- Bound to the workload's runtime context (this Pod, this process)
- Short-lived, automatically rotated
- Standardized format

This identity is called an **SVID (SPIFFE Verifiable Identity Document)**, typically an x509 certificate whose Subject Alternative Name contains a SPIFFE ID:

```
spiffe://trust-domain/path

e.g.:
spiffe://mycompany.com/backend/payment-service
spiffe://mycompany.com/backend/order-service
```

### Verification via mTLS

```
Regular TLS:  client verifies server's certificate (you verify github.com is real)
mTLS:         both sides verify each other's certificate

Service A → Service B:
  A presents its SVID
  B checks: signed by a CA we both trust? SPIFFE ID is a service I recognize?
  B presents its own SVID
  A does the same check
  → mutual identity confirmed, connection established
```

### SPIFFE vs SPIRE

SPIFFE is the specification. SPIRE is the implementation.

```
SPIRE Server  →  central node, maintains trust, signs certificates
SPIRE Agent   →  runs on every machine/node, issues SVIDs to local workloads
Workload API  →  local Unix socket; services fetch their own cert through it,
                 without needing to know SPIRE exists
```

```
Service starts
  → requests cert from local SPIRE Agent via Workload API
  → Agent verifies: is this process really payment-service? (checks k8s ServiceAccount, UID, etc.)
  → Agent requests signature from SPIRE Server
  → Service receives x509 SVID
  → uses it for mTLS with other services

Certificate rotates automatically before expiry — service has no awareness of it
```

### Compared to OAuth/OIDC

```
                  OAuth / OIDC            SPIFFE / SPIRE
Subject           human user              service / workload
Credential        access_token (JWT)      x509 certificate (SVID)
Transport         HTTP Authorization      mTLS
Issuer            Keycloak / Auth0        SPIRE Server
Lifetime          minutes to hours        minutes to hours, auto-rotated
Revocation        ban token at AS         certificate expires; no renewal = no access
```
