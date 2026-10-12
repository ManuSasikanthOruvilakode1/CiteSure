# Northwind Platform API: Authentication

> Sample documentation for a fictional product, used to exercise the RAG pipeline.
> Replace the files in `data/documents/` with your own corpus.

## API keys

Every request to the Northwind API must include an API key in the `Authorization` header
using the Bearer scheme:

```
Authorization: Bearer nw_live_4f9a...
```

Keys come in two types. **Live keys** (prefix `nw_live_`) act on production data and are
billed. **Test keys** (prefix `nw_test_`) act on an isolated sandbox that is reset every
Sunday at 00:00 UTC; requests made with test keys are never billed.

Each workspace can hold at most 20 active keys. Keys are scoped to a single workspace and
cannot be shared across workspaces.

### Creating a key

Open the dashboard, go to **Settings > API Keys**, and click **Create key**. Choose a name,
a type (live or test), and optionally an expiry date. The full key is displayed exactly
once; Northwind stores only a salted hash, so a lost key cannot be recovered and must be
replaced.

### Rotating a key

To rotate a key without downtime, click **Rotate** next to it. Rotation issues a new key
immediately and keeps the old key valid for a 24-hour grace period, after which it is
revoked automatically. You can end the grace period early with **Revoke now**.

Rotate live keys at least every 90 days. Workspaces on the Enterprise plan can enforce a
maximum key age under **Settings > Security > Key policy**; keys older than the limit are
revoked automatically and the workspace owners receive an email seven days beforehand.

### Revoking a key

Revocation takes effect within 60 seconds across all regions. Requests made with a revoked
key fail with HTTP 401 and the error code `key_revoked`.

## Scopes

Keys can be restricted to a set of scopes. A key with no scopes selected has full access.

| Scope | Grants |
|---|---|
| `orders:read` | List and retrieve orders |
| `orders:write` | Create, update, and cancel orders |
| `customers:read` | List and retrieve customers |
| `customers:write` | Create and update customers |
| `webhooks:manage` | Create, update, and delete webhook endpoints |

A request that needs a scope the key lacks fails with HTTP 403 and the error code
`insufficient_scope`. The response body lists the missing scope in `error.required_scope`.

## OAuth 2.0 for third-party apps

Applications that act on behalf of other Northwind workspaces must use OAuth 2.0 with the
authorization code flow and PKCE. Access tokens are valid for 1 hour; refresh tokens are
valid for 60 days and are single-use, so every refresh returns a new refresh token.
The authorization endpoint is `https://auth.northwind.example/oauth/authorize` and the
token endpoint is `https://auth.northwind.example/oauth/token`.
