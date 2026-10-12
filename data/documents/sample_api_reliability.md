# Northwind Platform API: Rate Limits, Errors, and Retries

> Sample documentation for a fictional product, used to exercise the RAG pipeline.

## Rate limits

Limits are applied per API key using a token-bucket algorithm.

| Plan | Sustained rate | Burst |
|---|---|---|
| Free | 60 requests/minute | 20 |
| Growth | 600 requests/minute | 100 |
| Enterprise | 6,000 requests/minute | 1,000 |

Every response includes the headers `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and
`X-RateLimit-Reset` (a Unix timestamp in seconds). When a key exceeds its limit the API
returns HTTP 429 with the error code `rate_limited` and a `Retry-After` header giving the
number of seconds to wait.

Search endpoints (`/v2/search/*`) have a separate, lower limit of 10% of the plan's
sustained rate, because each search fans out across shards.

## Error format

Errors use a consistent JSON envelope:

```json
{
  "error": {
    "code": "invalid_parameter",
    "message": "limit must be between 1 and 100",
    "param": "limit",
    "request_id": "req_8c1d..."
  }
}
```

Always log `request_id`; support needs it to trace a request.

| HTTP status | Typical codes | Retry? |
|---|---|---|
| 400 | `invalid_parameter`, `invalid_json` | No, fix the request |
| 401 | `key_invalid`, `key_revoked` | No |
| 403 | `insufficient_scope` | No |
| 404 | `not_found` | No |
| 409 | `idempotency_conflict` | No, see idempotency below |
| 429 | `rate_limited` | Yes, after `Retry-After` |
| 500, 502, 503 | `internal_error`, `unavailable` | Yes, with backoff |

## Retries and backoff

Retry 429 and 5xx responses, and network errors, using exponential backoff with full
jitter: wait a random duration between 0 and `min(cap, base * 2^attempt)` seconds, with
`base = 0.5` and `cap = 30`. Stop after 5 attempts. When a `Retry-After` header is present,
wait at least that long. The official SDKs implement this policy by default; configure it
with the `max_retries` client option.

## Idempotency

`POST` requests accept an `Idempotency-Key` header (any unique string up to 255
characters, a UUIDv4 is recommended). If a request with the same key is retried within 24
hours, the API returns the original response instead of performing the operation again.
Reusing a key with a different request body returns HTTP 409 `idempotency_conflict`.
`GET`, `PUT`, and `DELETE` are idempotent by definition and ignore the header.

## Pagination

List endpoints use cursor pagination. Pass `limit` (1 to 100, default 20) and, for
subsequent pages, the `next_cursor` value from the previous response as `cursor`. When
`has_more` is `false` there are no further pages. Cursors expire after 10 minutes; an
expired cursor returns HTTP 400 `cursor_expired`, and the listing must be restarted.
