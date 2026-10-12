# Northwind Platform API: Webhooks

> Sample documentation for a fictional product, used to exercise the RAG pipeline.

## Overview

Webhooks push events to your HTTPS endpoint as they happen, so you don't have to poll.
Create endpoints under **Settings > Webhooks** or with `POST /v2/webhook_endpoints`
(requires the `webhooks:manage` scope). Each workspace can have up to 16 endpoints, and
each endpoint subscribes to a list of event types such as `order.created`,
`order.cancelled`, `customer.updated`, or the wildcard `*`.

## Delivery

Events are sent as `POST` requests with a JSON body. Your endpoint must respond with any
2xx status within 10 seconds; anything else counts as a failure. Return 2xx as soon as the
event is stored and do the processing asynchronously.

Failed deliveries are retried with exponential backoff for up to 3 days (about 15
attempts). After 3 days of continuous failure the endpoint is disabled and workspace owners
are emailed. Re-enable it from the dashboard; events from the disabled period can be
replayed with `POST /v2/events/{id}/redeliver` for up to 30 days.

Delivery is at-least-once and unordered. Use the event `id` to deduplicate and the
`created` timestamp to order events.

## Verifying signatures

Every delivery includes a `Northwind-Signature` header of the form:

```
Northwind-Signature: t=1718035200,v1=5257a869e7ecebeda32affa62cdca3fa51cad7e77a0e56ff536d0ce8e108d8bd
```

To verify it:

1. Split the header on `,` to get the timestamp `t` and the signature `v1`.
2. Build the signed payload by concatenating `t`, a period `.`, and the raw request body
   (exactly as received, before JSON parsing).
3. Compute an HMAC-SHA256 of the signed payload using the endpoint's signing secret
   (prefix `whsec_`) as the key.
4. Compare your result to `v1` using a constant-time comparison.
5. Reject the event if `t` is more than 5 minutes from your current time, to prevent
   replay attacks.

Signing secrets can be rolled from the endpoint's settings page. During a roll, both the
old and new secrets sign each delivery for 24 hours, and the header carries two `v1` values.

## Testing locally

The Northwind CLI forwards events to a local server:

```
northwind listen --forward-to localhost:3000/webhooks
```

The CLI prints a temporary signing secret to use while forwarding. Trigger a test event with
`northwind trigger order.created`.
