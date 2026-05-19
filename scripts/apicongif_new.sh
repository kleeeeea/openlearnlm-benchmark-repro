#!/usr/bin/env bash

curl -sS --fail -X POST "https://g8gqe9chpgmkc9hbhjb9bqj59eejkh9o.openapi-qb.sii.edu.cn/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 3QU6C28LVpeylUHaFSWgjWywHOa02E3ywTY1RXPQGWM=" \
  -d '{
  "model": "Kimi-K2_5",
  "messages": [
    {
      "role": "user",
      "content": "hi"
    }
  ]
}'
