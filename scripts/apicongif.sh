#!/usr/bin/env bash

curl -sS --fail -X POST "https://pce5pjhmkeejckomjamehdjeekom5bhb.openapi-qb.sii.edu.cn/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer QF8XqjmZhi9sygq1MT9Nr0rk8xZzVlid/aLvCRurzWw=" \
  -d '{
  "model": "GLM-5.1-FP8",
  "messages": [
    {
      "role": "user",
      "content": "hi"
    }
  ]
}'
