# README for Alpaca 7B

## Summary

TODO

## Payload

```
{
    "instruction": "What is the...",
    "temperature": 0.1,
    "top_p": 0.75,
    "top_k": 40,
    "num_beams": 4,
    "max_new_tokens": 128
}
```

## Response

```
{
    "text": "The number one...",
    "msg": "Success"
}
```

## Caller Example

TODO


## Notes

 - Requires a certain amount of memory: mem=12000,mem_limit=16000