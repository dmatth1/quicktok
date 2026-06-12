"""Alias so `python -m quicktok.import_tokenizer` works too (same CLI as
`python -m quicktok.importer`)."""
from .importer import main

if __name__ == "__main__":
    main()
