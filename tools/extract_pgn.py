import zstandard as zstd


input_file = "data/pgn/lichess_2013_01.pgn.zst"
output_file = "data/pgn/lichess_2013_01.pgn"


print("Extracting PGN...")


with open(input_file, "rb") as compressed:
    dctx = zstd.ZstdDecompressor()

    with open(output_file, "wb") as output:

        dctx.copy_stream(
            compressed,
            output
        )


print("Extraction complete.")
print(
    "Output:",
    output_file
)