import numpy as np
from rorycommon import StorageBuilder,StorageParams
from rory.core.security.dataowner import DataOwner
from rory.core.enums import Scheme, Algorithm
from rory.core.security.scheme_params import LiuParams,CkksParams


do = DataOwner.with_algorithm(Algorithm.SKMEANS)\
    .with_scheme(Scheme.LIU).\
    with_scheme_params(
        LiuParams(
            security_level = 128,
            _round         = False,
            decimals       = 2,
            secure_random  = False,
            seed           = None,
            use_np_random  = False,
            output_path    = "/rory/keys",
            save           = False
        )
    )\
    .build()

sb = StorageBuilder()\
    .with_dataowner(do)\
    .with_storage_path("/rory/data")\
    .build()

async def main(bucket_id="test_bucket", ball_id="test_key"):
    np.random.seed(42)  # For reproducibility
    X                       = np.random.rand(10, 2)                                                                                                                                    # Example data to store
    plaintext_stored_result = await sb.put(bucket_id=bucket_id, ball_id=f"{ball_id}_pt", data=X, delete=True, encrypt=False, segment=True, tags={"tag1": "value1", "tag2": "value2"})
    print(plaintext_stored_result)

    result = await sb.put(bucket_id=bucket_id, ball_id=ball_id, data=X,delete=True,encrypt=True,segment=True,tags={"tag1": "value1", "tag2": "value2"})
    print(result)

    result = await sb.get(bucket_id=bucket_id, ball_id=ball_id, encrypt=True, segment=True)
    if result.is_err:
        print(f"Error retrieving data: {result.unwrap_err()}")

    data           = result.unwrap()
    decrypted_data = do.primary_scheme.decrypt_matrix(data.raw_value)
    print("Decrypted data:", decrypted_data.data)
    # Get plaintext data to verify correctness
    result = await sb.get(bucket_id=bucket_id, ball_id=f"{ball_id}_pt", encrypt=False, segment=True)
    print(result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main(bucket_id="test_bucketliu", ball_id="test_keyliu"))
