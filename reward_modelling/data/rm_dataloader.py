from typing import Callable, Iterator, Dict, Literal, Optional, NamedTuple, Tuple
import datasets
import jax


class ProcessedTextBatch(NamedTuple):
    """
    Default output from HF tokenizer isn't a valid JAX tree.
    """

    input_ids: jax.numpy.ndarray
    attention_mask: jax.numpy.ndarray


NumBatches = int


class PairwiseBatch(NamedTuple):
    chosen: ProcessedTextBatch
    rejected: ProcessedTextBatch


def get_pairwise_dataloader(
    dataset: datasets.Dataset,
    tokenizer: Callable,
    sep_token: str,
    batch_size: int,
    block_size: int,  # max num tokens per sequence
    num_epochs: float = 1.0,
    prng_key: Optional[jax.random.PRNGKeyArray] = None,
) -> Tuple[NumBatches, Callable[[], Iterator[PairwiseBatch]]]:
    """
    Yield pairwise examples.
    """
    assert not isinstance(
        dataset, datasets.DatasetDict
    ), 'Must select a particular dataset split (e.g., "train")'
    assert isinstance(dataset, datasets.Dataset)

    # Leftover items would be discarded
    num_batches = len(dataset) // batch_size
    num_batches_with_repeat = int((num_epochs * len(dataset)) // batch_size)
    indices = jax.numpy.arange(len(dataset))

    if prng_key is None:
        shuffled_indices = indices.tolist()
    else:
        shuffled_indices = jax.random.permutation(prng_key, indices).tolist()

    def concatenate_and_tokenize(prefixes, texts) -> ProcessedTextBatch:
        concatenated = []
        for prefix, text in zip(prefixes, texts):
            concatenated.append(prefix + sep_token + text)

        tokenizer_output = tokenizer(
            concatenated,
            return_tensors="jax",
            max_length=block_size,
            padding="max_length",
            truncation=True,
        )

        return ProcessedTextBatch(
            input_ids=tokenizer_output.input_ids,
            attention_mask=tokenizer_output.attention_mask,
        )

    def _initialize_dataloader() -> Iterator[PairwiseBatch]:
        for batch_index in range(num_batches_with_repeat):
            first = batch_size * (batch_index % num_batches)
            last = first + batch_size
            batch_indices = shuffled_indices[first:last]
            data = dataset[batch_indices]

            yield PairwiseBatch(
                chosen=concatenate_and_tokenize(data["prompt"], data["chosen"]),
                rejected=concatenate_and_tokenize(data["prompt"], data["rejected"]),
            )

    return num_batches, _initialize_dataloader
