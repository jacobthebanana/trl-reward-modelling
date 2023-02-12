from typing import Tuple, Any, Callable, Iterator, Optional
import argparse
import os


import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from optax import adamw, linear_schedule, sigmoid_binary_cross_entropy

from transformers import FlaxOPTForCausalLM, AutoTokenizer
import datasets

from tqdm.auto import tqdm
from socket import gethostname
import datetime
import wandb

from ..data.rm_dataloader import get_pairwise_dataloader, PairwiseBatch
from .partition_utils import get_sharding_scheme, device_put_leaf

Loss = jnp.ndarray
Accuracy = jnp.ndarray
LossFloat = float
AccuracyFloat = float
Params = Any
Gradients = Params

EVAL_EVERY: int = int(os.environ.get("EVAL_EVERY", 125))


def _get_reward_from_projection_logits(logits: jnp.ndarray):
    """
    Apply softmax over selected keyword input ids.
    For example, if the keyword is "Yes", the "reward" will be the logit of
    "Yes" after the word embedding projection.

    Input shapes:
    - logits: (batch, token)

    Output shape:
    (batch,)
    """

    return logits[:, -1, 0]


def loss_accuracy_fn(
    apply_fn: Callable, batch: PairwiseBatch, params
) -> Tuple[Loss, Accuracy]:
    """
    Reproduced from Dahoas/reward-modeling
    """
    chosen_logits = apply_fn(
        input_ids=batch.chosen.input_ids,
        attention_mask=batch.chosen.attention_mask,
        params=params,
    ).logits  # type: ignore
    chosen_rewards = _get_reward_from_projection_logits(chosen_logits)

    rejected_logits = apply_fn(
        input_ids=batch.rejected.input_ids,
        attention_mask=batch.rejected.attention_mask,
        params=params,
    ).logits  # type: ignore
    rejected_rewards = _get_reward_from_projection_logits(rejected_logits)

    loss = -jax.nn.log_sigmoid(chosen_rewards - rejected_rewards + 1e-6).mean()
    accuracy = jnp.mean(chosen_rewards > rejected_rewards)

    return loss, accuracy


def jit_loss_accuracy_fn(
    apply_fn: Callable, batch: PairwiseBatch, params
) -> Tuple[Loss, Accuracy]:
    ...


jit_loss_accuracy_fn = jax.jit(loss_accuracy_fn, static_argnames=["apply_fn"])


def grad_fn(
    apply_fn: Callable, batch: PairwiseBatch, params
) -> Tuple[Tuple[Loss, Accuracy], Gradients]:
    ...


grad_fn = jax.value_and_grad(loss_accuracy_fn, argnums=2, has_aux=True)


def train_step(
    state: TrainState, batch: PairwiseBatch
) -> Tuple[TrainState, Tuple[Loss, Accuracy]]:
    (loss, accuracy), gradients = grad_fn(state.apply_fn, batch, state.params)
    state = state.apply_gradients(grads=gradients)

    return state, (loss, accuracy)


def jit_train_step(
    state: TrainState, batch: PairwiseBatch
) -> Tuple[TrainState, Tuple[Loss, Accuracy]]:
    ...


jit_train_step = jax.jit(train_step)


def evaluate_reward_model(
    apply_fn: Callable,
    params: Params,
    eval_dataloader: Iterator[PairwiseBatch],
    keyword_token_id: int,
    num_eval_batches: Optional[int] = None,
) -> Tuple[LossFloat, AccuracyFloat]:
    """
    Evaluate model on the given eval dataloader.
    """
    loss_tally = 0.0
    accuracy_tally = 0.0
    num_batches = 0

    for batch in tqdm(eval_dataloader, total=num_eval_batches, ncols=80, leave=False):
        num_batches += 1
        loss, accuracy = jit_loss_accuracy_fn(model, batch, params)
        loss_tally += loss.item()
        accuracy_tally += accuracy.item()

    return loss_tally / num_batches, accuracy_tally / num_batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_hf_model", required=True)
    parser.add_argument(
        "--tokenizer",
        required=False,
        default=None,
        help="same as base_hf_model if not specified",
    )
    parser.add_argument("--hf_dataset_dict", required=True)
    parser.add_argument("--max_learning_rate", required=True, type=float)
    parser.add_argument("--train_batch_size", required=True, type=int)
    parser.add_argument("--train_block_size", required=True, type=int)
    parser.add_argument("--train_prng_seed", required=False, type=int, default=0)
    parser.add_argument("--num_epochs", required=False, type=float, default=1.0)
    args = parser.parse_args()

    base_hf_model_name: str = args.base_hf_model
    hf_dataset_dict: str = args.hf_dataset_dict
    max_learning_rate: float = args.max_learning_rate
    train_batch_size: int = args.train_batch_size
    block_size: int = args.train_block_size
    train_prng_seed: int = args.train_prng_seed
    num_epochs: float = args.num_epochs

    hf_tokenizer_name: str = args.tokenizer
    if hf_tokenizer_name is None:
        hf_tokenizer_name = base_hf_model_name

    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)

    dataset_dict = datasets.load_from_disk(hf_dataset_dict)
    assert isinstance(dataset_dict, datasets.DatasetDict)
    num_train_batches, init_train_dataloader = get_pairwise_dataloader(
        dataset_dict["train"],
        tokenizer,
        tokenizer.bos_token,
        train_batch_size,
        block_size,
        num_epochs,
        jax.random.PRNGKey(train_prng_seed),
    )
    num_eval_batches, init_eval_dataloader = get_pairwise_dataloader(
        dataset_dict["validation"],
        tokenizer,
        tokenizer.bos_token,
        train_batch_size,
        block_size,
    )

    model, params_cpu = FlaxOPTForCausalLM.from_pretrained(
        base_hf_model_name, _do_init=False
    )  # type: ignore
    assert isinstance(model, FlaxOPTForCausalLM)

    lr_schedule = linear_schedule(max_learning_rate, 0.0, num_train_batches)
    optimizer = adamw(lr_schedule, eps=1e-6, eps_root=1e-6)

    # Shard and initialize model parameters.
    sharding_scheme = get_sharding_scheme(params_cpu, num_replicas=1)
    initial_params = jax.tree_util.tree_map(
        device_put_leaf, params_cpu, sharding_scheme
    )
    random_state_prng_key = jax.random.PRNGKey(train_prng_seed)
    params = model.init_weights(random_state_prng_key, (1, 1), initial_params)
    opt_state = optimizer.init(params)
    train_state = TrainState(0, jax.jit(model.__call__), params, optimizer, opt_state)

    wandb_run_name = datetime.datetime.now().isoformat() + "-" + gethostname()
    wandb.init(project="reward-modelling", name=wandb_run_name)

    for batch in tqdm(
        init_train_dataloader(), total=num_train_batches, desc="Training", ncols=80
    ):
        stats = {}

        if train_state.step % EVAL_EVERY == 0:
            eval_loss, eval_accuracy = evaluate_reward_model(
                train_state.apply_fn,
                train_state.params,
                init_eval_dataloader(),
                num_eval_batches,
            )
            stats["validation_loss"] = eval_loss
            stats["validation_accuracy"] = eval_accuracy

        train_state, (loss, accuracy) = jit_train_step(train_state, batch)
        stats["train_loss"] = loss.item()
        stats["train_accuracy"] = accuracy.item()

        wandb.log(stats)

    # model.save_pretrained("data/artifacts/" + wandb_run_name, params)
