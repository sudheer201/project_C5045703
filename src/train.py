import os
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from src.utils import (
    load_config,
    prepare_dataset,
    make_tf_dataset,
    ensure_dir,
    distinct_n,
    compute_bleu,
    compute_meteor
)
from src.model import build_multimodal_model


def collate_for_training(batch_images, batch_input_ids, bos_id=101):
    """
    Prepare dec_input and dec_target for teacher-forcing training.
    batch_images: (B, seq_len, H, W, C)
    batch_input_ids: (B, seq_len, T)
    Returns: images, captions_seq, dec_input, dec_target
    """
    target = batch_input_ids[:, -1, :]  # last caption (B, T)

    dec_input = np.concatenate(
        [
            np.full((target.shape[0], 1), bos_id, dtype=np.int32),
            target[:, :-1]
        ],
        axis=1
    )

    return batch_images, batch_input_ids, dec_input, target


def greedy_decode(model, images, captions_seq, cfg):
    """
    Greedy decoding for evaluation.
    """
    max_len = cfg['dataset']['max_caption_len']
    bos_id = cfg['model']['bos_token_id']
    pad_id = cfg['model']['pad_token_id']

    dec_input = np.full((images.shape[0], max_len), pad_id, dtype=np.int32)
    dec_input[:, 0] = bos_id

    for t in range(max_len - 1):
        logits, _ = model([images, captions_seq, dec_input], training=False)
        next_token = np.argmax(logits[:, t, :], axis=-1)
        dec_input[:, t+1] = next_token

    return dec_input


def train(cfg, keep_small=False):
    # ===============================
    # 1) Load training data
    # ===============================
    print("Loading training dataset...")
    train_processed, tokenizer = prepare_dataset(cfg, split="train", keep_small=keep_small)
    train_ds = make_tf_dataset(train_processed, cfg, shuffle=True)
    print("Training examples:", len(train_processed))

    # ===============================
    # 2) Load validation data ONCE
    # ===============================
    print("Loading validation dataset...")
    val_processed, _ = prepare_dataset(cfg, split="test", keep_small=True)
    val_ds = make_tf_dataset(val_processed, cfg, shuffle=False).take(1)
    print("Validation examples:", len(val_processed))

    # ===============================
    # 3) Build model
    # ===============================
    models = build_multimodal_model(cfg)
    model = models["full_model"]
    visual_enc = models["visual_enc"]

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=float(cfg["training"].get("lr", 1e-4))
    )

    # ===============================
    # 4) Checkpoint directory
    # ===============================
    ensure_dir(cfg["training"]["save_dir"])
    ckpt_prefix = os.path.join(cfg["training"]["save_dir"], "ckpt")

    # ===============================
    # 5) Training step
    # ===============================
    def train_step(images, captions_seq, dec_input, dec_target):
        with tf.GradientTape() as tape:
            logits, img_pred = model(
                [images, captions_seq, dec_input], training=True
            )

            # ---- Text loss ----
            per_token_loss = tf.keras.losses.sparse_categorical_crossentropy(
                dec_target, logits, from_logits=True
            )

            pad_id = cfg["model"]["pad_token_id"]
            mask = tf.cast(tf.not_equal(dec_target, pad_id), tf.float32)
            loss_text = tf.reduce_sum(per_token_loss * mask) / (
                tf.reduce_sum(mask) + 1e-8
            )

            # ---- Image feature loss ----
            last_images = images[:, -1]
            target_feat = tf.stop_gradient(
                visual_enc(last_images, training=False)
            )
            loss_img = tf.reduce_mean(tf.square(target_feat - img_pred))

            loss = loss_text + 0.5 * loss_img

        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(
            grads, cfg["training"].get("grad_clip", 1.0)
        )
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss_text.numpy(), loss_img.numpy(), loss.numpy()

    # ===============================
    # 6) Training loop
    # ===============================
    epochs = int(cfg["training"]["epochs"])

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        prog = tqdm(train_ds)

        for step, (images_batch, input_ids_batch) in enumerate(prog):
            imgs, caps_seq, dec_inp, dec_tgt = collate_for_training(
                images_batch.numpy(),
                input_ids_batch.numpy(),
                bos_id=cfg["model"]["bos_token_id"]
            )

            loss_text, loss_img, loss = train_step(
                imgs, caps_seq, dec_inp, dec_tgt
            )

            if step % cfg["training"]["log_interval"] == 0:
                prog.set_description(
                    f"loss={loss:.4f} text={loss_text:.4f} img={loss_img:.4f}"
                )

            if step >= 3:  # limit to 4 steps per epoch
                break

        # ===============================
        # 7) Save checkpoint
        # ===============================
        ckpt_path = f"{ckpt_prefix}_epoch{epoch+1}.weights.h5"
        model.save_weights(ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # ===============================
        # 8) Validation + Metrics
        # ===============================
        all_preds, all_refs = [], []

        for images_batch, input_ids_batch in val_ds:
            images = images_batch.numpy()
            captions_seq = input_ids_batch.numpy()

            pred_ids = greedy_decode(model, images, captions_seq, cfg)

            for i in range(pred_ids.shape[0]):
                pred_text = tokenizer.decode(
                    pred_ids[i], skip_special_tokens=True
                )
                ref_text = tokenizer.decode(
                    captions_seq[i, -1], skip_special_tokens=True
                )
                all_preds.append(pred_text)
                all_refs.append(ref_text)

        dist1 = distinct_n(all_preds, 1)
        dist2 = distinct_n(all_preds, 2)
        bleu = np.mean(
            [compute_bleu(r, p) for r, p in zip(all_refs, all_preds)]
        )
        meteor = np.mean(
            [compute_meteor(r, p) for r, p in zip(all_refs, all_preds)]
        )

        print(
            f"[Validation] "
            f"Dist-1={dist1:.4f} "
            f"Dist-2={dist2:.4f} "
            f"BLEU={bleu:.4f} "
            f"METEOR={meteor:.4f}"
        )

    print("\nTraining complete.")
    return model, models, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--small", action="store_true", help="Use small subset")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, keep_small=args.small)
