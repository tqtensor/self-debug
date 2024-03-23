import json
import os
from typing import List

import pandas as pd
import tiktoken as tk
from tqdm.auto import tqdm

from src.dataset import Dataset

encoding = tk.encoding_for_model("gpt-3.5-turbo-0613")


def retrieve(
    index: pd.DataFrame, post_id: int, max_token: int, min_comments: int, step: int
) -> List[str]:
    post_template = """
    Post: {post}

    Comments:
    """

    # Retrieve the post and comments
    post = json.load(
        open(os.path.join("stack_overflow", "posts", str(post_id) + ".json"), "r")
    ).get("Body", "")
    comments = []
    for comment_id in index[index["post_id"] == post_id]["comment_id"]:
        comment = json.load(
            open(
                os.path.join(
                    "stack_overflow",
                    "comments",
                    str(comment_id) + "_" + str(post_id) + ".json",
                ),
                "r",
            )
        ).get("Text", "")
        comments.append(comment)
    if len(comments) < min_comments:
        return [""]

    # Count the number of tokens from post and comments
    post_tokens = len(encoding.encode(post))
    if post_tokens > max_token:
        return [""]
    comments_tokens = [len(encoding.encode(comment)) for comment in comments]

    rendered_posts = []

    # Outer loop: iterate over the comments
    for i in range(0, len(comments_tokens), step):
        # Initialize the total number of tokens with the number of tokens in the post
        total_tokens = post_tokens

        # Initialize the list of comments for this post
        post_comments = []

        # Inner loop: try to add as many subsequent comments as possible
        for comment, comment_tokens in zip(comments[i:], comments_tokens[i:]):
            if total_tokens + comment_tokens > max_token:
                break
            total_tokens += comment_tokens
            post_comments.append("- comment: " + comment)

        # Render the post and comments, and add it to the list of rendered posts
        rendered_posts.append(
            post_template.format(post=post) + "\n".join(post_comments)
        )

    # Return the list of rendered posts
    return rendered_posts


if __name__ == "__main__":
    # Initialize the dataset
    stack_overflow_dataset = Dataset(
        dataset="stack_overflow", kwargs={"download": False}
    ).dataset

    # Generate a list of rendered posts
    if not os.path.exists("stack_overflow/rendered_posts"):
        os.makedirs("stack_overflow/rendered_posts")

    post_ids = stack_overflow_dataset.index["post_id"].unique()
    for post_id in tqdm(post_ids, total=len(post_ids)):
        rendered_posts = retrieve(
            index=stack_overflow_dataset.index,
            post_id=post_id,
            max_token=3000,
            min_comments=10,
            step=5,
        )
        if rendered_posts[0] != "":
            for i, rendered_post in enumerate(rendered_posts):
                with open(
                    os.path.join(
                        "stack_overflow", "rendered_posts", f"{post_id}_{i}.txt"
                    ),
                    "w",
                ) as f:
                    f.write(rendered_post)
