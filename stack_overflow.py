import glob
import json
import logging
import os
import sys
import xml.etree.ElementTree as ET
from typing import List, Optional

import psutil

# Create a logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def get_posts(file_name: str, num_posts: Optional[int] = None, tags: List[str] = [""]):
    if not os.path.exists("stack_overflow/posts"):
        os.makedirs("stack_overflow/posts")

    num_posts = float("inf") if num_posts is None else num_posts
    posts_count = 1

    xmlit = iter(
        ET.iterparse(
            source=file_name,
            events=("start", "end"),
        )
    )
    event, _ = next(xmlit)
    while True:
        try:
            event, elem = next(xmlit)
            if event == "end" and elem.tag == "row":
                post = elem.attrib
                post_tags = post.get("Tags", "")
                if any(tag in post_tags for tag in tags):
                    posts_count += 1
                    # Save post to file
                    if not os.path.exists(
                        os.path.join(
                            "stack_overflow", "posts", post.get("Id", 1) + ".json"
                        )
                    ):
                        with open(
                            os.path.join(
                                "stack_overflow", "posts", post.get("Id", 1) + ".json"
                            ),
                            "w",
                        ) as f:
                            json.dump(post, f, indent=4)

                    if posts_count % 1000 == 0:
                        process = psutil.Process(os.getpid())
                        mem_info = process.memory_info()
                        logger.info(
                            f"Found {posts_count} posts. Current memory usage: {mem_info.rss / 1024 ** 2} MB"
                        )

                if posts_count >= num_posts:
                    logger.info("Reached the desired number of posts")
                    break

                # Clear the element to free up memory
                elem.clear()
        except ET.ParseError:
            logger.warning("ParseError occurred, skipping to next row")
            continue
        except StopIteration:
            logger.info(elem.attrib)
            logger.info("StopIteration occurred")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            break

    logger.info("Completed!")


def get_comments(file_name: str, post_ids: set):
    if not os.path.exists("stack_overflow/comments"):
        os.makedirs("stack_overflow/comments")

    comments_count = 1

    xmlit = iter(
        ET.iterparse(
            source=file_name,
            events=("start", "end"),
        )
    )
    event, _ = next(xmlit)
    while True:
        try:
            event, elem = next(xmlit)
            if event == "end" and elem.tag == "row":
                comment = elem.attrib
                if comment.get("PostId", 1) in post_ids:
                    comments_count += 1

                    # Save comment to file
                    if not os.path.exists(
                        os.path.join(
                            "stack_overflow",
                            "comments",
                            comment.get("Id", 1)
                            + "_"
                            + comment.get("PostId", 1)
                            + ".json",
                        )
                    ):
                        with open(
                            os.path.join(
                                "stack_overflow",
                                "comments",
                                comment.get("Id", 1)
                                + "_"
                                + comment.get("PostId", 1)
                                + ".json",
                            ),
                            "w",
                        ) as f:
                            json.dump(comment, f, indent=4)

                    if comments_count % 1000 == 0:
                        process = psutil.Process(os.getpid())
                        mem_info = process.memory_info()
                        logger.info(
                            f"Found {comments_count} comments. Current memory usage: {mem_info.rss / 1024 ** 2} MB"
                        )

                # Clear the element to free up memory
                elem.clear()
        except ET.ParseError:
            logger.warning("ParseError occurred, skipping to next row")
            continue
        except StopIteration:
            logger.info(elem.attrib)
            logger.info("StopIteration occurred")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            break

    logger.info("Completed!")


if __name__ == "__main__":
    # Get all related posts
    get_posts(
        file_name="Posts.xml",
        tags=[
            "matplotlib",
            "pandas",
            "numpy",
            "scipy",
            "seaborn",
            "sklearn",
            "tensorflow",
            "pytorch",
        ],
    )

    # Get list of PostId
    post_ids = set(
        [
            os.path.basename(file).replace(".json", "")
            for file in list(glob.glob("stack_overflow/posts/*.json"))
        ]
    )

    # Get all comments
    get_comments(file_name="Comments.xml", post_ids=post_ids)
