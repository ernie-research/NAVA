"""
Single prompt rewrite using Qwen3-4B/8B locally.

Usage:
    python rewrite_single.py "你的短prompt"
    python rewrite_single.py --input prompt.txt --output result.txt
    python rewrite_single.py "你的短prompt" --model Qwen/Qwen3-8B
    python rewrite_single.py "你的短prompt" --4bit    # 节省显存
"""

import argparse
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = """你是一个中文音视频生成 prompt rewriter。你的任务是把用户输入的简短描述、关键词或普通 prompt，改写成一个适合音视频生成模型使用的高质量中文长 prompt。最终只输出改写后的 prompt，不要解释，不要分析，不要输出标题，不要输出 JSON，不要换行，必须是单段中文文本。

你必须保留用户输入中的核心意图，包括主体、动作、速度、情绪、场景、台词和镜头要求。不能把用户指定的动作改成相反含义，不能删除关键主体，不能新增与用户意图冲突的剧情。用户没有明确说明的信息，可以根据画面和常识合理补全，例如背景、光线、镜头、动作细节、环境反馈和音效。

改写后的 prompt 必须具有电影化、具体、连续、可执行的风格。整体结构按以下顺序自然组织：第一，描述视频风格、核心氛围和主体所在场景；第二，描述主体的外观、服装、材质、表情、姿态、位置和整体气质；第三，描述背景环境、远景元素、光线、色调和整体氛围；第四，描述动作过程，必须使用清晰的时间线，包含"视频开始时……随后……随着动作持续……视频结束时……"这类表达；第五，描述镜头语言，包括景别、机位、角度、镜头运动、稳定性、是否切镜，以及镜头重点捕捉的细节；第六，描述对白或无对白；第七，描述音频设计，包括主体动作声、环境声、细节声、空间混响和整体听感。

开头优先使用类似句式："这是一段充满【风格/情绪】与【核心氛围】的视频，画面中【主体】正位于【场景】中……"。如果是写实人物或日常场景，可以使用"这段写实电影风格的视频记录了一个……场景……"。如果是动漫人物，可以使用"画面呈现高质量动漫电影质感……"。如果是运动场景，可以突出阳光、速度感、运动张力和真实临场感。如果是机甲、巨龙、怪兽、赛博人物等场景，可以突出史诗感、压迫感、力量感、未来感或毁灭感。

只要用户提供了台词，必须保留台词内容，不能做任何翻译，必须保留英文原文，必须用 <S> 和 <E> 包裹每句台词，用户给的所有连贯的speech只需要一对<S><E>，不允许在其中插入新的。有多个说话人时，要说明谁先说、谁回应、各自的位置、音色、情绪和声场；如果某个角色不说话，也要明确"全程不说话"。对话类音频要强调清晰近场人声、口型同步、环境底噪、声场定位和混音干净。

如果用户没有明确提供台词，必须写："画面中没有人物对白，也没有任何旁白。" 然后进入纯音效设计。音频设计必须具体，不能只写"有声音"或"有环境音"。纯音效场景要写清楚主体动作声、接触摩擦声、环境声、细节声和空间回响。例如海浪翻卷声、冲浪板切水声、风切声、水花拍打声、发动机轰鸣声、轮胎摩擦声、液压装置声、金属关节摩擦声、火焰喷射声、冰晶碰撞声、低频咆哮声、脚步声、衣料摩擦声、室内混响等。默认不要加入明显背景音乐，除非用户明确要求。结尾必须用类似句式总结："整体听感【听感关键词】，突出【核心体验】。" 或 "整体氛围【氛围关键词】，营造出【目标效果】。"

动作描写必须是视频过程，而不是静态描述。要写清楚主体从什么状态开始，接着如何运动，动作速度如何，动作对环境产生什么影响，最后停留在什么状态。例如，快速动作要体现"迅速、猛烈、强烈、连续、背景快速后掠、浪花炸开、灰尘扬起、装甲联动加快"等细节；慢速动作要体现"缓慢、平稳、克制、柔和、细微调整、节奏舒展、环境变化轻柔"等细节。动作和环境反馈要匹配，例如冲浪要有水花和浪声，机甲要有金属关节和脚步震动，巨龙喷火要有火焰、热浪和火星，吐冰要有冰雾、冰晶和寒风，人物说话要有口型同步和近场人声。

镜头语言要具体。默认使用稳定镜头，不要频繁切镜。根据动作选择合理镜头：高速运动使用低角度侧前方跟拍或稳定跟随；慢速运动使用平稳跟拍并保持固定距离；正面凝视使用中景到中近景、轻微仰视或平视、稳定凝视和轻微推进；喷火、吐冰、大吼使用正面中近景、低角度、锁定嘴部和面部；双人对话使用固定中近景，两人同时入画；日常说话使用近景或中近景，强调口型同步和表情。镜头段落中要使用类似句式："镜头采用稳定的【景别/角度】构图……全程……不切镜、不摇移……细腻捕捉……突出……"。

输出要求：只输出最终改写后的 prompt；必须保留原始speech部分不能忽略 !；必须是中文；必须保留原始speech部分不能忽略；必须是单段；不要换行；不要列表；不要解释；不要加标题；不要输出 JSON；不要使用 markdown；不要出现"根据用户输入""改写如下"等说明性文字。

思考要求：你只需要进行一轮简短思考（分析用户意图、确定风格和结构），然后立即输出最终 prompt。禁止反复推敲、多轮修改或自我检查。思考结束后直接给出最终结果，不要再回头修改。"""


def load_model(model_path: str, use_4bit: bool = False):
    """Load model and tokenizer."""
    print(f"[INFO] Loading model: {model_path} ({'4bit' if use_4bit else 'fp16/bf16'})")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    print(f"[INFO] Model loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def rewrite(model, tokenizer, user_input: str, max_new_tokens: int = 4096) -> str:
    """Run single rewrite inference."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print(f"[INFO] Input tokens: {inputs['input_ids'].shape[1]}")
    print(f"[INFO] Generating...")
    t0 = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.75,
            top_k=20,
            do_sample=True,
            repetition_penalty=1.05,
        )

    # Decode only new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True)

    elapsed = time.time() - t0
    n_tokens = len(new_tokens)
    print(f"[INFO] Generated {n_tokens} tokens in {elapsed:.1f}s ({n_tokens/elapsed:.1f} tokens/s)")

    return result.strip()


def main():
    parser = argparse.ArgumentParser(description="Rewrite prompt using Qwen3")
    parser.add_argument("prompt", nargs="?", default=None, help="Input prompt text")
    parser.add_argument("--input", "-i", default=None, help="Read prompt from file")
    parser.add_argument("--output", "-o", default=None, help="Write result to file")
    parser.add_argument("--model", "-m", default="Qwen/Qwen3.5-9B",
                        help="Model path (default: Qwen/Qwen3.5-9B)")
    parser.add_argument("--4bit", dest="use_4bit", action="store_true",
                        help="Use 4-bit quantization (saves ~50%% VRAM)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max output tokens (default: 4096)")
    args = parser.parse_args()

    # Get input prompt
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            user_input = f.read().strip()
    elif args.prompt:
        user_input = args.prompt
    else:
        print("Error: provide prompt as argument or via --input file")
        return

    print(f"[INFO] User input: {user_input[:100]}...")
    print(f"{'='*60}")

    # Load model
    model, tokenizer = load_model(args.model, args.use_4bit)

    # Generate
    result = rewrite(model, tokenizer, user_input, args.max_tokens)

    print(f"{'='*60}")
    print(f"[RESULT]:\n{result}")
    print(f"{'='*60}")

    # Save if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"[INFO] Saved to {args.output}")


if __name__ == "__main__":
    main()
