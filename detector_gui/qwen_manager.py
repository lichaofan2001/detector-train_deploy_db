import ast
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

DEFAULT_SYSTEM_PROMPT = 'Locate every instance that belongs to the following categories: "military vehicle", Report bbox coordinates in JSON format.'

class Qwen35Manager:

    def __init__(self, model_path: str):

        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_path)

    def inference(self, image, prompt: str = None, min_pixels=64 * 32 * 32, max_pixels=9800 * 32 * 32, write2file = False):
        import torch
        prompt = prompt or DEFAULT_SYSTEM_PROMPT

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = inputs.to('cuda')

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ) # 模型输出的内容是output_text，如果想直接导出的话就直接print(output_text[0])

        # 清理GPU内存
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

        bboxes = self._output2bbox(image, output_text[0], is_yolo=True) # 将output_text的内容

        if write2file is True:
            self._write2file(image, bboxes)

        # 处理解析失败的情况
        if bboxes is None:
            return []

        bboxes_float = [list(map(float, row)) for row in bboxes]
        return bboxes_float

    def _output2bbox(self, image, output_text, is_yolo=False):

        bounding_boxes = self._parse_json(output_text)

        try:
            json_output = ast.literal_eval(bounding_boxes)
        except Exception as e:
            # end_idx = bounding_boxes.rfind('"}') + len('"}')
            # truncated_text = bounding_boxes[:end_idx] + "]"
            # json_output = ast.literal_eval(truncated_text)
            print(f"the image {image} can't be inferred correctly.")
            return

        if not isinstance(json_output, list):
            json_output = [json_output]

        # Iterate over the bounding boxes
        bboxes = []
        for i, bounding_box in enumerate(json_output):
            bbox = bounding_box['bbox_2d']
            bboxes.append(bbox)

        if is_yolo:
            boxes = []
            for bbox in bboxes:

                x1 = float(bbox[0]) / 1000
                y1 = float(bbox[1]) / 1000
                x2 = float(bbox[2]) / 1000
                y2 = float(bbox[3]) / 1000

                x = round((x1 + x2) / 2, 6)
                y = round((y1 + y2) / 2, 6)
                w = round((x2 - x1), 6)
                h = round((y2 - y1), 6)

                str6_x = f"{x:.6f}"
                str6_y = f"{y:.6f}"
                str6_w = f"{w:.6f}"
                str6_h = f"{h:.6f}"

                boxes.append([str6_x, str6_y, str6_w, str6_h])

            bboxes = boxes

        return bboxes

    def _parse_json(self, json_output):
        # Parsing out the markdown fencing
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
                json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
                t1 = json_output[-2]
                t2 = json_output[-4]
                if json_output[-2] != "]" or json_output[-4] != "}":
                    num = json_output.rfind("\t")
                    json_output = json_output[:num-2] + "\n]\n"
                break  # Exit the loop once "```json" is found
        return json_output

    def _write2file(self, image, bboxes):

        label_folder = Path(image).parent / 'labels'
        label_folder.mkdir(parents=True, exist_ok=True)

        label_file = label_folder / (str(Path(image).stem) + '.txt')

        with open(label_file, 'w') as f:
            for bbox in bboxes:
                b = " ".join(map(str, bbox))
                f.write("-1 " + b + ' 1.000000 ' + '\n')
