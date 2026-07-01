# DATA OCR
Cấu trúc data ocr như sau : 
```
data/
└── processed/
    └── ocr/
        ├── L21/
        │   ├── L21_V001.json
        │   ├── L21_V002.json
        │   └── ...
        ├── L22/
        │   ├── L22_V001.json
        │   └── ...
        └── ...

L21_V001.json
{
    "<frame_id>": [
        [
            [x1, y1, x2, y2],
            [x1, y1, x2, y2],
            ...
        ],
        [
            "text1",
            "text2",
            ...
        ]
    ],

    "<frame_id>": [
        [
            [x1, y1, x2, y2],
            ...
        ],
        [
            "text1",
            ...
        ]
    ]
}
```