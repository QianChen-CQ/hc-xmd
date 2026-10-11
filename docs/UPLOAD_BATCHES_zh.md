# 网页上传顺序

先解压 GitHub 上传包，进入 `hc-xmd`。上传下面各批，保持目录层级；每批上传完成后点击 Commit changes。网页拖拽不会读取 `.gitignore`，不要在运行依赖下载之后从实验目录手工上传。

| 批次 | 拖入内容 | 文件数 |
| --- | --- | --- |
| 1 | (root files) | 11 |
| 2 | common | 9 |
| 3 | docs | 3 |
| 4 | gemnet | 8 |
| 5 | phase1 | 8 |
| 6 | phase3_author | 11 |
| 7 | postprocessing | 37 |
| 8 | tools | 2 |
| 9 | vendor | 90 |

根目录文件包含 README.md、README_zh.md、LICENSE、THIRD_PARTY_NOTICES.md、requirements.txt、run_hcxmd.sh、suite.py、pack_feedback.py、OPTIMALER_MANIFEST.json 和 .gitignore。`SHA256SUMS.txt` 也一并上传。

只上传目录里的文件，不要把 `hc-xmd` 外层文件夹再套一层。最大顶层目录也少于100个文件。不要上传服务器完整备份。

GitHub网页单文件限制25 MiB；本包所有文件均低于此限。正式规则见 https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository 。
