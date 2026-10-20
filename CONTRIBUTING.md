# BEAM CellReasoner 合作开发须知

## 基本约束

- 禁止直接向 `main` push；所有改动通过 Pull Request 合入。
- 每个 PR 必须由另一位成员 Review 并 Approve 后才能 merge。
- 一个功能或修复使用一个清晰命名的分支，例如
  `feature/wangyue-import-celltok`、`fix/name-issue`。
- 不提交模型权重、完整数据集、运行日志、缓存、密钥或个人环境文件。

## 初次配置

```bash
git config --global user.name "你的 GitHub 用户名"
git config --global user.email "你的 GitHub 邮箱"

git clone git@github.com:BruthYU/BEAM_CellReasoner.git
cd BEAM_CellReasoner
git switch -c feature/name-topic
```

如需在 worker 上配置 SSH：

```bash
ssh-keygen -t ed25519 -C "你的 GitHub 邮箱"
cat ~/.ssh/id_ed25519.pub
ssh -T git@github.com
```

将公钥添加到 GitHub 的 `Settings -> SSH and GPG keys` 后再测试连接。

## 日常提交与 PR

```bash
git status
git add <本次改动的文件>
git commit -m "清晰描述本次改动"
git push -u origin feature/name-topic
```

随后在 GitHub 上以该分支向 `main` 创建 PR，等待另一位成员审核。审核后继续
修改时，正常 `commit` 和 `push` 即会更新原 PR。

## 基于最新 main 开发

开始 rebase 前先提交或妥善保存当前改动：

```bash
git fetch origin
git rebase origin/main
```

若发生冲突：

```bash
git status
# 手工解决冲突并删除冲突标记
git add <已解决的文件>
git rebase --continue
```

已推送过的功能分支在 rebase 后使用：

```bash
git push --force-with-lease
```

不要使用 `--force` 覆盖远端分支。

## 提交前检查

至少确认：

```bash
git status --short
git diff --check
```

涉及论文数据时运行：

```bash
bash scripts/validate_paper_data.sh
```

涉及模型推理时，按照 `docs/GPFS2_RUN_GUIDE.md` 在 GPU worker 上运行 smoke test。
