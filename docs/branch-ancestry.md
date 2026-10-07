# Canonical branch ancestry

The compatibility branch was created from `f555d81`, while the OpenSpiel branch
advanced to `57872b4`. Commit `85f0ab1` carries the OpenSpiel source changes again,
so equivalent contents have different ancestry. At review time the compatibility
branch is one commit ahead and three behind `OpenSpiel-compatable`.

The intended canonical graph is:

```text
OpenSpiel-compatable @ 57872b4
    |
    +-- compatibility changes (including the current review fixes)
```

Do not rewrite or force-push the published branch merely to fix this graph.
A safe alternative is a new branch/worktree based on the OpenSpiel head, with a
patch containing only the current tree's differences from that head. Review and
publish that new branch, retaining the old branch as an archival reference.

For this review, `/tmp/sc42-contract-compatibility.patch` contains that complete
relative patch, including new files and uncommitted review fixes. It has been
checked by applying it to a separate filesystem copy of the OpenSpiel tree and
comparing the resulting files. The temporary artifact belongs to this workspace
session; preserve it before ending the session if needed.

With a clean new worktree, the repair procedure is:

```console
git worktree add -b contract-compatibility-clean /tmp/sc42-contract-clean OpenSpiel-compatable
git -C /tmp/sc42-contract-clean apply /tmp/sc42-contract-compatibility.patch
git -C /tmp/sc42-contract-clean diff --check
```

Review that worktree, run the tests with OpenSpiel installed, and commit the
changes on the new branch. The existing compatibility branch and working tree
remain intact. No branch switching, rebase, reset, or force-push was performed
as part of this review.
