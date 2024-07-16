# Development
mypy .
autopep8 --recursive --in-place .
isort .


## Coding basics

Package installation

Install package with package dependencies:

`pip install -e .`


Install package with development dependencies:

`pip install -e .[dev]`

## Release to Github

### Setup Github remote

Add Github as additional remote to already locally cloned project: `git remote add upstream git@github.com:discovertomorrow/futureEXPERT.git`

### Perform release to Github

```
git checkout release && git pull     # update local release branch
git checkout github && git pull      # update local github branch
git merge -Xtheirs --squash release  # Merge all release changes to github branch without taking the commit history
git commit -m 'release x.x.x'        # Commit all staged changes with a meaningful commit message
git push origin github               # Push updated github branch to out Gitlab repo
git push upstream github:main        # Push local branch 'github' to Github remote branch 'main'
```