## Issues

* closes prognostica/future/planning#<insert-ticket-id-here>

## Development progress

Current progress of development. Changes should only be done by the developer.

### General coding

- [ ] The functionality is available as expected from description of treated issues (see closed/fixed issues above).
- [ ] The new/updated code fits well into the overall project structure/architecture.
- [ ] There are sufficient unit tests and integration tests available.
- [ ] If functionality is not guaranteed by unit tests or integration tests, manual tests have been made.
- [ ] The user and dev documentation is extended/updated where necessary.
- [ ] Newly added third-party dependencies are added with an explicit version to `setup.py`/`requirements.txt`/`Dockerfile`.
- [ ] Newly added third-party dependencies provide a compatible license, see [Handbook](https://prognostica.pages.prognostica.de/handbook/operations/THIRD_PARTY_CODE.html).
- [ ] The source branch is up-to-date with the target branch (0 commits behind).

When the development is finished, that's when all above tasks are completed, mark review as _ready_ and set a reviewer as _Assignee_ and _Reviewer_.


## Review progress

Current progress of code review. Changes should only be done by the reviewer.

- [ ] The functionality is available as expected from description of treated issues (see closed/fixed issues above).
- [ ] The new/updated code fits well into the overall project structure/architecture.
- [ ] There are sufficient unit tests and integration tests available.
- [ ] The user and dev documentation is extended/updated where necessary.
- [ ] Newly arised issues during review process are created in Jira and linked in the related MR discussion.
- [ ] Newly added third-party dependencies are added with an explicit version to `setup.py`/`requirements.txt`/`Dockerfile`.
- [ ] Newly added third-party dependencies provide a compatible license, see [Handbook](https://prognostica.pages.prognostica.de/handbook/operations/THIRD_PARTY_CODE.html).
- [ ] The source branch is up-to-date with the target branch (0 commits behind).
- [ ] If functionality is not guaranteed by unit tests or integration tests, manual tests have been made **at the end of the review**.

When the code review is finished, that's when all tasks are completed, _approve_ the Merge Request before merging. After merging, ensure the issue is closed.
