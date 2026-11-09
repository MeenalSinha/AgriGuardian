# AgriGuardian Data Directory

This directory stores runtime application datasets and persistence artifacts:

* `advisories.csv`: CSV store of farmer advisory cases, symptoms, risks, feedback, and review statuses. Created automatically by `DataManager`.
* `rules.json`: Configurable dynamic agricultural diagnostic rules loaded by `DynamicRuleEngine` (located in `config/rules.json`).

