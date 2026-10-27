"""Allow ``python -m damage_analytics`` after an editable installation."""

from damage_analytics.cli import main

raise SystemExit(main())
