-- This file defines a sample transformation.
-- Edit the sample below or add new transformations
-- using "+ Add" in the file browser.

CREATE MATERIALIZED VIEW sample_aggregation_nov_5_1044 AS
SELECT
    user_type,
    COUNT(user_type) AS total_count
FROM sample_users_nov_5_1044
GROUP BY user_type;
