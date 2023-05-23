CREATE TABLE users (
	id BIGSERIAL PRIMARY KEY,
	first_name VARCHAR(50) NOT NULL,
	last_name VARCHAR(50),
	contact VARCHAR(100)
);

CREATE TABLE credentials (
	id BIGINT PRIMARY KEY,
	username VARCHAR(50) NOT NULL,
	passwd VARCHAR(50) NOT NULL,
	FOREIGN KEY (id) REFERENCES users(id)
);

CREATE TABLE companies (
    id BIGINT PRIMARY KEY,
    ticker VARCHAR(40) NOT NULL
);

CREATE TABLE news (
	id BIGSERIAL PRIMARY KEY,
	title VARCHAR(200),
	content TEXT,
	date_time TIMESTAMP,
	company_id BIGINT,
	FOREIGN KEY (company_id) REFERENCES companies(id)
);

CREATE TABLE craudsource (
	id BIGSERIAL PRIMARY KEY,
	positive BIGINT,
	negative BIGINT,
	neutral BIGINT,
	date_time TIMESTAMP,
    news_id BIGINT,
	FOREIGN KEY (news_id) REFERENCES news(id)
);

CREATE TABLE augment (
	id BIGSERIAL PRIMARY KEY,
	text_a TEXT,
	text_b TEXT,
	date_time TIMESTAMP,
	news_id BIGINT,
	FOREIGN KEY (news_id) REFERENCES news(id)
);

CREATE TABLE quotes (
	id BIGSERIAL PRIMARY KEY,
	date_ DATE,
	time_ TIME,
	open_ REAL,
	high_ REAL,
	low_ REAL,
	close_ REAL,
    vol_ REAL,
    company_id BIGINT,
    FOREIGN KEY (company_id) REFERENCES companies(id)
);

CREATE TABLE aggregate_data (
	id BIGINT PRIMARY KEY,
	price_ REAL,
	total_ REAL,
	weekday_ BIGINT,
	month_ BIGINT,
	hour_ BIGINT,
	day_ BIGINT,
	year_ BIGINT,
	date_time TIMESTAMP,
    quotes_id BIGSERIAL,
    FOREIGN KEY (quotes_id) REFERENCES quotes(id)
);

WITH ins1 AS (
    INSERT INTO users (first_name, last_name)
    VALUES ('Admin', 'ADMIN')
    RETURNING id AS id
)
INSERT INTO credentials (id, username, passwd)
SELECT id, 'admin', 'admin' FROM ins1;