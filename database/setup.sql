create database deep_fake_detection;

USE deep_fake_detection;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(15),
    date_of_birth DATE,
    gender VARCHAR(10),
    username VARCHAR(50) Unique,
    password VARCHAR(255)
);
