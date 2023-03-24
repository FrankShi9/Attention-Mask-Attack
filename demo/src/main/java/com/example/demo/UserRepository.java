package com.example.demo;

@Repository
public interface UserRepository extends JpaRepository < UserEntity, Long > {
    UserEntity findByEmail(String email);
}