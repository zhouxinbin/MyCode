package com.zhouxinbin.jdk8.optional;

import java.util.Optional;

/**
 * @project: JavaNew
 * @description:
 * @author: zxb
 * @date: 2025/12/27 19:42:44
 * @version: 1.0
 */
public class optional {
    public static void main(String[] args) {
        low("ZHOUXINBIN");
    }

    private static void low(String str) {
        Optional.ofNullable(str).ifPresent(s -> {
            System.out.println(s.toLowerCase());
        });
    }
}
