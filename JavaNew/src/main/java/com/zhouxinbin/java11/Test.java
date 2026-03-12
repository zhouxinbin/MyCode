package com.zhouxinbin.java11;

import java.util.function.Consumer;

/**
 * @project: JavaNew
 * @description:
 * @author: zxb
 * @date: 2026/1/6 17:03:29
 * @version: 1.0
 */
public class Test {
    public static void main(String[] args) {
        Consumer<String> consumer= (var s) -> {
            System.out.println(s);
        };


        consumer.accept("zxb");

        String sql =
                """
                select *
                from student
                where name = 1
                """;

    }
}
