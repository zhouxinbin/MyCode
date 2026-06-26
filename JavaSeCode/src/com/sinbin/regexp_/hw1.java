package com.sinbin.regexp_;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @project: JavaSeCode
 * @description:
 * @author: sinpin
 * @date: 2025/11/17 14:51:26
 * @version: 1.0
 */
public class hw1 {
    public static void main(String[] args) {
        String email = "zxb@zhouxinbin.com";
        String reg = "^[\\w-]+@([a-zA-Z]+\\.)+[a-zA-Z]+$";

        System.out.println(email.matches(reg));
    }
}
