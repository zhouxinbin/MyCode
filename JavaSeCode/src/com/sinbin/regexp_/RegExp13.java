package com.sinbin.regexp_;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @project: JavaSeCode
 * @description:
 * @author: sinpin
 * @date: 2025/11/17 14:25:58
 * @version: 1.0
 */
public class RegExp13 {
    public static void main(String[] args) {
        String content = "我...我要...学学学....编程";
        // 去除所有 .
        content = Pattern.compile("\\.").matcher(content).replaceAll("");
        // 找到重复字段 并使用单个替换整个
        content = Pattern.compile("(.)\\1+").matcher(content).replaceAll("$1");

        System.out.println(content);


    }
}
