package com.sinbin.regexp_;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @project: JavaSeCode
 * @description: Matcher 类方法
 * @author: sinpin
 * @date: 2025/11/16 23:06:24
 * @version: 1.0
 */
public class MatcherCls {
    public static void main(String[] args) {
        String content = "hello 123 hello 456 hello";
        String reg = "hello.*";
        Pattern compile = Pattern.compile(reg);
        Matcher matcher = compile.matcher(content);

        while (matcher.find()) {
            System.out.println(matcher.start());
            System.out.println(matcher.end());
        }

        System.out.println();
    }
}
