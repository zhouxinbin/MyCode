package com.sinbin.regexp_;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author sinpin
 * @version 1.0
 * @project JavaSeCode
 * @description 非贪婪匹配
 * @date 2025/11/16 21:18:55
 */
public class RegExp09 {
    public static void main(String[] args) {
        String content = "hello1111 ok";
        String regStr = "\\d+";
        String regStr2 = "\\d+?";

        Matcher matcher = Pattern.compile(regStr).matcher(content);
        while (matcher.find()) {
            System.out.println(matcher.group(0));
        }
    }
}
