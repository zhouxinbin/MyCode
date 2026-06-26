package com.sinbin.regexp_;

import java.util.regex.Pattern;

/**
 * @author sinpin
 * @version 1.0
 * @project JavaSeCode
 * @description 整体匹配
 * @date 2025/11/16 22:12:56
 */
public class matches_ {
    public static void main(String[] args) {
        String content = "hello 周新斌";
        String reg = "hello.*";

        boolean matches = Pattern.matches(reg, content);
        System.out.println(matches);
    }
}
