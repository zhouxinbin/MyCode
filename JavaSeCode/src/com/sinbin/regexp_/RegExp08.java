package com.sinbin.regexp_;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author sinpin
 * @version 1.0
 * @project JavaSeCode
 * @description 非捕获分组
 * @date 2025/11/16 21:02:22
 */
public class RegExp08 {
    public static void main(String[] args) {
        String content = "hello周新斌跑步 hhh周新斌老师 周新斌同学hhh 周新斌钓鱼";

        String reg1 = "周新斌(?:同学|老师|跑步)";
        String reg2 = "周新斌(?=同学|老师|跑步)";
        String reg3 = "周新斌(?!同学|老师|跑步)";

        Pattern pattern = Pattern.compile(reg3);
        Matcher matcher = pattern.matcher(content);

        while(matcher.find()) {
            // 匹配输出
            System.out.println(matcher.group(0));
        }
    }
}
