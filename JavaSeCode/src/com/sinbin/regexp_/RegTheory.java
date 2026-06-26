package com.sinbin.regexp_;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class RegTheory {
    public static void main(String[] args) {
        String content = "1881三个1991四个2002n2003";
        // 1. \\d表示任意0-9数字
        // 2.
        String reg = "\\d(\\d\\d)\\d";
        // 正则对象
        Pattern pattern = Pattern.compile(reg);
        // 匹配对象
        Matcher matcher = pattern.matcher(content);
        /** 匹配结果
         * find 根据指定规则，定位符合规则的下标
         * 将符合规则的字符串的首尾下标，记录到groups[]
         * 例如1881 groups[0] = 0，groups[1] = 4
         * 分组后：groups[1] = 1 groups[2] = 3
         * 将 oldlast 值记录为 4, 下次 find 从 4 开始
         */
        while(matcher.find()) {
            // 匹配输出
            System.out.println(matcher.group(0)); // 1881
            System.out.println(matcher.group(1)); // 88
            /**
             * public String group(int group) {
             *         if (first < 0)
             *             throw new IllegalStateException("No match found");
             *         if (group < 0 || group > groupCount())
             *             throw new IndexOutOfBoundsException("No group " + group);
             *         if ((groups[group*2] == -1) || (groups[group*2+1] == -1))
             *             return null;
             *         return getSubSequence(groups[group * 2], groups[group * 2 + 1]).toString();
             *     }
             * 传 0 的时候，根据 groups[0] 和 groups[1] 的值进行截取子字符串
             */
        }
    }
}
