package com.sinbin.regexp_;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author sinpin
 * @version 1.0
 * @project JavaSeCode
 * @description 验证 URL
 * @date 2025/11/16 21:32:18
 */
public class RegURL {
    public static void main(String[] args) {
        String content = "https://www.bilibili.com/video/BV1Eq4y1E79W?" +
                "spm_id_from=333.788.player.switch&vd_source=1c1e33cb7fae15eec6ad3d52ff00c843&p=17";
        String reg = "^((http|https)://)([\\w-]+\\.)+([\\w-]+)(\\/[\\w-?=&/%.#]*)?$";
        Matcher matcher = Pattern.compile(reg).matcher(content);
        while (matcher.find()) {
            System.out.println(matcher.group(0));
            System.out.println(matcher.group(1));
            System.out.println(matcher.group(2));
            System.out.println(matcher.group(3));
            System.out.println(matcher.group(4));
            System.out.println(matcher.group(5));
        }
    }
}
