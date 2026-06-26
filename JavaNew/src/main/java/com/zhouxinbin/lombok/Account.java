package com.zhouxinbin.lombok;

/**
 * @project: JavaNew
 * @description:
 * @author: zxb
 * @date: 2026/1/10 14:13:07
 * @version: 1.0
 */

import lombok.Data;
import lombok.Getter;
import lombok.ToString;

@Data
@ToString
public class Account {
    private int id;
    private String name;
    private int age;
}
