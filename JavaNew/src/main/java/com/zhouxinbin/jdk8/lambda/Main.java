package com.zhouxinbin.jdk8.lambda;

/**
 * @project: JavaNew
 * @description:
 * @author: zxb
 * @date: 2025/12/27 18:52:37
 * @version: 1.0
 */
public class Main {
    public static void main(String[] args) {
        // 匿名内部类写法
        Test t1 = new Test() {
            @Override
            public String test(Integer i) {
                return i + "";
            }
        };
        System.out.println(t1.test(10));
        // Lambda 表达式写法
        Test t2 = i -> i + "";
        System.out.println(t2.test(20));

        // Lambda 表达式引用 已经实现的静态方法
        Test t3 = Main::function;
        System.out.println(t3.test(30));

        // Lambda 表达式引用 构造方法
        Test2 t4 = String::new;
        System.out.println(t4.test2("zxb"));
    }

    public static String function(Integer i) {
        return "已经实现的方法"+i;
    }
}
