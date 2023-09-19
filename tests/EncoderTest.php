<?php

declare(strict_types=1);

namespace Yethee\Tiktoken\Tests;

use PHPUnit\Framework\TestCase;
use Yethee\Tiktoken\Encoder;
use Yethee\Tiktoken\Vocab\Vocab;

final class EncoderTest extends TestCase
{
    public function testEncode(): void
    {
        $vocab = Vocab::fromFile(__DIR__ . '/Fixtures/cl100k_base.tiktoken');
        $encoder = new Encoder(
            'cl100k_base',
            $vocab,
            '/(?i:\'s|\'t|\'re|\'ve|\'m|\'ll|\'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/u',
        );

        self::assertSame(
            [15339, 1917],
            $encoder->encode('hello world'),
        );

        self::assertSame(
            [8164, 2233, 28089, 8341, 11562, 78746],
            $encoder->encode('привет мир'),
        );

        self::assertSame(
            [9468, 234, 114],
            $encoder->encode('🌶'),
        );

        self::assertSame(
            [627],
            $encoder->encode(".\n"),
        );
    }

    public function testEncodeChunks(): void
    {
        $vocab = Vocab::fromFile(__DIR__ . '/Fixtures/cl100k_base.tiktoken');
        $encoder = new Encoder(
            'cl100k_base',
            $vocab,
            '/(?i:\'s|\'t|\'re|\'ve|\'m|\'ll|\'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/u',
        );
        self::assertSame(
            [
                [
                    57668,
                    53901,
                ],
                [
                    3922,
                    37271,
                    36827,
                    36827,
                    30320,
                    242,
                    89151,
                    16937,
                    29826,
                    28308,
                    232,
                ],
                [
                    3922,
                    7305,
                    225,
                    165,
                    98,
                    255,
                    35287,
                    72027,
                    11571,
                ],
            ],
            $encoder->encodeChunks('你好，今天天气真不错啊，吃饭了没？', 10),
        );
    }

    public function testChunks(): void
    {
        $vocab = Vocab::fromFile(__DIR__ . '/Fixtures/cl100k_base.tiktoken');
        $encoder = new Encoder(
            'cl100k_base',
            $vocab,
            '/(?i:\'s|\'t|\'re|\'ve|\'m|\'ll|\'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/u',
        );
        self::assertSame(
            [
                '你好',
                '，今天天气真不错啊',
                '，吃饭了没？',
            ],
            $encoder->chunk('你好，今天天气真不错啊，吃饭了没？', 10),
        );
    }

    public function testDecode(): void
    {
        $vocab = Vocab::fromFile(__DIR__ . '/Fixtures/cl100k_base.tiktoken');
        $encoder = new Encoder(
            'cl100k_base',
            $vocab,
            '/(?i:\'s|\'t|\'re|\'ve|\'m|\'ll|\'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/u',
        );

        self::assertSame('hello world', $encoder->decode([15339, 1917]));
        self::assertSame('привет мир', $encoder->decode([8164, 2233, 28089, 8341, 11562, 78746]));
        self::assertSame('🌶', $encoder->decode([9468, 234, 114]));
        self::assertSame(".\n", $encoder->decode([627]));
    }
}
