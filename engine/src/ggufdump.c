/* Print what is inside a GGUF checkpoint.
 *
 * Useful on its own for finding out what a downloaded model actually is,
 * and useful here as proof the parser works on real files rather than only
 * on the ones the tests build.
 *
 * It reads only the directory, never the weights, so it returns instantly
 * on a file of any size.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf.h"

static void print_size(uint64_t bytes)
{
    if (bytes >= 1000000000ull) printf("%.2f GB", (double)bytes / 1e9);
    else if (bytes >= 1000000ull) printf("%.1f MB", (double)bytes / 1e6);
    else printf("%.0f KB", (double)bytes / 1e3);
}

/* Try a list of keys and print the first one that exists.
 * Model writers do not agree on prefixes, so llama.arch and qwen3.arch and
 * so on all mean the same thing and any of them will do. */
static void print_first_u64(const GgufFile *g, const char *label,
                            const char *const *keys, int n)
{
    for (int i = 0; i < n; i++) {
        uint64_t v;
        if (gguf_kv_u64(g, keys[i], &v) == 0) {
            printf("  %-22s %llu\n", label, (unsigned long long)v);
            return;
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: stratum-gguf FILE [--tensors]\n\n"
               "Prints the metadata and shape of a GGUF checkpoint without "
               "reading any weights.\n");
        return 1;
    }

    const int show_tensors = (argc > 2 && !strcmp(argv[2], "--tensors"));

    GgufFile g;
    if (gguf_open(&g, argv[1]) != 0) {
        fprintf(stderr, "%s\n", g.err);
        return 1;
    }

    char arch[128] = "";
    char name[256] = "";
    gguf_kv_str(&g, "general.architecture", arch, sizeof arch);
    gguf_kv_str(&g, "general.name", name, sizeof name);

    printf("%s\n", argv[1]);
    printf("  file size              ");
    print_size((uint64_t)g.map.size);
    printf("\n");
    printf("  gguf version           %u\n", g.version);
    if (*name) printf("  name                   %s\n", name);
    if (*arch) printf("  architecture           %s\n", arch);
    printf("  tensors                %llu\n", (unsigned long long)g.n_tensors);
    printf("  metadata keys          %llu\n", (unsigned long long)g.n_kv);

    /* The shape keys are prefixed with the architecture, which is only known
     * once the file is open, so they get built here. */
    if (*arch) {
        char k1[192], k2[192], k3[192], k4[192], k5[192];
        snprintf(k1, sizeof k1, "%s.block_count", arch);
        snprintf(k2, sizeof k2, "%s.embedding_length", arch);
        snprintf(k3, sizeof k3, "%s.expert_count", arch);
        snprintf(k4, sizeof k4, "%s.expert_used_count", arch);
        snprintf(k5, sizeof k5, "%s.context_length", arch);
        const char *a[] = { k1 }, *b[] = { k2 }, *c[] = { k3 },
                   *d[] = { k4 }, *e[] = { k5 };
        print_first_u64(&g, "layers", a, 1);
        print_first_u64(&g, "embedding length", b, 1);
        print_first_u64(&g, "context length", e, 1);
        print_first_u64(&g, "experts", c, 1);
        print_first_u64(&g, "experts per token", d, 1);
    }

    /* Total weight bytes and a count per element type, which together say
     * how the model was quantized without needing to trust its filename. */
    uint64_t total = 0;
    uint64_t per_type[GG_TYPE_MAX];
    memset(per_type, 0, sizeof per_type);
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        total += g.tensors[i].nbytes;
        if (g.tensors[i].type < GG_TYPE_MAX) per_type[g.tensors[i].type]++;
    }
    printf("  weights                ");
    print_size(total);
    printf("\n");

    printf("  types                  ");
    int first = 1;
    for (uint32_t t = 0; t < GG_TYPE_MAX; t++) {
        if (!per_type[t]) continue;
        printf("%s%s x%llu", first ? "" : ", ", gguf_type_name(t),
               (unsigned long long)per_type[t]);
        first = 0;
    }
    printf("\n");

    if (show_tensors) {
        printf("\n");
        for (uint64_t i = 0; i < g.n_tensors; i++) {
            const GgufTensor *t = &g.tensors[i];
            printf("  %-46.*s %-6s ", (int)t->name_len, t->name,
                   gguf_type_name(t->type));
            for (uint32_t d = 0; d < t->n_dims; d++)
                printf("%s%llu", d ? "x" : "", (unsigned long long)t->dims[d]);
            printf("  ");
            print_size(t->nbytes);
            printf("\n");
        }
    }

    gguf_close(&g);
    return 0;
}
