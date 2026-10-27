// Expr -> Brainfuck compiler.
//
// Pipeline:
//   1. Lexer + recursive-descent parser (one statement per line, errors carry line numbers).
//   2. Symbolic execution: there is no control flow, so every variable is a polynomial over
//      the runtime input() values, mod 256.  Values are linear forms over "atoms" (inputs and
//      products of affine operand forms).  Products are hash-consed on normalized operands,
//      which gives constant folding/propagation, dead-code elimination and common
//      subexpression elimination for free.
//   3. Analysis: atoms depending on a single runtime cell are unary functions of it and are
//      tabulated over all 256 values.  Runtime cells ("sources") are the inputs and the
//      products that need a real multiplier; every output cell and multiplier operand port
//      receives, from each source, a function given as a 256-entry table.
//   4. Code generation: the live inputs are read, then each source is distributed either by
//      a destructive loop (linear tables with cheap coefficients, planned as a small
//      multiplication tree through temporaries) or by a cascade: 255 nested "x[-" levels
//      whose deepest case knows x at compile time and adds each table entry as a constant.
//      Multipliers cascade over one operand and add u * V along a precomputed factor chain
//      (or use a compact base-4 shift-and-add once the code budget is spent).  Finally every
//      print walks to its output cell, adjusts the constant offset and prints.
//   5. Cell positions come from a local search over an estimated step count, where cells
//      share memory whenever their lifetimes on the event timeline do not overlap.

#include <algorithm>
#include <bit>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using u8 = std::uint8_t;
// Linear combination of atoms: (atom id, nonzero coefficient), sorted by atom id.
using Terms = std::vector<std::pair<int, u8>>;

struct CompileError {
    int line;
    std::string message;
};

// Number of '+' or '-' needed to add c to a cell.
int addCost(unsigned c) {
    c &= 255U;
    return c <= 128 ? int(c) : int(256 - c);
}

u8 inverse(unsigned odd) {
    unsigned x = odd;
    for (int i = 0; i < 4; ++i) x *= 2U - odd * x;
    return u8(x);
}

// ===================================================================== Lexer

enum class Tok { Ident, Number, Plus, Minus, Star, LParen, RParen, Assign, End };

struct Token {
    Tok kind;
    std::string text;
    int value = 0;
};

bool isLetter(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
bool isDigit(char c) { return c >= '0' && c <= '9'; }

std::string describeChar(char c) {
    auto u = static_cast<unsigned char>(c);
    if (u >= 0x21 && u < 0x7f) return std::string("'") + c + "'";
    static const char* hex = "0123456789abcdef";
    return std::string("byte 0x") + hex[u >> 4] + hex[u & 15];
}

std::vector<Token> lexLine(const std::string& line, int lineNo) {
    std::vector<Token> tokens;
    std::size_t i = 0;
    while (i < line.size()) {
        char c = line[i];
        if (c == '#') break;
        if (c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f') {
            ++i;
            continue;
        }
        if (isLetter(c)) {
            std::size_t j = i;
            while (j < line.size() && (isLetter(line[j]) || isDigit(line[j]))) ++j;
            tokens.push_back({Tok::Ident, line.substr(i, j - i)});
            i = j;
            continue;
        }
        if (isDigit(c)) {
            std::size_t j = i;
            int value = 0;
            while (j < line.size() && isDigit(line[j])) {
                value = std::min(value * 10 + (line[j] - '0'), 256);
                ++j;
            }
            std::string text = line.substr(i, j - i);
            if (value > 255) throw CompileError{lineNo, "integer literal " + text + " is out of range 0..255"};
            tokens.push_back({Tok::Number, text, value});
            i = j;
            continue;
        }
        Tok kind;
        switch (c) {
            case '+': kind = Tok::Plus; break;
            case '-': kind = Tok::Minus; break;
            case '*': kind = Tok::Star; break;
            case '(': kind = Tok::LParen; break;
            case ')': kind = Tok::RParen; break;
            case '=': kind = Tok::Assign; break;
            default: throw CompileError{lineNo, "unsupported character " + describeChar(c)};
        }
        tokens.push_back({kind, std::string(1, c)});
        ++i;
    }
    tokens.push_back({Tok::End, ""});
    return tokens;
}

// ========================================================== Symbolic values

struct Value {
    u8 k = 0;     // constant part
    Terms terms;  // non-constant part
};

// ca * a + cb * b
Terms lincomb(const Terms& a, u8 ca, const Terms& b, u8 cb) {
    Terms r;
    r.reserve(a.size() + b.size());
    std::size_t i = 0, j = 0;
    auto push = [&](int atom, unsigned c) {
        if (u8(c) != 0) r.emplace_back(atom, u8(c));
    };
    while (i < a.size() || j < b.size()) {
        if (j == b.size() || (i < a.size() && a[i].first < b[j].first)) {
            push(a[i].first, unsigned(a[i].second) * ca);
            ++i;
        } else if (i == a.size() || b[j].first < a[i].first) {
            push(b[j].first, unsigned(b[j].second) * cb);
            ++j;
        } else {
            push(a[i].first, unsigned(a[i].second) * ca + unsigned(b[j].second) * cb);
            ++i;
            ++j;
        }
    }
    return r;
}

Value combine(const Value& a, u8 ca, const Value& b, u8 cb) {
    return Value{u8(unsigned(a.k) * ca + unsigned(b.k) * cb), lincomb(a.terms, ca, b.terms, cb)};
}

Value scale(const Value& a, u8 c) { return combine(a, c, Value{}, 0); }

struct Program {
    struct Atom {
        int formA = -1;  // product operands (normalized form ids); -1 for inputs
        int formB = -1;
        bool isInput() const { return formA < 0; }
    };
    std::vector<Atom> atoms;
    std::vector<Value> forms;
    std::map<std::pair<Terms, u8>, int> formIds;
    std::map<std::pair<int, int>, int> productIds;
    std::vector<int> inputs;  // atom of each executed input(), in order
    std::vector<Value> prints;

    Value input() {
        int id = int(atoms.size());
        atoms.push_back({});
        inputs.push_back(id);
        return Value{0, {{id, u8(1)}}};
    }

    // Writes a non-constant affine form F as scale * N where N is a canonical representative:
    // the common power of two is pulled out and, among all unit multiples, the one with the
    // cheapest coefficients is chosen.  Equal-up-to-scale operands therefore share products.
    // Coefficients are paid once per unit of the atom at runtime, the constant only once.
    std::pair<u8, int> normalize(const Value& v) {
        int g = 8;
        for (const auto& term : v.terms) g = std::min(g, std::countr_zero(unsigned(term.second)));
        if (v.k != 0) g = std::min(g, std::countr_zero(unsigned(v.k)));
        const unsigned m = 256U >> g;  // the normalized form only matters mod m
        auto rep = [m](unsigned x) { return x <= m - x ? x : 256 - (m - x); };
        std::vector<unsigned> residue;
        residue.reserve(v.terms.size());
        for (const auto& term : v.terms) residue.push_back((unsigned(term.second) >> g) % m);
        const unsigned kResidue = (unsigned(v.k) >> g) % m;
        Value best, cand{0, Terms(v.terms.size())};
        long bestCost = LONG_MAX;
        unsigned bestUnit = 1;
        for (unsigned u = 1; u < m; u += 2) {
            long cost = 0;
            for (std::size_t i = 0; i < v.terms.size(); ++i) {
                unsigned x = rep((u * residue[i]) % m);
                cand.terms[i] = {v.terms[i].first, u8(x)};
                cost += 128 * addCost(x);
            }
            cand.k = u8(rep((u * kResidue) % m));
            cost += addCost(cand.k);
            if (cost < bestCost || (cost == bestCost && std::tie(cand.terms, cand.k) < std::tie(best.terms, best.k))) {
                bestCost = cost;
                best = cand;
                bestUnit = u;
            }
        }
        u8 s = u8((1U << g) * inverse(bestUnit));
        auto [it, inserted] = formIds.try_emplace({best.terms, best.k}, int(forms.size()));
        if (inserted) forms.push_back(best);
        return {s, it->second};
    }

    Value mul(const Value& x, const Value& y) {
        if (x.terms.empty()) return scale(y, x.k);
        if (y.terms.empty()) return scale(x, y.k);
        auto [sa, fa] = normalize(x);
        auto [sb, fb] = normalize(y);
        u8 c = u8(unsigned(sa) * sb);
        if (c == 0) return Value{};
        if (fa > fb) std::swap(fa, fb);
        auto [it, inserted] = productIds.try_emplace({fa, fb}, int(atoms.size()));
        if (inserted) atoms.push_back({fa, fb});
        return Value{0, {{it->second, c}}};
    }
};

// ==================================================================== Parser

class Parser {
public:
    explicit Parser(Program& prog) : prog_(prog) {}

    void statement(const std::vector<Token>& tokens, int line) {
        toks_ = &tokens;
        i_ = 0;
        line_ = line;
        const Token& first = peek();
        if (first.kind == Tok::End) return;
        if (first.kind == Tok::Ident && first.text == "print") {
            ++i_;
            expect(Tok::LParen, "'(' after print");
            Value v = expression();
            expect(Tok::RParen, "')'");
            end();
            prog_.prints.push_back(std::move(v));
            return;
        }
        if (first.kind == Tok::Ident && first.text == "input")
            fail("reserved word 'input' can only appear as 'name = input()'");
        if (first.kind != Tok::Ident) fail("unexpected token " + describe(first));
        std::string name = first.text;
        ++i_;
        expect(Tok::Assign, "'=' after variable name");
        if (peek().kind == Tok::Ident && peek().text == "input" && (*toks_)[i_ + 1].kind == Tok::LParen) {
            i_ += 2;
            expect(Tok::RParen, "')' after 'input('");
            end();
            vars_[name] = prog_.input();
            return;
        }
        Value v = expression();
        end();
        vars_[name] = std::move(v);
    }

private:
    Program& prog_;
    std::map<std::string, Value> vars_;
    const std::vector<Token>* toks_ = nullptr;
    std::size_t i_ = 0;
    int line_ = 0;

    const Token& peek() const { return (*toks_)[i_]; }
    [[noreturn]] void fail(const std::string& msg) const { throw CompileError{line_, msg}; }
    static std::string describe(const Token& t) { return t.kind == Tok::End ? "end of line" : "'" + t.text + "'"; }

    void expect(Tok kind, const std::string& what) {
        if (peek().kind != kind) fail("expected " + what + ", found " + describe(peek()));
        ++i_;
    }
    void end() {
        if (peek().kind != Tok::End) fail("unexpected token " + describe(peek()));
    }

    Value expression() {
        Value v = term();
        while (peek().kind == Tok::Plus || peek().kind == Tok::Minus) {
            bool minus = peek().kind == Tok::Minus;
            ++i_;
            v = combine(v, 1, term(), minus ? 255 : 1);
        }
        return v;
    }
    Value term() {
        Value v = unary();
        while (peek().kind == Tok::Star) {
            ++i_;
            v = prog_.mul(v, unary());
        }
        return v;
    }
    Value unary() {
        bool negate = false;
        while (peek().kind == Tok::Plus || peek().kind == Tok::Minus) {
            negate ^= peek().kind == Tok::Minus;
            ++i_;
        }
        Value v = primary();
        return negate ? scale(v, 255) : v;
    }
    Value primary() {
        const Token& t = peek();
        switch (t.kind) {
            case Tok::Number:
                ++i_;
                return Value{u8(t.value), {}};
            case Tok::Ident: {
                if (t.text == "input" || t.text == "print")
                    fail("reserved word '" + t.text + "' cannot be used in an expression");
                auto it = vars_.find(t.text);
                if (it == vars_.end()) fail("undefined variable '" + t.text + "'");
                ++i_;
                return it->second;
            }
            case Tok::LParen: {
                ++i_;
                Value v = expression();
                expect(Tok::RParen, "')'");
                return v;
            }
            default:
                fail("expected expression, found " + describe(t));
        }
    }
};

// ================================================================== Emitter

class Emitter {
public:
    std::string code;
    long pos = 0;

    void move(long to) {
        if (to > pos)
            code.append(std::size_t(to - pos), '>');
        else
            code.append(std::size_t(pos - to), '<');
        pos = to;
    }
    void add(int n) {
        n &= 255;
        if (n <= 128)
            code.append(std::size_t(n), '+');
        else
            code.append(std::size_t(256 - n), '-');
    }
    void addAt(long p, int n) {
        move(p);
        add(n);
    }
    void open() { code += '['; }
    void close() { code += ']'; }
    void put(char c) { code += c; }
    // Raw code whose final pointer position is data independent but not tracked by move().
    void raw(const char* s, long newPos) {
        code += s;
        pos = newPos;
    }
};

// ================================================================ Compiler

// Multiplier block shapes: offsets of the operand ports U and V, the product W, the flag
// used when W is then distributed by a cascade, and cells only touched while multiplying.
struct Shape {
    long width, u, v, w, wFlag;
    std::vector<long> scratch;
};
// [U F V T1 W T2]: a cascade over U (flag F) whose case u adds u * V into W through at most
// two temporaries.  Fast, but ~256 cases of code.
const Shape kCascadeShape{6, 0, 2, 4, 5, {1, 3}};
// [M3 M2 M1 U P1 P2 VA W VB]: base-4 shift-and-add with ~1 KB of code.
const Shape kCompactShape{9, 3, 6, 7, 8, {0, 1, 2, 4, 5}};
constexpr long kMaxBlockWidth = 9;
// Expected loop count of a runtime value (inputs are uniform in 0..255).
constexpr long kAvgValue = 128;
// Upper bounds on emitted cascade code (bytes): per distribution cascade, and in total.  The
// reference runner needs ~32-64 bytes of memory per instruction.  Beyond them multipliers
// use the compact shape, linear sources use loops and nonlinear ones cascade into hubs.
constexpr long kMaxCascadeCode = 200000;
constexpr long kTotalCascadeCode = 2000000;

// Cheapest way to add u * V into W for a constant u (V consumed), as up to three loops:
//   V  -> W * a,  T1 * f
//   T1 -> W * b,  T2 * g        (only if f != 0)
//   T2 -> W * h                 (only if g != 0)
// so u == a + f * (b + g * h).  Costs are per unit of V in the [U F V T1 W T2] geometry.
struct Chain {
    int cost = INT_MAX;
    u8 a = 0, f = 0, b = 0, g = 0, h = 0;
};
const std::vector<Chain>& chainTable() {
    static const std::vector<Chain> table = [] {
        std::vector<unsigned> cheap;  // multipliers worth an extra loop
        for (unsigned f = 1; f < 256; ++f)
            if (addCost(f) <= 24) cheap.push_back(f);
        // Adding y * T1 into W, optionally through T2.
        std::vector<Chain> viaT1(256);
        for (unsigned y = 0; y < 256; ++y) viaT1[y] = {2 + addCost(y) + (y ? 2 : 0), 0, 0, u8(y), 0, 0};
        for (unsigned g : cheap)
            for (unsigned h : cheap)
                for (unsigned b = 0; b < 256; ++b) {
                    const unsigned y = (b + g * h) & 255U;
                    const int cost = 2 + addCost(b) + addCost(g) + 4 + 4 + addCost(h);
                    if (cost < viaT1[y].cost) viaT1[y] = {cost, 0, 0, u8(b), u8(g), u8(h)};
                }
        std::vector<Chain> t(256);
        t[0] = {2, 0, 0, 0, 0, 0};
        for (unsigned u = 1; u < 256; ++u) t[u] = {6 + addCost(u), u8(u), 0, 0, 0, 0};
        for (unsigned f : cheap)
            for (unsigned y = 1; y < 256; ++y)
                for (unsigned a = 0; a < 256; ++a) {
                    const unsigned u = (a + f * y) & 255U;
                    const int cost = 2 + addCost(a) + addCost(f) + (a ? 4 : 2) + viaT1[y].cost;
                    if (cost < t[u].cost) t[u] = {cost, u8(a), u8(f), viaT1[y].b, viaT1[y].g, viaT1[y].h};
                }
        return t;
    }();
    return table;
}

// Approximate code size (bytes) of one cascade multiplier.
long cascadeMultiplyCode() {
    static const long size = [] {
        long total = 0;
        for (const Chain& c : chainTable()) total += 36 + addCost(c.a) + addCost(c.f) + addCost(c.b) + addCost(c.g) + addCost(c.h);
        return total;
    }();
    return size;
}

struct Add {
    long pos;
    u8 coeff;
};

class Compiler {
public:
    explicit Compiler(const Program& prog) : prog_(prog) {}

    std::string run() {
        analyze();
        layout();
        emit();
        return e_.code;
    }

private:
    struct Target {
        const Terms* terms;
        u8 k;       // constant part (ports only)
        int block;  // multiplier block for operand ports, else -1
        int port;   // 0 = U operand, 1 = V operand
        int group;  // output group, else -1
    };
    // Cell positions.  Every logical cell is live over an interval of the event timeline
    // (0 = reads, then input distributions, then per block a multiply and a distribution
    // event); cells may share a position whenever their intervals are disjoint.
    struct Layout {
        std::vector<long> base;    // per block
        std::vector<char> mirror;  // per block: geometry reversed
        std::vector<long> gpos;    // per group not living in a multiplier's W cell
        std::vector<long> ipos;    // per atom: cell of live, non-aliased inputs
    };
    struct Occupancy {
        long pos;
        int start, end;
    };
    struct LoopSpec {
        long src;
        std::vector<Add> adds;
    };

    const Program& prog_;
    std::vector<Terms> groupTerms_;               // distinct non-constant print forms
    std::vector<std::pair<int, u8>> printSeq_;    // (group or -1 for pure constants, constant)
    std::vector<char> live_;
    // Runtime cells ("sources") are inputs and products that need a real multiply.  Every
    // other atom is a unary function of one source: srcOf_ names it and table_ holds the
    // atom's value for each of the 256 source values (identity for sources themselves).
    std::vector<int> srcOf_;
    std::vector<std::vector<u8>> table_;
    std::vector<int> blocks_;                     // block -> product atom needing a multiplier
    std::vector<Target> targets_;
    struct Feed {
        int target;
        std::vector<u8> table;  // contribution as a function of the source value
    };
    std::vector<std::vector<Feed>> feeds_;        // source atom -> contributions
    std::vector<std::vector<std::pair<int, u8>>> uses_;  // loop-distributed source -> (target, coeff)
    std::vector<char> cascade_;                   // source distributed by a cascade
    std::vector<char> hub_;                       // ... into one hub per unary atom, then loops
    std::vector<char> compact_;                   // block uses the compact multiplier shape
    std::vector<u8> preset_;                      // target -> constant already added by a cascade
    std::vector<int> inputAlias_;                 // input atom read straight into this target
    std::vector<int> wAlias_;                     // block whose product is accumulated in a group cell
    std::vector<int> groupBlock_;
    std::vector<int> writeEvent_;   // source -> event at which it is added into its targets
    std::vector<int> firstWrite_;   // target -> first event that writes it
    std::vector<int> distInputs_;   // live inputs that need a loop or cascade, in order
    int mulEvent0_ = 0, endEvent_ = 0;
    Layout lay_;
    Emitter e_;
    std::vector<char> nz_;  // cells that may currently be nonzero

    // ------------------------------------------------------------ analysis

    void analyze() {
        std::map<Terms, int> groupIds;
        for (const Value& v : prog_.prints) {
            if (v.terms.empty()) {
                printSeq_.emplace_back(-1, v.k);
                continue;
            }
            auto [it, inserted] = groupIds.try_emplace(v.terms, int(groupTerms_.size()));
            if (inserted) groupTerms_.push_back(v.terms);
            printSeq_.emplace_back(it->second, v.k);
        }

        const std::size_t n = prog_.atoms.size();
        live_.assign(n, 0);
        std::vector<int> stack;
        auto mark = [&](const Terms& t) {
            for (const auto& term : t)
                if (!live_[std::size_t(term.first)]) {
                    live_[std::size_t(term.first)] = 1;
                    stack.push_back(term.first);
                }
        };
        for (const Terms& t : groupTerms_) mark(t);
        while (!stack.empty()) {
            const auto& atom = prog_.atoms[std::size_t(stack.back())];
            stack.pop_back();
            if (atom.isInput()) continue;
            mark(prog_.forms[std::size_t(atom.formA)].terms);
            mark(prog_.forms[std::size_t(atom.formB)].terms);
        }

        // Value tables (atoms only depend on lower ids).
        srcOf_.assign(n, -1);
        table_.assign(n, {});
        auto evalForm = [&](const Value& f, unsigned v) {
            unsigned x = f.k;
            for (const auto& term : f.terms) x += unsigned(term.second) * table_[std::size_t(term.first)][v];
            return u8(x);
        };
        auto makeSource = [&](std::size_t a) {
            srcOf_[a] = int(a);
            table_[a].resize(256);
            for (unsigned v = 0; v < 256; ++v) table_[a][v] = u8(v);
        };
        std::vector<int> candidates;  // products needing a multiplier
        for (std::size_t a = 0; a < n; ++a) {
            if (!live_[a]) continue;
            const auto& atom = prog_.atoms[a];
            if (atom.isInput()) {
                makeSource(a);
                continue;
            }
            const Value& fa = prog_.forms[std::size_t(atom.formA)];
            const Value& fb = prog_.forms[std::size_t(atom.formB)];
            int src = -2;
            for (const Value* f : {&fa, &fb})
                for (const auto& term : f->terms) {
                    int s = srcOf_[std::size_t(term.first)];
                    src = (src == -2 || src == s) ? s : -1;
                }
            if (src < 0) {
                makeSource(a);
                candidates.push_back(int(a));
                continue;
            }
            srcOf_[a] = src;
            table_[a].resize(256);
            for (unsigned v = 0; v < 256; ++v) table_[a][v] = u8(unsigned(evalForm(fa, v)) * evalForm(fb, v));
        }

        // Drop multipliers whose result cancels out everywhere it is used (highest first, so
        // every consumer has been decided), then build the final targets.
        buildTargets(candidates);
        std::vector<char> usefulTarget(targets_.size(), 0);
        for (std::size_t t = 2 * blocks_.size(); t < targets_.size(); ++t) usefulTarget[t] = 1;
        std::vector<int> useful;
        for (std::size_t b = blocks_.size(); b-- > 0;) {
            const auto& feeds = feeds_[std::size_t(blocks_[b])];
            if (std::none_of(feeds.begin(), feeds.end(), [&](const Feed& f) { return usefulTarget[std::size_t(f.target)]; }))
                continue;
            useful.insert(useful.begin(), blocks_[b]);
            usefulTarget[2 * b] = usefulTarget[2 * b + 1] = 1;
        }
        if (useful.size() != candidates.size()) buildTargets(useful);

        inputAlias_.assign(n, -1);
        cascade_.assign(n, 0);
        hub_.assign(n, 0);
        uses_.assign(n, {});
        wAlias_.assign(blocks_.size(), -1);
        groupBlock_.assign(groupTerms_.size(), -1);
        std::vector<char> targetFed(targets_.size(), 0);
        // Multipliers get the cascade shape while the code budget lasts.
        long cascadeBudget = kTotalCascadeCode;
        compact_.assign(blocks_.size(), 0);
        for (std::size_t b = 0; b < blocks_.size(); ++b) {
            if (cascadeBudget >= cascadeMultiplyCode())
                cascadeBudget -= cascadeMultiplyCode();
            else
                compact_[b] = 1;
        }
        auto decide = [&](int s, int block) {
            auto& feeds = feeds_[std::size_t(s)];
            bool linear = true;
            for (const Feed& f : feeds)
                for (unsigned v = 0; v < 256 && linear; ++v) linear = f.table[v] == u8(f.table[1] * v);
            if (linear && feeds.size() == 1 && feeds[0].table[1] == 1) {
                const int t = feeds[0].target, g = targets_[std::size_t(t)].group;
                if (block < 0 && !targetFed[std::size_t(t)]) {
                    inputAlias_[std::size_t(s)] = t;
                    targetFed[std::size_t(t)] = 1;
                    return;
                }
                if (block >= 0 && g >= 0 && groupBlock_[std::size_t(g)] < 0) {
                    wAlias_[std::size_t(block)] = g;
                    groupBlock_[std::size_t(g)] = block;
                    return;
                }
            }
            // Per-unit loop cost vs. cascade (6 steps per unit plus one constant add per target).
            // The case work also approximates the cascade's code size, which is bounded.
            long loopEst = 2, caseWork = 0;
            for (const Feed& f : feeds) {
                loopEst += std::min(addCost(f.table[1]), 16) + 2;
                for (unsigned v = 0; v < 256; ++v) caseWork += addCost(f.table[v]) + 2;
            }
            const long cascadeEst = 6 * kAvgValue + caseWork / 256;
            const bool tooBig = caseWork > std::min(kMaxCascadeCode, cascadeBudget);
            if (linear && (tooBig || loopEst * kAvgValue <= cascadeEst)) {
                for (const Feed& f : feeds) uses_[std::size_t(s)].emplace_back(f.target, f.table[1]);
            } else {
                cascade_[std::size_t(s)] = 1;
                hub_[std::size_t(s)] = tooBig;
                if (!tooBig) cascadeBudget -= caseWork;
            }
        };
        for (int a : prog_.inputs) {
            if (!live_[std::size_t(a)]) continue;
            if (feeds_[std::size_t(a)].empty()) {  // every use cancels out
                live_[std::size_t(a)] = 0;
                continue;
            }
            decide(a, -1);
        }
        for (std::size_t b = 0; b < blocks_.size(); ++b) decide(blocks_[b], int(b));

        // A cascade adds an arbitrary constant per case anyway, so a port's constant and a
        // group's first printed constant ride along for free in one cascade writing the cell.
        std::vector<u8> want(targets_.size(), 0);
        for (std::size_t t = 0; t < 2 * blocks_.size(); ++t) want[t] = targets_[t].k;
        for (auto it = printSeq_.rbegin(); it != printSeq_.rend(); ++it)
            if (it->first >= 0) want[2 * blocks_.size() + std::size_t(it->first)] = it->second;
        preset_.assign(targets_.size(), 0);
        std::vector<char> folded(targets_.size(), 0);
        for (std::size_t s = 0; s < n; ++s) {
            if (!cascade_[s] || hub_[s]) continue;
            for (Feed& f : feeds_[s]) {
                const auto t = std::size_t(f.target);
                if (folded[t] || want[t] == 0) continue;
                folded[t] = 1;
                preset_[t] = want[t];
                for (u8& x : f.table) x = u8(x + want[t]);
            }
        }

        writeEvent_.assign(n, 0);
        int ev = 1;
        for (int a : prog_.inputs)
            if (live_[std::size_t(a)] && inputAlias_[std::size_t(a)] < 0) {
                distInputs_.push_back(a);
                writeEvent_[std::size_t(a)] = ev++;
            }
        mulEvent0_ = ev;
        for (std::size_t b = 0; b < blocks_.size(); ++b)
            writeEvent_[std::size_t(blocks_[b])] = multiplyEvent(b) + (wAlias_[b] >= 0 ? 0 : 1);
        endEvent_ = mulEvent0_ + 2 * int(blocks_.size()) + 1;
        firstWrite_.assign(targets_.size(), endEvent_);
        auto noteWrites = [&](int s) {
            for (const Feed& f : feeds_[std::size_t(s)])
                firstWrite_[std::size_t(f.target)] = std::min(firstWrite_[std::size_t(f.target)], writeEvent_[std::size_t(s)]);
        };
        for (int a : prog_.inputs)
            if (live_[std::size_t(a)]) noteWrites(a);
        for (int a : blocks_) noteWrites(a);
    }

    // Ports of the given multipliers plus output groups; every target's form is split into
    // one table per source.
    void buildTargets(const std::vector<int>& blockAtoms) {
        blocks_ = blockAtoms;
        targets_.clear();
        for (std::size_t b = 0; b < blocks_.size(); ++b) {
            const auto& atom = prog_.atoms[std::size_t(blocks_[b])];
            const Value& fa = prog_.forms[std::size_t(atom.formA)];
            const Value& fb = prog_.forms[std::size_t(atom.formB)];
            targets_.push_back({&fa.terms, fa.k, int(b), 0, -1});
            targets_.push_back({&fb.terms, fb.k, int(b), 1, -1});
        }
        for (std::size_t g = 0; g < groupTerms_.size(); ++g) targets_.push_back({&groupTerms_[g], 0, -1, 0, int(g)});
        feeds_.assign(prog_.atoms.size(), {});
        for (std::size_t t = 0; t < targets_.size(); ++t) {
            std::map<int, std::vector<u8>> perSource;
            for (const auto& [atom, c] : *targets_[t].terms) {
                auto& acc = perSource[srcOf_[std::size_t(atom)]];
                acc.resize(256);
                for (unsigned v = 0; v < 256; ++v) acc[v] = u8(acc[v] + unsigned(c) * table_[std::size_t(atom)][v]);
            }
            for (auto& [s, tab] : perSource)
                if (std::any_of(tab.begin(), tab.end(), [](u8 x) { return x != 0; }))
                    feeds_[std::size_t(s)].push_back({int(t), std::move(tab)});
        }
    }

    // -------------------------------------------------------------- layout

    int multiplyEvent(std::size_t b) const { return mulEvent0_ + 2 * int(b); }

    const Shape& shape(std::size_t b) const { return compact_[b] ? kCompactShape : kCascadeShape; }
    long cellPos(const Layout& L, std::size_t b, long off) const {
        return L.base[b] + (L.mirror[b] ? shape(b).width - 1 - off : off);
    }
    long groupPos(const Layout& L, std::size_t g) const {
        int b = groupBlock_[g];
        return b >= 0 ? cellPos(L, std::size_t(b), shape(std::size_t(b)).w) : L.gpos[g];
    }
    long targetPos(const Layout& L, int t) const {
        const Target& tg = targets_[std::size_t(t)];
        if (tg.group >= 0) return groupPos(L, std::size_t(tg.group));
        const auto b = std::size_t(tg.block);
        return cellPos(L, b, tg.port == 0 ? shape(b).u : shape(b).v);
    }
    long inputCell(const Layout& L, int atom) const {
        int t = inputAlias_[std::size_t(atom)];
        return t >= 0 ? targetPos(L, t) : L.ipos[std::size_t(atom)];
    }

    void occupancy(const Layout& L, std::vector<Occupancy>& occ) const {
        occ.clear();
        for (int a : distInputs_) {
            occ.push_back({L.ipos[std::size_t(a)], 0, writeEvent_[std::size_t(a)]});
            if (cascade_[std::size_t(a)]) occ.push_back({L.ipos[std::size_t(a)] + 1, 0, writeEvent_[std::size_t(a)]});
        }
        for (std::size_t b = 0; b < blocks_.size(); ++b) {
            const int m = multiplyEvent(b);
            const bool distributes = wAlias_[b] < 0;
            const bool cascades = cascade_[std::size_t(blocks_[b])];
            const Shape& sh = shape(b);
            occ.push_back({cellPos(L, b, sh.u), firstWrite_[2 * b], m});
            occ.push_back({cellPos(L, b, sh.v), firstWrite_[2 * b + 1], m});
            for (long off : sh.scratch) occ.push_back({cellPos(L, b, off), m, m});
            occ.push_back({cellPos(L, b, sh.wFlag), m, distributes && cascades ? m + 1 : m});
            if (distributes) occ.push_back({cellPos(L, b, sh.w), m, m + 1});
        }
        for (std::size_t g = 0; g < groupTerms_.size(); ++g)
            occ.push_back({groupPos(L, g), firstWrite_[2 * blocks_.size() + g], endEvent_});
    }

    bool feasible(const Layout& L, std::vector<Occupancy>& occ) const {
        occupancy(L, occ);
        std::sort(occ.begin(), occ.end(),
                  [](const Occupancy& x, const Occupancy& y) { return std::tie(x.pos, x.start) < std::tie(y.pos, y.start); });
        long pos = LONG_MIN;
        int maxEnd = 0;
        for (const Occupancy& o : occ) {
            if (o.pos != pos) {
                pos = o.pos;
                maxEnd = o.end;
                continue;
            }
            if (o.start <= maxEnd) return false;
            maxEnd = std::max(maxEnd, o.end);
        }
        return true;
    }

    long spanOf(const Layout& L, long src, int atom) const {
        long lo = src, hi = src;
        for (const auto& use : uses_[std::size_t(atom)]) {
            long p = targetPos(L, use.first);
            lo = std::min(lo, p);
            hi = std::max(hi, p);
        }
        return hi - lo;
    }

    // Estimated layout-dependent steps: distribution loop travel plus pointer walks.
    long cost(const Layout& L) const {
        long c = 0, at = 0;
        auto walk = [&](long p) {
            c += std::abs(p - at);
            at = p;
        };
        for (int a : prog_.inputs)
            if (live_[std::size_t(a)]) walk(inputCell(L, a));
        // A cascade walks once from its flag to all targets; a loop walks every unit.
        auto distribution = [&](int s, long src, long flag) {
            if (!cascade_[std::size_t(s)]) {
                c += 2 * kAvgValue * spanOf(L, src, s);
                return;
            }
            long lo = std::min(src, flag), hi = std::max(src, flag);
            for (const Feed& f : feeds_[std::size_t(s)]) {
                lo = std::min(lo, targetPos(L, f.target));
                hi = std::max(hi, targetPos(L, f.target));
            }
            c += 2 * (hi - lo);
        };
        for (int a : distInputs_) {
            long p = L.ipos[std::size_t(a)];
            walk(p);
            distribution(a, p, p + 1);
        }
        for (std::size_t b = 0; b < blocks_.size(); ++b) {
            walk(cellPos(L, b, shape(b).u));
            if (wAlias_[b] >= 0) continue;
            long w = cellPos(L, b, shape(b).w);
            walk(w);
            distribution(blocks_[b], w, cellPos(L, b, shape(b).wFlag));
        }
        for (const auto& [g, k] : printSeq_)
            if (g >= 0) walk(groupPos(L, std::size_t(g)));
        return c;
    }

    void layout() {
        Layout L;
        L.base.resize(blocks_.size());
        L.mirror.assign(blocks_.size(), 0);
        L.gpos.assign(groupTerms_.size(), 0);
        L.ipos.assign(prog_.atoms.size(), 0);
        long pos = 0;
        for (std::size_t b = 0; b < blocks_.size(); ++b) {
            L.base[b] = pos;
            pos += shape(b).width;
        }
        std::vector<std::size_t> standalone;
        for (std::size_t g = 0; g < groupTerms_.size(); ++g)
            if (groupBlock_[g] < 0) {
                standalone.push_back(g);
                L.gpos[g] = pos++;
            }
        for (int a : distInputs_) {
            L.ipos[std::size_t(a)] = pos;
            pos += cascade_[std::size_t(a)] ? 2 : 1;
        }

        std::vector<Occupancy> occ;
        long curCost = cost(L);
        const std::size_t nBlocks = blocks_.size(), nGroups = standalone.size(), nInputs = distInputs_.size();
        const std::size_t nItems = nBlocks + nGroups + nInputs;
        if (nItems > 1) {
            std::mt19937 rng(12345);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
            long lo = 0, hi = pos;
            for (long iter = 0; iter < 400000; ++iter) {
                if ((iter & 255) == 0 && std::chrono::steady_clock::now() > deadline) break;
                Layout cand = L;
                std::size_t item = rng() % nItems;
                long span = hi - lo + 2 * kMaxBlockWidth;
                long anywhere = lo - kMaxBlockWidth + long(rng() % std::size_t(span));
                if (item < nBlocks) {
                    switch (rng() % 4) {
                        case 0: cand.base[item] += long(rng() % 19) - 9; break;
                        case 1: cand.mirror[item] ^= 1; break;
                        case 2: cand.base[item] = anywhere; break;
                        default: std::swap(cand.base[item], cand.base[rng() % nBlocks]); break;
                    }
                } else if (item < nBlocks + nGroups) {
                    std::size_t g = standalone[item - nBlocks];
                    cand.gpos[g] = (rng() & 1) ? anywhere : cand.gpos[g] + long(rng() % 7) - 3;
                } else {
                    auto a = std::size_t(distInputs_[item - nBlocks - nGroups]);
                    cand.ipos[a] = (rng() & 1) ? anywhere : cand.ipos[a] + long(rng() % 7) - 3;
                }
                if (!feasible(cand, occ)) continue;
                long c = cost(cand);
                if (c > curCost) continue;
                curCost = c;
                L = std::move(cand);
                lo = occ.front().pos;
                hi = occ.back().pos;
            }
        }
        // Shift everything to non-negative positions.
        if (!feasible(L, occ)) throw std::logic_error("internal error: overlapping cell layout");
        occupancy(L, occ);
        long minPos = 0;
        for (const Occupancy& o : occ) minPos = std::min(minPos, o.pos);
        for (long& p : L.base) p -= minPos;
        for (long& p : L.gpos) p -= minPos;
        for (long& p : L.ipos) p -= minPos;
        lay_ = std::move(L);
    }

    // ------------------------------------------------------------ emission

    bool isNonzero(long p) const { return p < long(nz_.size()) && nz_[std::size_t(p)]; }
    void setNonzero(long p, bool v) {
        if (p >= long(nz_.size())) nz_.resize(std::size_t(p) * 2 + 64, 0);
        nz_[std::size_t(p)] = v;
    }

    // Destructively adds src * coeff to every target in a single loop.
    void emitLoop(long src, const std::vector<Add>& adds) {
        std::vector<Add> sorted = adds;
        std::sort(sorted.begin(), sorted.end(), [](const Add& x, const Add& y) { return x.pos < y.pos; });
        e_.move(src);
        e_.open();
        e_.add(-1);
        for (auto it = sorted.rbegin(); it != sorted.rend(); ++it)
            if (it->pos < src) e_.addAt(it->pos, it->coeff);
        for (const Add& a : sorted)
            if (a.pos > src) e_.addAt(a.pos, a.coeff);
        e_.move(src);
        e_.close();
    }

    long zeroCellNear(long p, const std::vector<long>& excluded) const {
        for (long d = 1;; ++d)
            for (long q : {p - d, p + d})
                if (q >= 0 && !isNonzero(q) && std::find(excluded.begin(), excluded.end(), q) == excluded.end()) return q;
    }

    // Cheapest multiplier m with from * m == to (mod 256); mult < 0 when unreachable.
    struct Step {
        short mult = -1;
        short cost = SHRT_MAX;
    };
    static const std::vector<Step>& stepTable() {
        static const std::vector<Step> table = [] {
            std::vector<Step> t(256 * 256);
            for (unsigned from = 0; from < 256; ++from)
                for (unsigned m = 1; m < 256; ++m) {
                    Step& s = t[from * 256 + ((from * m) & 255U)];
                    if (addCost(m) < s.cost) s = {short(m), short(addCost(m))};
                }
            return t;
        }();
        return table;
    }

    // Plans the loops that add src * coeff into every target.  Node i of a multiplication
    // tree holds value[i] * src (node 0 is src itself, the rest are temporaries); each target
    // hangs off the node reaching its coefficient with the fewest '+'/'-'.  Temporaries are
    // added greedily while the estimated per-unit cost (overhead + coefficients + travel)
    // drops; every loop runs about as often as src, so per-unit costs simply add up.
    std::vector<LoopSpec> planLoops(long src, const std::vector<Add>& adds) const {
        const auto& table = stepTable();
        struct Node {
            long pos;
            unsigned value;
            int parent;
        };
        std::vector<Node> nodes{{src, 1, -1}};
        std::vector<int> assign(adds.size());
        std::vector<long> lo, hi, cost;
        auto evaluate = [&](const std::vector<Node>& ns) {
            lo.assign(ns.size(), 0);
            hi.assign(ns.size(), 0);
            cost.assign(ns.size(), 2);
            for (std::size_t i = 0; i < ns.size(); ++i) lo[i] = hi[i] = ns[i].pos;
            auto attach = [&](std::size_t parent, long pos, long c) {
                cost[parent] += c;
                lo[parent] = std::min(lo[parent], pos);
                hi[parent] = std::max(hi[parent], pos);
            };
            for (std::size_t i = 1; i < ns.size(); ++i)
                attach(std::size_t(ns[i].parent), ns[i].pos, table[ns[std::size_t(ns[i].parent)].value * 256 + ns[i].value].cost);
            for (std::size_t t = 0; t < adds.size(); ++t) {
                std::size_t best = 0;
                for (std::size_t j = 1; j < ns.size(); ++j)
                    if (table[ns[j].value * 256 + adds[t].coeff].cost < table[ns[best].value * 256 + adds[t].coeff].cost) best = j;
                assign[t] = int(best);
                attach(best, adds[t].pos, table[ns[best].value * 256 + adds[t].coeff].cost);
            }
            long total = 0;
            for (std::size_t i = 0; i < ns.size(); ++i) total += cost[i] + 2 * (hi[i] - lo[i]);
            return total;
        };
        long current = evaluate(nodes);
        std::vector<long> excluded{src};
        for (const Add& a : adds) excluded.push_back(a.pos);
        for (int round = 0; round < 4; ++round) {
            const long tmp = zeroCellNear(src, excluded);
            long best = current;
            Node bestNode{};
            std::vector<Node> cand = nodes;
            cand.push_back({tmp, 0, 0});
            for (std::size_t parent = 0; parent < nodes.size(); ++parent)
                for (unsigned value = 2; value < 256; ++value) {
                    if (table[nodes[parent].value * 256 + value].mult < 0) continue;
                    cand.back() = {tmp, value, int(parent)};
                    if (long c = evaluate(cand); c < best) {
                        best = c;
                        bestNode = cand.back();
                    }
                }
            if (best >= current) break;
            nodes.push_back(bestNode);
            excluded.push_back(tmp);
            current = best;
        }
        evaluate(nodes);
        std::vector<LoopSpec> loops(nodes.size());
        for (std::size_t i = 0; i < nodes.size(); ++i) loops[i].src = nodes[i].pos;
        for (std::size_t i = 1; i < nodes.size(); ++i) {
            const Node& n = nodes[i];
            loops[std::size_t(n.parent)].adds.push_back({n.pos, u8(table[nodes[std::size_t(n.parent)].value * 256 + n.value].mult)});
        }
        for (std::size_t t = 0; t < adds.size(); ++t) {
            const Node& n = nodes[std::size_t(assign[t])];
            loops[std::size_t(assign[t])].adds.push_back({adds[t].pos, u8(table[n.value * 256 + adds[t].coeff].mult)});
        }
        return loops;
    }

    void distribute(long src, const std::vector<Add>& adds) {
        for (const LoopSpec& l : planLoops(src, adds)) {
            emitLoop(l.src, l.adds);
            setNonzero(l.src, false);
            for (const Add& a : l.adds) setNonzero(a.pos, true);
        }
    }

    std::vector<Add> useAdds(int atom) const {
        std::vector<Add> adds;
        for (const auto& [t, c] : uses_[std::size_t(atom)]) adds.push_back({targetPos(lay_, t), c});
        return adds;
    }

    // Runs caseBody(x) for the runtime value of x (consumed), leaving every cell it does not
    // touch unchanged.  255 nested "x[-" levels leave the code position at depth x when x hits
    // zero; the flag lets only that deepest case run, so the cost is ~6 steps per unit of x
    // plus the one case that executes.
    void emitCascade(long x, long flag, const std::function<void(int)>& caseBody) {
        e_.addAt(flag, 1);
        for (int level = 1; level < 256; ++level) {
            e_.move(x);
            e_.open();
            e_.add(-1);
        }
        for (int v = 255; v >= 0; --v) {
            e_.move(flag);
            e_.open();
            e_.add(-1);
            caseBody(v);
            e_.move(flag);
            e_.close();
            e_.move(x);
            if (v > 0) e_.close();
        }
        setNonzero(x, false);
    }

    // Adds table[x] to every listed cell with a cascade over source cell x.
    void emitTableCascade(long x, long flag, std::vector<std::pair<long, const std::vector<u8>*>> cells) {
        std::sort(cells.begin(), cells.end());
        emitCascade(x, flag, [&](int v) {
            for (auto it = cells.rbegin(); it != cells.rend(); ++it)
                if (it->first < flag) e_.addAt(it->first, (*it->second)[std::size_t(v)]);
            for (const auto& [pos, table] : cells)
                if (pos > flag) e_.addAt(pos, (*table)[std::size_t(v)]);
        });
        for (const auto& c : cells) setNonzero(c.first, true);
    }

    // Distributes source s (in cell x) by a cascade.  Normally each target receives its
    // table directly; for huge fan-outs every unary atom of s is cascaded into its own hub
    // cell instead, and the hubs are distributed linearly with loops.
    void emitSourceCascade(long x, long flag, int s) {
        const auto& feeds = feeds_[std::size_t(s)];
        std::vector<std::pair<long, const std::vector<u8>*>> cells;
        if (!hub_[std::size_t(s)]) {
            for (const Feed& f : feeds) cells.emplace_back(targetPos(lay_, f.target), &f.table);
            emitTableCascade(x, flag, std::move(cells));
            return;
        }
        std::map<int, std::vector<std::pair<int, u8>>> perAtom;  // unary atom -> (target, coeff)
        std::vector<long> excluded{x, flag};
        for (const Feed& f : feeds) {
            excluded.push_back(targetPos(lay_, f.target));
            for (const auto& [atom, c] : *targets_[std::size_t(f.target)].terms)
                if (srcOf_[std::size_t(atom)] == s) perAtom[atom].emplace_back(f.target, c);
        }
        std::vector<std::pair<long, std::vector<Add>>> hubs;
        for (const auto& [atom, uses] : perAtom) {
            const long hub = zeroCellNear(x, excluded);
            excluded.push_back(hub);
            cells.emplace_back(hub, &table_[std::size_t(atom)]);
            std::vector<Add> adds;
            for (const auto& [t, c] : uses) adds.push_back({targetPos(lay_, t), c});
            hubs.emplace_back(hub, std::move(adds));
        }
        emitTableCascade(x, flag, std::move(cells));
        for (const auto& [hub, adds] : hubs) distribute(hub, adds);
    }

    // W += U * V (mod 256); U and V are consumed, all scratch cells end at zero.
    void emitMultiply(std::size_t b, long w, u8 ku, u8 kv) {
        auto at = [&](long off) { return cellPos(lay_, b, off); };
        e_.addAt(at(shape(b).u), ku);
        e_.addAt(at(shape(b).v), kv);
        if (compact_[b])
            emitCompactMultiply(at, w);
        else
            emitCascadeMultiply(at, w);
        for (long off = 0; off < shape(b).width; ++off) setNonzero(at(off), false);
        setNonzero(w, true);
    }

    // A cascade over U makes u a compile-time constant in each case, which then adds u * V
    // into W along the cheapest factor chain.
    template <class At>
    void emitCascadeMultiply(const At& at, long w) {
        const long u = at(0), flag = at(1), v = at(2), t1 = at(3), t2 = at(5);
        emitCascade(u, flag, [&](int value) {
            const Chain& c = chainTable()[std::size_t(value)];
            auto stage = [&](long src, u8 toW, long next, u8 toNext) {
                std::vector<Add> adds;
                if (toW) adds.push_back({w, toW});
                if (toNext) adds.push_back({next, toNext});
                emitLoop(src, adds);
            };
            stage(v, c.a, t1, c.f);
            if (c.f) stage(t1, c.b, t2, c.g);
            if (c.g) stage(t2, c.h, -1, 0);
        });
    }

    // Base-4 shift-and-add: each digit of U is peeled off with two halvings, then V (shifted
    // left by two bits per digit) is added 0..3 times while being moved to the next shift.
    template <class At>
    void emitCompactMultiply(const At& at, long w) {
        const long m2 = at(1), m1 = at(2), p = at(3), p1 = at(4), va = at(6), vb = at(8);
        // u -> (u / 2 in q, u % 2 in r); r and r + (r - u) must be zero, |r - u| == 1.
        auto halve = [&](long u, long r, long q) {
            const bool right = r > u;
            e_.move(u);
            e_.open();
            e_.add(-1);
            e_.addAt(r, 1);
            e_.move(u);
            e_.open();
            e_.add(-1);
            e_.addAt(q, 1);
            e_.addAt(r, -1);
            e_.close();  // pointer is at r if the second unit existed, else at u
            e_.raw(right ? ">[>]" : "<[<]", r + (r - u));
            e_.move(u);
            e_.close();
        };
        auto vloop = [&](long v, int d, long vn) {
            std::vector<Add> adds;
            if (d) adds.push_back({w, u8(d)});
            if (vn >= 0) adds.push_back({vn, u8(4)});
            emitLoop(v, adds);
        };
        std::function<void(int)> digit = [&](int j) {
            const long v = j % 2 == 0 ? va : vb, vn = j % 2 == 0 ? vb : va;
            if (j == 3) {  // p holds u >> 6, at most 3
                e_.move(p);
                e_.open();
                e_.add(-1);
                e_.open();
                e_.add(-1);
                e_.open();
                e_.add(-1);
                vloop(v, 3, -1);
                e_.move(p);
                e_.close();
                vloop(v, 2, -1);
                e_.move(p);
                e_.close();
                vloop(v, 1, -1);
                e_.move(p);
                e_.close();
                return;
            }
            halve(p, p1, m1);  // bit 0 -> p1
            halve(m1, m2, p);  // bit 1 -> m2, rest back in p
            // Adds digit * v to w; every branch consumes v, so later branches are skipped.
            auto dispatch = [&](long vnext) {
                e_.move(m2);
                e_.open();
                e_.add(-1);
                e_.move(p1);
                e_.open();
                e_.add(-1);
                vloop(v, 3, vnext);
                e_.move(p1);
                e_.close();
                vloop(v, 2, vnext);
                e_.move(m2);
                e_.close();
                e_.move(p1);
                e_.open();
                e_.add(-1);
                vloop(v, 1, vnext);
                e_.move(p1);
                e_.close();
            };
            e_.move(p);
            e_.open();  // more digits follow: shift v while adding
            dispatch(vn);
            vloop(v, 0, vn);
            digit(j + 1);
            e_.move(p);
            e_.close();
            dispatch(-1);  // last digit: no shift needed
        };
        digit(0);
        emitLoop(va, {});  // v is left over only when u was 0
    }

    void emit() {
        // 1. Read inputs up to the last live one; dead ones land in the next live input's cell.
        long lastLive = -1;
        for (std::size_t i = 0; i < prog_.inputs.size(); ++i)
            if (live_[std::size_t(prog_.inputs[i])]) lastLive = long(i);
        std::vector<long> readCell(std::size_t(lastLive + 1));
        long next = -1;
        for (long i = lastLive; i >= 0; --i) {
            int a = prog_.inputs[std::size_t(i)];
            if (live_[std::size_t(a)]) next = inputCell(lay_, a);
            readCell[std::size_t(i)] = next;
        }
        for (long c : readCell) {
            e_.move(c);
            e_.put(',');
            setNonzero(c, true);
        }
        // 2. Distribute inputs.
        for (int a : distInputs_) {
            const long cell = lay_.ipos[std::size_t(a)];
            if (cascade_[std::size_t(a)])
                emitSourceCascade(cell, cell + 1, a);
            else
                distribute(cell, useAdds(a));
        }
        // 3. Products in dependency order.
        for (std::size_t b = 0; b < blocks_.size(); ++b) {
            const Shape& sh = shape(b);
            long w = wAlias_[b] >= 0 ? groupPos(lay_, std::size_t(wAlias_[b])) : cellPos(lay_, b, sh.w);
            emitMultiply(b, w, u8(targets_[2 * b].k - preset_[2 * b]), u8(targets_[2 * b + 1].k - preset_[2 * b + 1]));
            if (wAlias_[b] >= 0) continue;
            const int atom = blocks_[b];
            if (cascade_[std::size_t(atom)])
                emitSourceCascade(w, cellPos(lay_, b, sh.wFlag), atom);
            else
                distribute(w, useAdds(atom));
        }
        // 4. Prints.
        // Groups that are never written (their terms cancel for every input) print constants.
        std::vector<std::pair<int, u8>> prints = printSeq_;
        std::vector<long> groupCells;
        for (auto& [g, k] : prints) {
            if (g < 0) continue;
            if (firstWrite_[2 * blocks_.size() + std::size_t(g)] >= endEvent_)
                g = -1;
            else
                groupCells.push_back(groupPos(lay_, std::size_t(g)));
        }
        std::vector<int> current(groupTerms_.size());
        for (std::size_t g = 0; g < groupTerms_.size(); ++g) current[g] = preset_[2 * blocks_.size() + g];
        long constCell = -1;
        int constValue = 0;
        bool anyConst = false;
        std::vector<long> neighbours;
        for (std::size_t i = 0; i < prints.size(); ++i) {
            if (prints[i].first >= 0) continue;
            anyConst = true;
            for (std::size_t j = i; j-- > 0;)
                if (prints[j].first >= 0) {
                    neighbours.push_back(groupPos(lay_, std::size_t(prints[j].first)));
                    break;
                }
            for (std::size_t j = i + 1; j < prints.size(); ++j)
                if (prints[j].first >= 0) {
                    neighbours.push_back(groupPos(lay_, std::size_t(prints[j].first)));
                    break;
                }
        }
        if (anyConst) {
            long anchor = e_.pos;
            if (!neighbours.empty()) {
                std::nth_element(neighbours.begin(), neighbours.begin() + long(neighbours.size() / 2), neighbours.end());
                anchor = neighbours[neighbours.size() / 2];
            }
            const bool usable = !isNonzero(anchor) && std::find(groupCells.begin(), groupCells.end(), anchor) == groupCells.end();
            constCell = usable ? anchor : zeroCellNear(anchor, groupCells);
        }
        for (const auto& [g, k] : prints) {
            if (g < 0) {
                e_.move(constCell);
                e_.add(int(k) - constValue);
                constValue = k;
            } else {
                e_.move(groupPos(lay_, std::size_t(g)));
                e_.add(int(k) - current[std::size_t(g)]);
                current[std::size_t(g)] = k;
            }
            e_.put('.');
        }
    }
};

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: exprc input.expr > output.bf\n";
        return 43;
    }
    std::ifstream source(argv[1], std::ios::binary);
    if (!source) {
        std::cerr << "Error: cannot open source file\n";
        return 43;
    }
    try {
        Program program;
        Parser parser(program);
        std::string line;
        int lineNo = 0;
        while (std::getline(source, line)) {
            ++lineNo;
            parser.statement(lexLine(line, lineNo), lineNo);
        }
        std::string code = Compiler(program).run();
        std::cout << code << '\n';
    } catch (const CompileError& err) {
        std::cerr << "Error: line " << err.line << ": " << err.message << '\n';
        return 43;
    } catch (const std::exception& err) {
        std::cerr << "Error: " << err.what() << '\n';
        return 43;
    }
    return 0;
}
