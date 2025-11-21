#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <utility>
#include <tuple>
#include <vector>
#include <ranges>

namespace py = pybind11;
using namespace std;

using card_t = tuple<int, int, int>;

enum {
	TYPE,
	SUIT,
	NUMBER
};

card_t tok_to_card(int tok) {
	if(tok < 52) return {tok >= 48, tok % 4, tok / 4};
	else return {2, 0, tok - 52};
}

int card_to_tok(const card_t &card) {
	auto &&[type, suit, number] = card;
	return type <= 1 ? 4 * number + suit : 52 + number;
}

/**
 * Output: (type, suit, number)
 *   type - 2=joker, 1=level card, 0=normal
 *   suit - 0=h,1=d,2=s,3=c
 */
card_t id_to_card(int id, int level) {
	id %= 54;
	if(id >= 52) return {2, 0, id - 52};
	int suit = id % 4, number = id / 4;
	number = (number + 12) % 13;
	if(number == level) number = 12;
	else if(number > level) number--;
	return {number == 12, suit, number};
}

int card_to_id(const card_t &card, int level) {
	auto [type, suit, number] = card;
	if(type == 2) return 52 + number;
	if(number == 12) number = level;
	else if(number >= level) number++;
	number = (number + 1) % 13;
	return 4 * number + suit;
}

bool is_major(const card_t &card, int major) {
	auto &&[type, suit, _] = card;
	return type >= 1 || suit == major;
}

bool match_suit(const card_t &card, int suit, int major) {
	return (get<TYPE>(card) == 0 && get<SUIT>(card) == suit)
		|| (suit == -1 && is_major(card, major));
}

bool consecutive(const card_t &a, const card_t &b) {
	return get<TYPE>(a) == get<TYPE>(b)
		&& get<SUIT>(a) == get<SUIT>(b)
		&& get<NUMBER>(a) + 1 == get<NUMBER>(b);
}

/**
 * Output: ascending pairs
 */
auto get_pairs(vector<int> cards, int level) {
	for(int &i: cards) i %= 54;
	sort(cards.begin(), cards.end());

	vector<card_t> pairs;
	card_t last = {-1, 0, 0};
	for(int i: cards) {
		auto card = id_to_card(i, level);
		if(card == last) {
			pairs.push_back(card);
			last = {-1, 0, 0};
		}
		else last = card;
	}

	return pairs;
}

/**
 * Input:
 *   sorted - whether cards is ascending
 */
template<bool sorted, typename T>
auto find(vector<T> &cards, const T &card) {
	if constexpr(sorted) {
		auto le = cards.begin(), ri = cards.end();
		while(le < ri) {
			auto mid = le + (ri - le) / 2;
			if(card < *mid) ri = mid;
			else if(*mid < card) le = ++mid;
			else return mid;
		}
		return cards.end();
	} else {
		return find(cards.begin(), cards.end(), card);
	}
}

/**
 * Split cards into singles and pairs.
 * Input:
 *   cards   - required to be ascending
 * Output:
 *   singles - ascending single cards
 *   pairs   - ascending pairs
 */
void process_cards(const vector<card_t> &cards, int suit, int major,
	vector<card_t> &singles, vector<card_t> &pairs) {
	card_t last = {-1, 0, 0};
	for(auto &&card: cards) {
		if(suit != 4 && !match_suit(card, suit, major))
			continue;
		if(card == last) {
			pairs.push_back(card);
			last = {-1, 0, 0};
		}
		else {
			if(get<TYPE>(last) >= 0)
				singles.push_back(last);
			last = card;
		}
	}
	if(get<TYPE>(last) >= 0)
		singles.push_back(last);
}

/**
 * Input:
 *   cards - required to be ascending
 */
auto consecutive_lengths(const vector<card_t> &cards) {
	int len = 0;
	vector<int> lengths;
	card_t last = {-1, 0, 0};
	for(auto &&card: cards) {
		if(consecutive(last, card)) {
			len++;
		} else {
			if(len > 0)
				lengths.push_back(len);
			len = 1;
		}
		last = card;
	}
	if(len > 0)
		lengths.push_back(len);
	return lengths;
}

/**
 * Input:
 *   pairs - required to be ascending
 */
bool can_match(const vector<card_t> &pairs, vector<int> len_tractor) {
	auto len_pairs = consecutive_lengths(pairs);
	sort(len_pairs.begin(), len_pairs.end(), greater<int>{});
	sort(len_tractor.begin(), len_tractor.end(), greater<int>{});

	auto mod_front = [](vector<int> &vec, int x) {
		vec[0] = x;
		for(size_t i = 0, size = vec.size(); i+1 < size; i++)
			if(vec[i] < vec[i+1]) swap(vec[i], vec[i+1]);
			else break;
		if(vec.back() == 0)
			vec.pop_back();
	};
	while(!len_pairs.empty() && !len_tractor.empty()) {
		int x = len_pairs[0], y = len_tractor[0];
		if(x < y) return false;
		mod_front(len_pairs, x - y);
		mod_front(len_tractor, 0);
	}
	return len_tractor.empty();
}

/**
 * Input:
 *   cards - required to be ascending
 */
bool can_follow(const vector<card_t> &singles, const vector<card_t> &pairs,
	size_t total, const vector<int> &len_tractor) {
	return singles.size() + 2 * pairs.size() >= total
		&& can_match(pairs, len_tractor);
}

/**
 * Input:
 *   pairs       - required to be ascending
 *   len_tractor - required not empty
 */
bool legal_action(card_t card, vector<card_t> pairs, const vector<int> &len_tractor) {
	int len = len_tractor.back();

	auto it = find<true>(pairs, card);
	if(it == pairs.end() || it < pairs.begin() + len - 1)
		return false;
	it -= len - 1;
	for(int i = 1; i < len; i++)
		if(!consecutive(*(it + i - 1), *(it + i)))
			return false;
	pairs.erase(it, it + len);

	return can_match(pairs, {len_tractor.begin(), prev(len_tractor.end())});
}

struct DealState {
	int level;
	vector<int> hand;
	DealState(int level_): level(level_) {}

	void action_mask(int id, int caller, int snatcher, py::array_t<bool> &out) {
		auto mask = out.mutable_unchecked();
		mask(10) = true;

		hand.push_back(id);
		if(caller == -1) {
			for(int i: hand) {
				auto &&[type, suit, number] = id_to_card(i, level);
				if(type == 1)
					mask(suit) = true;
			}
		} else if(snatcher == -1) {
			bool occured[6] {};
			for(int i: hand) {
				auto &&[type, suit, number] = id_to_card(i, level);
				if(type == 0)
					continue;
				int idx = type == 1 ? suit : 4 + number;
				if(occured[idx])
					mask(4 + idx) = true;
				else
					occured[idx] = true;
			}
		}
	}

	vector<int> tok_to_ids(int tok) {
		if(tok < 8) {
			int id = card_to_id({1, tok % 4, 12}, level);
			if(tok < 4) {
				if(find<false>(hand, id) == hand.end())
					id += 54;
				return {id};
			}
			else return {id, 54 + id};
		} else if(tok < 10) {
			int id = card_to_id({2, 0, tok - 8}, level);
			return {id, 54 + id};
		} else {
			return {};
		}
	}
};

struct CoverState {
	int level;
	vector<int> hand;
	CoverState(int level_, vector<int> &&hand_): level(level_), hand(move(hand_)) {}

	void action_mask(py::array_t<bool> &out) const {
		auto mask = out.mutable_unchecked();

		for(int id: hand)
			mask(card_to_tok(id_to_card(id, level))) = true;
	}

	vector<int> tok_to_ids(int tok) {
		int id = card_to_id(tok_to_card(tok), level);
		auto it = find<false>(hand, id);
		if(it == hand.end())
			it = find<false>(hand, id += 54);
		hand.erase(it);
		return {id};
	}
};

struct PlayState {
	static constexpr int kill_tok = 108, eos_tok = 109;
	int level, major, play_count;
	vector<int> hand_raw;
	vector<card_t> hand;
	enum {
		LEADING,
		FOLLOW,
		KILL,
		MAYBE_KILL,
		DISCARD,
	} play_type;
	int leading_suit, n_singles, n_pairs;
	vector<int> len_tractor;
	vector<card_t> singles_leading, pairs_leading, singles_major, pairs_major;
	PlayState(int level, int major_, vector<int> &&hand_, const vector<int> &leading):
		level(level), major(major_), play_count(0), hand_raw(move(hand_)) {
		sort(hand_raw.begin(), hand_raw.end());
		hand.reserve(hand_raw.size());
		for(int i: hand_raw)
			hand.push_back(id_to_card(i, level));
		sort(hand.begin(), hand.end());

		if(leading.empty()) {
			play_type = LEADING;
			leading_suit = 4;
			return;
		}

		auto &&[type, suit, number] = id_to_card(leading[0], level);
		leading_suit = is_major({type, suit, number}, major) ? -1: suit;
		auto leading_pairs = get_pairs(leading, level);
		n_pairs = (int)leading_pairs.size();
		n_singles = (int)leading.size() - 2 * n_pairs;
		len_tractor = consecutive_lengths(leading_pairs);

		process_cards(hand, leading_suit, major, singles_leading, pairs_leading);

		if(can_follow(singles_leading, pairs_leading, leading.size(), len_tractor)) {
			play_type = FOLLOW;
		} else if(singles_leading.empty() && pairs_leading.empty()) {
			process_cards(hand, -1, major, singles_major, pairs_major);
			if(can_follow(singles_major, pairs_major, leading.size(), len_tractor))
				play_type = MAYBE_KILL;
			else
				play_type = DISCARD;
		} else {
			play_type = DISCARD;
		}
	}

	void action_mask(py::array_t<bool> &out) {
		auto mask = out.mutable_unchecked();

		auto mask_action = [&mask](const vector<card_t> &cards,
			bool mask_single, bool mask_pair) {
			for(auto &&card: cards) {
				if(mask_single) mask(card_to_tok(card)) = true;
				if(mask_pair) mask(54 + card_to_tok(card)) = true;
			}
		};

		auto leading_action_mask = [&] {
			if(leading_suit == 4)
				process_cards(hand, leading_suit, major, singles_leading, pairs_leading);
			else
				mask(eos_tok) = true;

			mask_action(singles_leading, true, false);
			mask_action(pairs_leading, true, true);
		};

		auto follow_action_mask = [&] {
			if(n_singles + n_pairs == 0) {
				mask(eos_tok) = true;
			} else if(n_pairs == 0) {
				mask_action(singles_leading, true, false);
				mask_action(pairs_leading, true, false);
			} else {
				for(auto &&card: pairs_leading)
					if(legal_action(card, pairs_leading, len_tractor))
						mask(54 + card_to_tok(card)) = true;
			}
		};

		auto kill_action_mask = [&] {
			if(n_singles + n_pairs == 0) {
				mask(eos_tok) = true;
			} else if(n_pairs == 0) {
				mask_action(singles_major, true, false);
				mask_action(pairs_major, true, false);
			} else {
				for(auto &&card: pairs_major)
					if(legal_action(card, pairs_major, len_tractor))
						mask(54 + card_to_tok(card)) = true;
			}
		};

		auto discard_action_mask = [&] {
			if(n_singles + n_pairs == 0) {
				mask(eos_tok) = true;
			} else if(n_pairs > 0 && !pairs_leading.empty()) {
				mask_action(pairs_leading, false, true);
			} else if(!singles_leading.empty() || !pairs_leading.empty()) {
				mask_action(singles_leading, true, false);
				mask_action(pairs_leading, true, false);
			} else {
				mask_action(hand, true, false);
			}
		};

		switch(play_type)
		{
		case LEADING:
			leading_action_mask();
			return;
		case FOLLOW:
			follow_action_mask();
			return;
		case KILL:
			kill_action_mask();
			return;
		case MAYBE_KILL:
			mask(kill_tok) = true;
		case DISCARD:
			discard_action_mask();
			return;
		}
	}

	vector<int> tok_to_ids(int tok) {
		if(play_type == MAYBE_KILL) {
			play_type = tok == kill_tok ? KILL : DISCARD;
			if(tok == kill_tok)
				return {};
		}

		int is_pair = tok / 54;
		auto card = tok_to_card(tok % 54);

		vector<int> ids;
		auto deliver = [&](const card_t &card, int count) {
			int id = card_to_id(card, level);
			for(;count-- > 0; id += 54) {
				hand.erase(find<true>(hand, card));

				auto it = find<true>(hand_raw, id);
				if(it == hand_raw.end())
					it = find<true>(hand_raw, id += 54);
				hand_raw.erase(it);

				ids.push_back(id);
			}
		};

		auto erase_card = [](const card_t &card, bool is_single,
			vector<card_t> &singles, vector<card_t> &pairs) {
			auto it = find<true>(pairs, card);
			if(it == pairs.end()) {
				it = find<true>(singles, card);
				singles.erase(it);
			} else {
				pairs.erase(it);
				if(is_single) {
					it = upper_bound(singles.begin(), singles.end(), card);
					singles.insert(it, card);
				}
			}
		};

		auto leading_step = [&] {
			if(leading_suit == 4) {
				leading_suit = is_major(card, major) ? -1 : get<SUIT>(card);

				singles_leading.clear();
				pairs_leading.clear();
				process_cards(hand, leading_suit, major, singles_leading, pairs_leading);
			}

			play_count++;
			deliver(card, is_pair ? 2 : 1);
			erase_card(card, false, singles_leading, pairs_leading);
		};

		auto follow_step = [&] {
			if(!is_pair) {
				n_singles--;
				deliver(card, 1);
				erase_card(card, true, singles_leading, pairs_leading);
				return;
			}

			int len = len_tractor.back();
			len_tractor.pop_back();
			n_pairs -= len;

			auto &&[type, suit, number] = card;
			for(int i = 0; i < len; i++)
				deliver({type, suit, number - i}, 2);

			auto it = find<true>(pairs_leading, card);
			it -= len - 1;
			pairs_leading.erase(it, it + len);
		};

		auto kill_step = [&] {
			if(!is_pair) {
				n_singles--;
				deliver(card, 1);
				erase_card(card, true, singles_major, pairs_major);
				return;
			}

			int len = len_tractor.back();
			len_tractor.pop_back();
			n_pairs -= len;

			auto &&[type, suit, number] = card;
			for(int i = 0; i < len; i++)
				deliver({type, suit, number - i}, 2);

			auto it = find<true>(pairs_major, card);
			it -= len - 1;
			pairs_major.erase(it, it + len);
		};

		auto discard_step = [&] {
			(is_pair ? n_pairs : n_singles)--;

			deliver(card, is_pair ? 2 : 1);
			if(match_suit(card, leading_suit, major))
				erase_card(card, !is_pair, singles_leading, pairs_leading);

			if(!is_pair) {
				n_singles += 2 * n_pairs;
				n_pairs = 0;
			}
		};

		switch(play_type)
		{
		case LEADING:
			leading_step();
			break;
		case FOLLOW:
			follow_step();
			break;
		case KILL:
			kill_step();
			break;
		case MAYBE_KILL:
		case DISCARD:
			discard_step();
			break;
		}
		return ids;
	}
};

struct JudgerState {
	int level, major;
	int suit, card_num, max_card, winner;
	vector<int> len_tractor;
	vector<pair<card_t, card_t>> tractor_card;
	JudgerState(int _leval, int _major): level(_leval), major(_major) {}

	// 辅助变量
	int max_num[13];
	vector<card_t> cards, hand;
	int get_num(card_t card) {
		auto [type, suit, number] = card;
		if(type == 0) return number;
		else if(type == 1) return suit == major ? 13 : 12;
		else return number + 14;
	}

	auto parse_leader(int playerpos, vector<int> ids, array<vector<int>, 4> raw_hand) {
		fill(max_num, max_num + 13, -1);

		max_card = -1;
		card_num = ids.size();
		len_tractor.clear();
		tractor_card.clear();
		winner = playerpos;

		cards.clear(); cards.reserve(ids.size());
		for(int i: ids) cards.push_back(id_to_card(i, level));
		sort(cards.begin(), cards.end());
		suit = is_major(cards[0], major) ? -1 : get<SUIT>(cards[0]);
		// 对于每个 i，找到连续 i 张牌最大的牌最大是多少
		for(int player = 0; player < 4; player++) if(player != playerpos) {
			hand.clear(); hand.reserve(raw_hand[player].size());
			for(int i: raw_hand[player]) hand.push_back(id_to_card(i, level));
			sort(hand.begin(), hand.end());
			card_t last = {-1, 0, 0}, last_pair = {-1, 0, 0};
			int len = 0;
			for(auto &&card: hand) {
				if(!match_suit(card, suit, major)) continue;
				int num = get_num(card);
				if(card == last) {
					if(consecutive(card, last_pair)) len++;
					else len = 1;
					max_num[len] = max(max_num[len], num);
					last_pair = card;
				}
				max_num[0] = max(max_num[0], num);
				last = card;
			}
		}
		for(int i = 11; i >= 0; i--) max_num[i] = max(max_num[i], max_num[i + 1]);
		// 判断甩牌是否合法
		int len = 0, last_num = 20, last_pair_num = 20, failcnt = 0, min_fail_id = -1;
		card_t last = {-1, 0, 0}, last_pair = {-1, 0, 0}, min_fail_card = {3, 0, 0}, last_pair_first;
		for(auto &&card: cards) {
			if(card == last) {
				if(consecutive(card, last_pair)) len++;
				else {
					if(len) {
						len_tractor.push_back(len);
						tractor_card.emplace_back(last_pair_first, last_pair);
					}
					len = 1;
					last_pair_first = card;
				}
				last_pair = card;
				last = {-1, 0, 0};
			} else {
				if(get<TYPE>(last) != -1) {
					len_tractor.push_back(0);
					tractor_card.emplace_back(last, last);
				}
				last = card;
			}
		}
		if(len) {
			len_tractor.push_back(len);
			tractor_card.emplace_back(last_pair_first, last_pair);
		}
		if(get<TYPE>(last) != -1) {
			len_tractor.push_back(0);
			tractor_card.emplace_back(last, last);
		}
		if(len_tractor.size() <= 1) return make_pair(move(ids), 0);
		for(auto &&[len, card_pair]: views::zip(len_tractor, tractor_card)) {
			auto [first_card, last_card] = card_pair;
			int num = get_num(last_card);
			if(num < max_num[len]) {
				failcnt += len;
				if(get<SUIT>(first_card) == major) get<SUIT>(first_card) = 4;
				if(first_card < min_fail_card) {
					min_fail_card = first_card;
				}
			}
		}
		if(failcnt == 0) return make_pair(move(ids), 0);
		// 甩牌失败
		if(get<SUIT>(min_fail_card) == major) get<SUIT>(min_fail_card) = major;
		int id = card_to_id(min_fail_card, level);
		if(find(ids.begin(), ids.end(), id) != ids.end()) id += 54;
		ids = {id};
		return make_pair(move(ids), failcnt);
	}

	void parse_follow(int playerpos, vector<int> ids) {
		cards.clear(); cards.reserve(ids.size());
		for(int i: ids) cards.push_back(id_to_card(i, level));
		sort(cards.begin(), cards.end());
		if(len_tractor.size() <= 1) { // 不是甩牌
			bool all_major = true, same_suit = true;
			for(auto &&card: cards) {
				all_major &= is_major(card, major);
				same_suit &= match_suit(card, suit, major);
			}
			if(suit < 0) {
				if(!all_major) return;
				
			} else {
				
			}
		}
	}
};

PYBIND11_MODULE(tractor, m) {
    m.doc() = "C++ backend for Tractor";

	m.def("tok_to_card", &tok_to_card);
	m.def("card_to_tok", &card_to_tok);
	m.def("id_to_card", &id_to_card);
	m.def("card_to_id", &card_to_id);

	py::class_<DealState>(m, "DealState")
		.def(py::init<int>())
		.def("action_mask", &DealState::action_mask)
		.def("tok_to_ids", &DealState::tok_to_ids);

	py::class_<CoverState>(m, "CoverState")
		.def(py::init<int, vector<int>&&>())
		.def("action_mask", &CoverState::action_mask)
		.def("tok_to_ids", &CoverState::tok_to_ids);

	py::class_<PlayState>(m, "PlayState")
		.def(py::init<int, int, vector<int>&&, const vector<int>&>())
		.def_readonly_static("eos_tok", &PlayState::eos_tok)
		.def("action_mask", &PlayState::action_mask)
		.def("tok_to_ids", &PlayState::tok_to_ids);
}
