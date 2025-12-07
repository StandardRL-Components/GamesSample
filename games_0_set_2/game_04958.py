
# Generated: 2025-08-28T03:32:07.911025
# Source Brief: brief_04958.md
# Brief Index: 4958

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Space to pick up/place cards. Shift to cancel selection."
    )

    game_description = (
        "A solitaire-style card puzzle. Move all cards to the four foundations (top left) by suit, from Ace to King."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    CARD_WIDTH, CARD_HEIGHT = 40, 60
    CARD_ARC = 5
    STACK_Y_OFFSET = 15
    CURSOR_SPEED = 15
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (20, 80, 50)
    COLOR_CARD_BG = (240, 240, 240)
    COLOR_CARD_BACK = (60, 60, 120)
    COLOR_CARD_OUTLINE = (0, 0, 0)
    COLOR_RED_SUIT = (210, 40, 40)
    COLOR_BLACK_SUIT = (20, 20, 20)
    COLOR_FOUNDATION = (25, 95, 60)
    COLOR_CURSOR = (255, 255, 0, 150)
    COLOR_TEXT = (220, 220, 220)
    COLOR_GAMEOVER_BG = (0, 0, 0, 180)
    COLOR_PARTICLE = (255, 223, 0)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_card = pygame.font.SysFont("arial", 14, bold=True)
        self.font_ui = pygame.font.SysFont("tahoma", 18, bold=True)
        self.font_gameover = pygame.font.SysFont("impact", 48)

        self.suits = ['H', 'D', 'C', 'S']
        self.ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self._create_deck()
        self._deal_cards()
        
        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        
        self.held_cards = []
        self.held_from_info = None
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _create_deck(self):
        self.deck = []
        for suit in self.suits:
            for i, rank in enumerate(self.ranks):
                color = 'red' if suit in ['H', 'D'] else 'black'
                self.deck.append({'rank_val': i + 1, 'rank_str': rank, 'suit': suit, 'color': color, 'face_up': False})

    def _deal_cards(self):
        random.shuffle(self.deck)
        
        self.tableau_piles = [[] for _ in range(7)]
        self.foundations = [[] for _ in range(4)]
        
        # Deal cards into 7 tableau piles
        card_idx = 0
        for i in range(7):
            for j in range(i + 1):
                if card_idx < len(self.deck):
                    self.tableau_piles[i].append(self.deck[card_idx])
                    card_idx += 1
        
        # Deal remaining cards
        while card_idx < len(self.deck):
            for i in range(7):
                if card_idx < len(self.deck):
                    self.tableau_piles[i].append(self.deck[card_idx])
                    card_idx += 1

        # Flip top card of each tableau pile
        for pile in self.tableau_piles:
            if pile:
                pile[-1]['face_up'] = True

        self._calculate_pile_rects()

    def _calculate_pile_rects(self):
        self.foundation_rects = []
        for i in range(4):
            x = 10 + i * (self.CARD_WIDTH + 10)
            y = 10
            self.foundation_rects.append(pygame.Rect(x, y, self.CARD_WIDTH, self.CARD_HEIGHT))

        self.tableau_rects = []
        for i in range(7):
            x = 10 + i * (self.CARD_WIDTH + 20)
            y = self.CARD_HEIGHT + 30
            self.tableau_rects.append(pygame.Rect(x, y, self.CARD_WIDTH, self.SCREEN_HEIGHT - y - 10))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        action_taken = False
        
        # --- Handle Input ---
        self._handle_movement(movement)
        
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        if space_pressed:
            if self.held_cards:
                # Try to place cards
                if self._try_place_cards():
                    action_taken = True
                    # Reward is handled in _try_place_cards
                else:
                    # SFX: illegal_move
                    pass
            else:
                # Try to pick up cards
                if self._try_pickup_cards():
                    action_taken = True
                    # SFX: pickup_card
                else:
                    # SFX: cannot_pickup
                    pass
        
        if shift_pressed and self.held_cards:
            # Cancel selection
            self._cancel_selection()
            action_taken = True
            # SFX: drop_card

        if not action_taken:
            reward -= 0.1

        # --- Update Game State ---
        self.steps += 1
        self._update_particles()
        
        # --- Check Termination ---
        terminated = False
        if sum(len(f) for f in self.foundations) == 52:
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
        elif not self._has_legal_moves():
            self.game_over = True
            terminated = True
            reward -= 100

        self.score += reward
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_movement(self, movement):
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED  # Up
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED  # Down
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED  # Left
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

    def _get_target_at_cursor(self):
        cursor_rect = pygame.Rect(self.cursor_pos[0] - 1, self.cursor_pos[1] - 1, 2, 2)

        # Check foundations
        for i, rect in enumerate(self.foundation_rects):
            if rect.colliderect(cursor_rect):
                return {"type": "foundation", "index": i}, -1

        # Check tableau piles (reverse order for correct stacking overlap)
        for i in range(len(self.tableau_piles) - 1, -1, -1):
            pile = self.tableau_piles[i]
            base_rect = self.tableau_rects[i]
            if not base_rect.colliderect(cursor_rect):
                continue
            
            # Check cards in pile from top to bottom
            for j in range(len(pile) - 1, -1, -1):
                card_rect = pygame.Rect(base_rect.x, base_rect.y + j * self.STACK_Y_OFFSET, self.CARD_WIDTH, self.CARD_HEIGHT)
                if card_rect.colliderect(cursor_rect):
                    return {"type": "tableau", "index": i}, j
            
            # If cursor is on pile area but not a specific card, target the pile itself
            return {"type": "tableau", "index": i}, -1

        return None, -1

    def _try_pickup_cards(self):
        target_info, card_idx = self._get_target_at_cursor()
        if not target_info or target_info["type"] != "tableau" or card_idx == -1:
            return False

        pile_idx = target_info["index"]
        pile = self.tableau_piles[pile_idx]
        card = pile[card_idx]

        if card['face_up']:
            self.held_cards = pile[card_idx:]
            self.held_from_info = {"type": "tableau", "index": pile_idx}
            self.tableau_piles[pile_idx] = pile[:card_idx]
            return True
        return False

    def _try_place_cards(self):
        target_info, _ = self._get_target_at_cursor()
        if not target_info:
            return False

        placed = False
        reward = 0

        if target_info["type"] == "foundation":
            pile_idx = target_info["index"]
            if self._is_legal_foundation_move(self.held_cards, self.foundations[pile_idx]):
                card = self.held_cards.pop(0)
                self.foundations[pile_idx].append(card)
                self._spawn_particles(self.foundation_rects[pile_idx].center, 10)
                reward += 5
                placed = True
                # SFX: success
        
        elif target_info["type"] == "tableau":
            pile_idx = target_info["index"]
            if self._is_legal_tableau_move(self.held_cards, self.tableau_piles[pile_idx]):
                self.tableau_piles[pile_idx].extend(self.held_cards)
                placed = True
                # SFX: place_card_wood

        if placed:
            # Handle revealing a card on the source pile
            if self.held_from_info and self.held_from_info["type"] == "tableau":
                source_pile_idx = self.held_from_info["index"]
                source_pile = self.tableau_piles[source_pile_idx]
                if source_pile and not source_pile[-1]['face_up']:
                    source_pile[-1]['face_up'] = True
                    reward += 1
            
            self.held_cards = []
            self.held_from_info = None
            self.score += reward
            return True

        return False

    def _cancel_selection(self):
        if self.held_from_info["type"] == "tableau":
            pile_idx = self.held_from_info["index"]
            self.tableau_piles[pile_idx].extend(self.held_cards)
        
        self.held_cards = []
        self.held_from_info = None

    def _is_legal_foundation_move(self, cards, foundation_pile):
        if len(cards) != 1: return False
        card = cards[0]
        if not foundation_pile:
            return card['rank_val'] == 1  # Ace
        else:
            top_card = foundation_pile[-1]
            return card['suit'] == top_card['suit'] and card['rank_val'] == top_card['rank_val'] + 1

    def _is_legal_tableau_move(self, cards, tableau_pile):
        if not cards: return False
        card_to_place = cards[0]
        if not tableau_pile:
            return card_to_place['rank_val'] == 13 # King on empty pile
        else:
            top_card = tableau_pile[-1]
            return (card_to_place['color'] != top_card['color'] and 
                    card_to_place['rank_val'] == top_card['rank_val'] - 1 and
                    top_card['face_up'])

    def _has_legal_moves(self):
        # Check tableau to tableau moves
        for i, source_pile in enumerate(self.tableau_piles):
            if not source_pile: continue
            
            first_face_up_idx = -1
            for k, card in enumerate(source_pile):
                if card['face_up']:
                    first_face_up_idx = k
                    break
            if first_face_up_idx == -1: continue

            for j in range(first_face_up_idx, len(source_pile)):
                stack_to_move = source_pile[j:]
                
                # Check moving to other tableau piles
                for dest_idx, dest_pile in enumerate(self.tableau_piles):
                    if i == dest_idx: continue
                    if self._is_legal_tableau_move(stack_to_move, dest_pile):
                        return True
                
                # Check moving to foundations (only single cards)
                if len(stack_to_move) == 1:
                    for found_pile in self.foundations:
                        if self._is_legal_foundation_move(stack_to_move, found_pile):
                            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "win": self.win}

    def _render_game(self):
        # Draw foundations
        for rect in self.foundation_rects:
            pygame.gfxdraw.box(self.screen, rect, self.COLOR_FOUNDATION)
            pygame.gfxdraw.rectangle(self.screen, rect, (*self.COLOR_TEXT, 50))

        # Draw tableau piles
        for i, pile in enumerate(self.tableau_piles):
            base_pos = self.tableau_rects[i].topleft
            if not pile:
                pygame.gfxdraw.box(self.screen, self.tableau_rects[i], self.COLOR_FOUNDATION)
                pygame.gfxdraw.rectangle(self.screen, self.tableau_rects[i], (*self.COLOR_TEXT, 50))
            else:
                for j, card in enumerate(pile):
                    pos = (base_pos[0], base_pos[1] + j * self.STACK_Y_OFFSET)
                    self._draw_card(card, pos)
        
        # Draw foundations content
        for i, pile in enumerate(self.foundations):
            if pile:
                self._draw_card(pile[-1], self.foundation_rects[i].topleft)
        
        # Draw held cards
        if self.held_cards:
            for i, card in enumerate(self.held_cards):
                pos = (self.cursor_pos[0] - self.CARD_WIDTH / 2, self.cursor_pos[1] - self.CARD_HEIGHT / 2 + i * self.STACK_Y_OFFSET)
                self._draw_card(card, pos, shadow=True)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, p['pos'], int(p['radius']))

        # Draw cursor
        cursor_rect = pygame.Rect(self.cursor_pos[0] - 5, self.cursor_pos[1] - 5, 10, 10)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 0, 3)

    def _draw_card(self, card, pos, shadow=False):
        card_rect = pygame.Rect(pos[0], pos[1], self.CARD_WIDTH, self.CARD_HEIGHT)
        
        if shadow:
            shadow_rect = card_rect.move(3, 3)
            pygame.gfxdraw.box(self.screen, shadow_rect, (0, 0, 0, 100))

        if not card['face_up']:
            pygame.gfxdraw.box(self.screen, card_rect, self.COLOR_CARD_BACK)
            pygame.gfxdraw.rectangle(self.screen, card_rect, self.COLOR_CARD_OUTLINE)
            # Add a simple pattern to the back
            inner_rect = card_rect.inflate(-8, -8)
            pygame.gfxdraw.rectangle(self.screen, inner_rect, (*self.COLOR_CARD_OUTLINE, 100))
            return

        pygame.gfxdraw.box(self.screen, card_rect, self.COLOR_CARD_BG)
        pygame.gfxdraw.rectangle(self.screen, card_rect, self.COLOR_CARD_OUTLINE)
        
        suit_color = self.COLOR_RED_SUIT if card['color'] == 'red' else self.COLOR_BLACK_SUIT
        
        # Rank
        rank_surf = self.font_card.render(card['rank_str'], True, suit_color)
        self.screen.blit(rank_surf, (pos[0] + 4, pos[1] + 2))
        
        # Suit symbol
        self._draw_suit_symbol(card['suit'], suit_color, (pos[0] + 7, pos[1] + 20))

    def _draw_suit_symbol(self, suit, color, pos):
        x, y = int(pos[0]), int(pos[1])
        if suit == 'H': # Heart
            pygame.gfxdraw.filled_polygon(self.screen, [(x, y+2), (x+5, y-3), (x+10, y+2), (x+5, y+7)], color)
        elif suit == 'D': # Diamond
            pygame.gfxdraw.filled_polygon(self.screen, [(x+5, y-4), (x+10, y+2), (x+5, y+8), (x, y+2)], color)
        elif suit == 'C': # Club
            pygame.gfxdraw.filled_circle(self.screen, x+3, y+4, 3, color)
            pygame.gfxdraw.filled_circle(self.screen, x+7, y+4, 3, color)
            pygame.gfxdraw.filled_circle(self.screen, x+5, y, 3, color)
            pygame.gfxdraw.box(self.screen, pygame.Rect(x+4, y, 2, 7), color)
        elif suit == 'S': # Spade
            pygame.gfxdraw.filled_polygon(self.screen, [(x, y+4), (x+10, y+4), (x+5, y-4)], color)
            pygame.gfxdraw.box(self.screen, pygame.Rect(x+4, y+4, 2, 4), color)

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        steps_text = self.font_ui.render(f"Moves: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 35))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_GAMEOVER_BG)
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            end_text = self.font_gameover.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            overlay.blit(end_text, text_rect)
            self.screen.blit(overlay, (0, 0))

    def _spawn_particles(self, pos, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': random.uniform(2, 5),
                'life': random.randint(15, 30)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['radius'] -= 0.1
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # Pygame event handling
        action = np.array([0, 0, 0]) # no-op, released, released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        game_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # Create a display if one doesn't exist
        try:
            display_surface = pygame.display.get_surface()
            if display_surface is None:
                raise AttributeError
        except (pygame.error, AttributeError):
            display_surface = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
            pygame.display.set_caption("Solitaire Gym Environment")

        display_surface.blit(game_surface, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Win: {info['win']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()

        env.clock.tick(30) # Limit to 30 FPS

    env.close()