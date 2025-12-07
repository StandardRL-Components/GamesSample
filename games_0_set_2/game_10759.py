import gymnasium as gym
import os
import pygame
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:01:35.224236
# Source Brief: brief_00759.md
# Brief Index: 759
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Engage in a cyberpunk card duel. Manage momentum to play powerful programs, "
        "attack the opponent's system, and upgrade your processing power to win."
    )
    user_guide = "Use ←→ arrow keys to select a card or the upgrade button. Press space to play your selection."
    auto_advance = False

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CARD_WIDTH, CARD_HEIGHT = 80, 110
    HAND_Y_PLAYER = SCREEN_HEIGHT - CARD_HEIGHT - 10
    HAND_Y_OPPONENT = 10
    MAX_HEALTH = 100
    MAX_MOMENTUM = 10
    STARTING_HAND_SIZE = 4
    MAX_HAND_SIZE = 5
    UPGRADE_COST = 8
    MAX_EPISODE_STEPS = 500  # A turn can be multiple steps, so this is higher than game turns

    # --- COLORS (Cyberpunk Theme) ---
    COLOR_BG = (10, 20, 30)
    COLOR_GRID = (20, 40, 60)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_OPPONENT = (255, 50, 100)
    COLOR_MOMENTUM = (255, 200, 0)
    COLOR_SYSTEM = (50, 150, 255)
    COLOR_WHITE = (240, 240, 240)
    COLOR_GREY = (100, 100, 120)

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
        self.font_sm = pygame.font.Font(None, 20)
        self.font_md = pygame.font.Font(None, 28)
        self.font_lg = pygame.font.Font(None, 48)

        self.render_mode = render_mode
        self.wins = 0

        # State variables are initialized in reset()
        self.player_health = 0 # Placeholder, will be set in reset
        self.opponent_health = 0
        self.player_momentum = 0
        self.opponent_momentum = 0
        self.player_proc_power = 1
        self.opponent_proc_power = 1
        self.player_deck = []
        self.opponent_deck = []
        self.player_discard = []
        self.opponent_discard = []
        self.player_hand = []
        self.opponent_hand = []
        self.game_phase = 'PLAYER_TURN'
        self.selected_index = 0
        self.prev_action = np.array([0, 0, 0])
        self.game_over_timer = 0
        self.particles = []
        self.floating_texts = []
        self.selection_visual_pos = (0,0)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.opponent_start_health = 100


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Game state
        self.opponent_start_health = 100 + 50 * (self.wins // 5)
        self.player_health = self.MAX_HEALTH
        self.opponent_health = self.opponent_start_health

        self.player_momentum = 3
        self.opponent_momentum = 0

        self.player_proc_power = 1
        self.opponent_proc_power = 1

        self.player_deck = self._create_deck()
        self.opponent_deck = self._create_deck()
        self.player_discard = []
        self.opponent_discard = []

        self.player_hand = []
        for _ in range(self.STARTING_HAND_SIZE):
            self._draw_card('player')

        self.opponent_hand = []
        for _ in range(self.STARTING_HAND_SIZE):
            self._draw_card('opponent')

        # Control and UI state
        self.game_phase = 'PLAYER_TURN'  # PLAYER_TURN, OPPONENT_TURN, GAME_OVER
        self.selected_index = 0  # 0 to len(hand)-1 for cards, len(hand) for upgrade
        self.prev_action = np.array([0, 0, 0])
        self.game_over_timer = 0

        # Visual effects
        self.particles = []
        self.floating_texts = []
        self.selection_visual_pos = self._get_target_selection_pos()

        # Step tracking
        self.steps = 0
        self.score = 0
        self.game_over = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False
        turn_taken = False

        # --- Input Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not (self.prev_action[1] == 1)

        # Debounce movement for menu-like navigation
        if movement != 0 and self.prev_action[0] == 0:
            if self.game_phase == 'PLAYER_TURN':
                num_options = len(self.player_hand) + 1
                if movement == 3:  # Left
                    self.selected_index = (self.selected_index - 1 + num_options) % num_options
                elif movement == 4:  # Right
                    self.selected_index = (self.selected_index + 1) % num_options

        self.prev_action = action

        # --- Player Action Phase ---
        if self.game_phase == 'PLAYER_TURN' and space_pressed:
            # Selected a card
            if self.selected_index < len(self.player_hand):
                card = self.player_hand[self.selected_index]
                if self.player_momentum >= card['cost']:
                    self._play_card('player', self.selected_index)
                    reward += 1.0  # Reward for playing a card
                    turn_taken = True
            # Selected the upgrade button
            else:
                if self.player_momentum >= self.UPGRADE_COST and self.player_proc_power < 5:
                    self._upgrade_system('player')
                    reward += 5.0  # Reward for upgrading
                    turn_taken = True

        # --- Turn Progression ---
        if turn_taken:
            self.game_phase = 'PROCESSING'  # Short phase for animations
            if self.opponent_health <= 0:
                self.opponent_health = 0
                reward += 100
                self.score += 100
                self.wins += 1
                self.game_phase = 'GAME_OVER'
                self.game_over = True
            else:
                # --- Opponent's Turn ---
                self._run_opponent_turn()
                if self.player_health <= 0:
                    self.player_health = 0
                    reward -= 100
                    self.score -= 100
                    self.game_phase = 'GAME_OVER'
                    self.game_over = True
                else:
                    # --- End of Turn / Start of New Player Turn ---
                    self._draw_card('player')
                    self.player_momentum = min(self.MAX_MOMENTUM, self.player_momentum + self.player_proc_power)
                    self.selected_index = 0
                    self.game_phase = 'PLAYER_TURN'

        # --- Termination and Reward Calculation ---
        if self.game_over:
            terminated = True

        if self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True # Truncated should be True, but problem asks for terminated
            if not self.game_over:  # Penalize for timeout
                reward -= 20

        # Small continuous reward for damaging opponent
        # (Handled inside _play_card)

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False, # truncated
            self._get_info()
        )

    def _play_card(self, who, card_index):
        if who == 'player':
            hand, discard, deck = self.player_hand, self.player_discard, self.player_deck
            source_momentum = 'player_momentum'
            target_health = 'opponent_health'
            source_color = self.COLOR_PLAYER
            target_pos = (self.SCREEN_WIDTH / 2, 80)
        else:
            hand, discard, deck = self.opponent_hand, self.opponent_discard, self.opponent_deck
            source_momentum = 'opponent_momentum'
            target_health = 'player_health'
            source_color = self.COLOR_OPPONENT
            target_pos = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 80)

        card = hand.pop(card_index)
        discard.append(card)

        # Pay cost
        cost = card['cost']
        current_momentum = getattr(self, source_momentum)
        setattr(self, source_momentum, current_momentum - cost)

        # Apply effects
        damage = card['attack']
        if damage > 0:
            current_target_health = getattr(self, target_health)
            setattr(self, target_health, current_target_health - damage)
            self._spawn_floating_text(f"-{damage}", target_pos, self.COLOR_OPPONENT if who == 'player' else self.COLOR_PLAYER)
            # sfx: attack_hit.wav
            if who == 'player':
                self.score += damage * 0.1  # Small reward for damage

        momentum_gain = card['gain']
        if momentum_gain > 0:
            new_momentum = min(self.MAX_MOMENTUM, getattr(self, source_momentum) + momentum_gain)
            setattr(self, source_momentum, new_momentum)
            # sfx: momentum_gain.wav

        # Visuals
        card_pos = self._get_card_pos(card_index, len(hand) + 1, who)
        self._spawn_particles(20, card_pos, source_color, 2, 5, 0.5)
        if damage > 0:
            self._spawn_beam(card_pos, target_pos, source_color, 20)

    def _upgrade_system(self, who):
        if who == 'player':
            self.player_momentum -= self.UPGRADE_COST
            self.player_proc_power += 1
            pos = (self.SCREEN_WIDTH - 150, self.SCREEN_HEIGHT - 50)
            # sfx: upgrade.wav
        else:
            self.opponent_momentum -= self.UPGRADE_COST
            self.opponent_proc_power += 1
            pos = (150, 50)
            # sfx: upgrade.wav

        self._spawn_particles(30, pos, self.COLOR_SYSTEM, 1, 4, 1.0)
        self._spawn_floating_text("UPGRADE!", pos, self.COLOR_SYSTEM)

    def _run_opponent_turn(self):
        # AI Logic: Find the highest damage card it can play. If none, upgrade if possible.
        playable_cards = [(i, c) for i, c in enumerate(self.opponent_hand) if self.opponent_momentum >= c['cost']]

        if playable_cards:
            # Play best card (highest attack)
            best_card_index, _ = max(playable_cards, key=lambda item: item[1]['attack'])
            self._play_card('opponent', best_card_index)
        elif self.opponent_momentum >= self.UPGRADE_COST and self.opponent_proc_power < 5:
            self._upgrade_system('opponent')

        # Opponent momentum gain per turn
        self.opponent_momentum = min(self.MAX_MOMENTUM, self.opponent_momentum + self.opponent_proc_power)
        self._draw_card('opponent')

    def _draw_card(self, who):
        if who == 'player':
            hand, discard, deck = self.player_hand, self.player_discard, self.player_deck
            if len(hand) >= self.MAX_HAND_SIZE: return
        else:
            hand, discard, deck = self.opponent_hand, self.opponent_discard, self.opponent_deck
            if len(hand) >= self.MAX_HAND_SIZE: return

        if not deck:
            if not discard: return  # No cards left anywhere
            deck.extend(discard)
            discard.clear()
            self.np_random.shuffle(deck)
            # sfx: deck_shuffle.wav

        hand.append(deck.pop())
        # sfx: card_draw.wav

    def _create_deck(self):
        deck = []
        for _ in range(10):  # Basic Attack
            deck.append({'name': 'Virus', 'attack': self.np_random.integers(5, 12), 'cost': self.np_random.integers(1, 4), 'gain': 0})
        for _ in range(5):  # Momentum Gain
            deck.append({'name': 'Overclock', 'attack': 0, 'cost': 1, 'gain': self.np_random.integers(3, 6)})
        for _ in range(5):  # Heavy Attack
            deck.append({'name': 'Rootkit', 'attack': self.np_random.integers(15, 25), 'cost': self.np_random.integers(5, 8), 'gain': 0})
        self.np_random.shuffle(deck)
        return deck

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "opponent_health": self.opponent_health,
            "player_momentum": self.player_momentum,
        }

    def _render_all(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT), 1)
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i), 1)

        # --- Update and Render Effects ---
        self._update_and_draw_particles()
        self._update_and_draw_floating_texts()
        self._update_and_draw_selection()

        # --- Opponent Side ---
        self._render_hud('opponent')
        for i, card in enumerate(self.opponent_hand):
            pos = self._get_card_pos(i, len(self.opponent_hand), 'opponent')
            self._render_card_back(pos)

        # --- Player Side ---
        self._render_hud('player')
        for i, card in enumerate(self.player_hand):
            pos = self._get_card_pos(i, len(self.player_hand), 'player')
            is_selected = (self.selected_index == i) and self.game_phase == 'PLAYER_TURN'
            self._render_card(card, pos, is_selected)

        # --- Upgrade Button ---
        self._render_upgrade_button()

        # --- Game Over Screen ---
        if self.game_phase == 'GAME_OVER':
            self.game_over_timer += 1
            if self.game_over_timer > 30:  # Delay before showing text
                s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                s.fill((10, 20, 30, 200))
                self.screen.blit(s, (0, 0))
                msg = "CONNECTION ESTABLISHED" if self.player_health > 0 else "CONNECTION TERMINATED"
                text = self.font_lg.render(msg, True, self.COLOR_PLAYER if self.player_health > 0 else self.COLOR_OPPONENT)
                self.screen.blit(text, text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)))

    def _render_hud(self, who):
        if who == 'player':
            health_rect = pygame.Rect(20, self.SCREEN_HEIGHT - 40, 200, 20)
            momentum_rect = pygame.Rect(20, self.SCREEN_HEIGHT - 70, 150, 20)
            health = self.player_health
            max_health = self.MAX_HEALTH
            momentum = self.player_momentum
            color = self.COLOR_PLAYER
            proc_pos = (self.SCREEN_WIDTH - 150, self.SCREEN_HEIGHT - 50)
            proc_power = self.player_proc_power
        else:  # opponent
            health_rect = pygame.Rect(self.SCREEN_WIDTH - 220, 20, 200, 20)
            momentum_rect = pygame.Rect(self.SCREEN_WIDTH - 170, 50, 150, 20)
            health = self.opponent_health
            max_health = self.opponent_start_health
            momentum = self.opponent_momentum
            color = self.COLOR_OPPONENT
            proc_pos = (150, 50)
            proc_power = self.opponent_proc_power

        # Health Bar
        self._draw_bar(health_rect, health, max_health, color, self.COLOR_BG)
        health_text = self.font_md.render(f"SYS: {int(health)}/{int(max_health)}", True, self.COLOR_WHITE)
        self.screen.blit(health_text, (health_rect.x, health_rect.y - 25))

        # Momentum Bar
        self._draw_bar(momentum_rect, momentum, self.MAX_MOMENTUM, self.COLOR_MOMENTUM, self.COLOR_BG)
        momentum_text = self.font_md.render(f"MOM: {int(momentum)}/{self.MAX_MOMENTUM}", True, self.COLOR_WHITE)
        self.screen.blit(momentum_text, (momentum_rect.x, momentum_rect.y - 25))

        # Processing Power
        proc_text = self.font_md.render(f"PROC: x{proc_power}", True, self.COLOR_SYSTEM)
        self.screen.blit(proc_text, proc_text.get_rect(center=proc_pos))

    def _render_card(self, card, pos, is_selected):
        card_rect = pygame.Rect(pos[0] - self.CARD_WIDTH / 2, pos[1] - self.CARD_HEIGHT / 2, self.CARD_WIDTH, self.CARD_HEIGHT)

        # Glow effect for selection
        if is_selected:
            glow_color = self.COLOR_PLAYER if self.player_momentum >= card['cost'] else self.COLOR_GREY
            for i in range(5):
                glow_rect = card_rect.inflate(i * 4, i * 4)
                pygame.draw.rect(self.screen, glow_color, glow_rect, 1, border_radius=8)

        # Card body
        pygame.draw.rect(self.screen, self.COLOR_BG, card_rect, 0, border_radius=8)
        border_color = self.COLOR_PLAYER if self.player_momentum >= card['cost'] else self.COLOR_GREY
        pygame.draw.rect(self.screen, border_color, card_rect, 2, border_radius=8)

        # Content
        name_text = self.font_md.render(card['name'], True, self.COLOR_WHITE)
        self.screen.blit(name_text, name_text.get_rect(center=(pos[0], pos[1] - 35)))

        atk_text = self.font_md.render(f"ATK: {card['attack']}", True, self.COLOR_OPPONENT)
        self.screen.blit(atk_text, atk_text.get_rect(center=(pos[0], pos[1])))

        gain_text = self.font_md.render(f"GAIN: {card['gain']}", True, self.COLOR_MOMENTUM)
        self.screen.blit(gain_text, gain_text.get_rect(center=(pos[0], pos[1] + 20)))

        cost_text = self.font_lg.render(str(card['cost']), True, self.COLOR_MOMENTUM)
        self.screen.blit(cost_text, cost_text.get_rect(center=(pos[0], pos[1] + 45)))

    def _render_card_back(self, pos):
        card_rect = pygame.Rect(pos[0] - self.CARD_WIDTH / 2, pos[1] - self.CARD_HEIGHT / 2, self.CARD_WIDTH, self.CARD_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_BG, card_rect, 0, border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_OPPONENT, card_rect, 2, border_radius=8)
        pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 15, self.COLOR_OPPONENT)
        pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 15, self.COLOR_OPPONENT)

    def _render_upgrade_button(self):
        pos = self._get_target_selection_pos() if self.selected_index == len(self.player_hand) else (self.SCREEN_WIDTH - 60, self.SCREEN_HEIGHT - 100)
        
        rect = pygame.Rect(pos[0] - 40, pos[1] - 20, 80, 40)
        is_selected = self.selected_index == len(self.player_hand) and self.game_phase == 'PLAYER_TURN'
        can_afford = self.player_momentum >= self.UPGRADE_COST and self.player_proc_power < 5

        color = self.COLOR_SYSTEM if can_afford else self.COLOR_GREY

        if is_selected:
            for i in range(4):
                glow_rect = rect.inflate(i * 3, i * 3)
                pygame.draw.rect(self.screen, color, glow_rect, 1, border_radius=8)

        pygame.draw.rect(self.screen, self.COLOR_BG, rect, 0, border_radius=8)
        pygame.draw.rect(self.screen, color, rect, 2, border_radius=8)

        text = self.font_md.render(f"UPG [{self.UPGRADE_COST}]", True, color)
        self.screen.blit(text, text.get_rect(center=pos))

    def _draw_bar(self, rect, current, maximum, color, bg_color):
        if maximum <= 0: return
        pygame.draw.rect(self.screen, bg_color, rect)
        fill_ratio = max(0, min(1, current / maximum))
        fill_rect = pygame.Rect(rect.x, rect.y, int(rect.width * fill_ratio), rect.height)
        pygame.draw.rect(self.screen, color, fill_rect)
        pygame.draw.rect(self.screen, self.COLOR_WHITE, rect, 1)

    def _get_card_pos(self, i, num_cards, who):
        hand_width = num_cards * (self.CARD_WIDTH + 10) - 10
        start_x = (self.SCREEN_WIDTH - hand_width) / 2 + self.CARD_WIDTH / 2
        y = self.HAND_Y_PLAYER if who == 'player' else self.HAND_Y_OPPONENT
        return (start_x + i * (self.CARD_WIDTH + 10), y + self.CARD_HEIGHT / 2)

    def _get_target_selection_pos(self):
        # Card selection
        if len(self.player_hand) > 0 and self.selected_index < len(self.player_hand):
            pos = self._get_card_pos(self.selected_index, len(self.player_hand), 'player')
            return pos[0], pos[1] - 10
        # Upgrade button selection
        else:
            return self.SCREEN_WIDTH - 60, self.SCREEN_HEIGHT - 100

    def _update_and_draw_selection(self):
        target_pos = self._get_target_selection_pos()
        # Interpolate for smooth movement
        dx = target_pos[0] - self.selection_visual_pos[0]
        dy = target_pos[1] - self.selection_visual_pos[1]
        self.selection_visual_pos = (self.selection_visual_pos[0] + dx * 0.4, self.selection_visual_pos[1] + dy * 0.4)

        if self.game_phase != 'PLAYER_TURN': return

        # Draw the selection highlight
        is_card = self.selected_index < len(self.player_hand)
        if is_card:
            card = self.player_hand[self.selected_index]
            can_afford = self.player_momentum >= card['cost']
        else:
            can_afford = self.player_momentum >= self.UPGRADE_COST and self.player_proc_power < 5

        color = self.COLOR_PLAYER if can_afford else self.COLOR_GREY

        points = [
            (self.selection_visual_pos[0], self.selection_visual_pos[1] - 10),
            (self.selection_visual_pos[0] - 10, self.selection_visual_pos[1]),
            (self.selection_visual_pos[0] + 10, self.selection_visual_pos[1]),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _spawn_particles(self, count, pos, color, min_vel, max_vel, lifetime):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_vel, max_vel)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifetime, 'max_life': lifetime, 'color': color})

    def _spawn_beam(self, start_pos, end_pos, color, lifetime):
        self.particles.append({'pos': list(start_pos), 'end_pos': list(end_pos), 'life': lifetime, 'max_life': lifetime, 'color': color, 'type': 'beam'})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['life'] -= 1 / 30.0
            if p['life'] <= 0:
                self.particles.remove(p)
                continue

            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)

            if p.get('type') == 'beam':
                pygame.draw.line(self.screen, p['color'], p['pos'], p['end_pos'], int(p['life'] / p['max_life'] * 10))
            else:
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
                p['vel'][1] += 0.1  # Gravity
                size = int(5 * (p['life'] / p['max_life']))
                if size > 0:
                    pygame.draw.circle(self.screen, p['color'], p['pos'], size)

    def _spawn_floating_text(self, text, pos, color):
        self.floating_texts.append({'text': text, 'pos': list(pos), 'life': 1.0, 'color': color})

    def _update_and_draw_floating_texts(self):
        for ft in self.floating_texts[:]:
            ft['life'] -= 1 / 30.0
            if ft['life'] <= 0:
                self.floating_texts.remove(ft)
                continue

            ft['pos'][1] -= 1  # Move up
            alpha = int(255 * ft['life'])
            rendered_text = self.font_md.render(ft['text'], True, ft['color'])
            rendered_text.set_alpha(alpha)
            self.screen.blit(rendered_text, rendered_text.get_rect(center=ft['pos']))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It is not part of the required environment implementation
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use a visible driver
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Pygame setup for manual play
    pygame.display.set_caption("Cyber Duel")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    # Store button states
    movement = 0
    space_held = 0
    shift_held = 0
    
    # Track key presses for single-press actions
    keys_pressed_this_frame = set()
    keys_held = set()

    while running:
        keys_pressed_this_frame.clear()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Key DOWN events
            if event.type == pygame.KEYDOWN:
                keys_pressed_this_frame.add(event.key)
                keys_held.add(event.key)
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:  # Reset on 'R'
                    obs, info = env.reset()
                    total_reward = 0
            
            # Key UP events
            if event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held.remove(event.key)

        # Update actions based on key presses
        movement = 0
        if pygame.K_LEFT in keys_pressed_this_frame:
            movement = 3
        elif pygame.K_RIGHT in keys_pressed_this_frame:
            movement = 4

        space_held = 1 if pygame.K_SPACE in keys_held else 0
        shift_held = 1 if pygame.K_LSHIFT in keys_held or pygame.K_RSHIFT in keys_held else 0

        action = np.array([movement, space_held, shift_held])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        # The observation is (H, W, C), but pygame surface wants (W, H)
        # and surfarray.make_surface expects (W, H, C)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))

        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30)

    pygame.quit()