
# Generated: 2025-08-27T18:49:46.584963
# Source Brief: brief_01960.md
# Brief Index: 1960

        
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


class Card:
    """A class to represent a single card with its properties and state."""
    def __init__(self, shape, color, shape_id):
        self.shape = shape
        self.color = color
        self.shape_id = shape_id
        self.state = 'hidden'  # Can be 'hidden', 'revealed', or 'matched'
        self.reveal_anim_progress = 0.0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to reveal a card."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A memory match game. Find all 16 pairs of matching shapes before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 8, 4
    CARD_WIDTH, CARD_HEIGHT = 60, 80
    GRID_MARGIN_X = (SCREEN_WIDTH - GRID_COLS * CARD_WIDTH) / (GRID_COLS + 1)
    GRID_MARGIN_Y = (SCREEN_HEIGHT - 50 - GRID_ROWS * CARD_HEIGHT) / (GRID_ROWS + 1) # 50px for top UI
    GRID_TOP_OFFSET = 50

    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_CARD_BACK = (60, 80, 100)
    COLOR_CARD_BORDER = (90, 110, 130)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_MATCH_GLOW = (255, 255, 255, 100)
    
    # We need 16 unique card types. We use 8 shapes and "double" them with more colors.
    SHAPES = ['circle', 'square', 'triangle', 'diamond', 'star', 'hexagon', 'cross', 'pentagon'] * 2
    COLORS = [
        (255, 87, 34), (76, 175, 80), (33, 150, 243), (255, 235, 59),
        (156, 39, 176), (233, 30, 99), (0, 188, 212), (255, 152, 0),
        (121, 85, 72), (158, 158, 158), (96, 125, 139), (255, 61, 0),
        (0, 230, 118), (29, 233, 182), (103, 58, 183), (245, 0, 87)
    ]

    MAX_MOVES = 32
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_big = pygame.font.SysFont("Arial", 48, bold=True)
        
        # Initialize state variables
        self.grid = []
        self.cursor_pos = [0, 0]
        self.revealed_cards_indices = []
        self.steps = 0
        self.score = 0
        self.moves_left = 0
        self.matched_pairs = 0
        self.game_over = False
        self.last_match_pos = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.matched_pairs = 0
        self.cursor_pos = [0, 0]
        self.revealed_cards_indices = []
        self.last_match_pos = []

        # Create and shuffle cards
        card_prototypes = []
        for i in range(16):
            card_prototypes.append({'shape': self.SHAPES[i], 'color': self.COLORS[i], 'id': i})

        deck = card_prototypes * 2
        self.np_random.shuffle(deck)

        self.grid = []
        for i in range(self.GRID_ROWS):
            row = []
            for j in range(self.GRID_COLS):
                card_data = deck.pop()
                row.append(Card(card_data['shape'], card_data['color'], card_data['id']))
            self.grid.append(row)
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        
        reward = 0.0
        self.steps += 1
        self.last_match_pos = [] # Clear match glow effect from previous step

        # 1. Flip back mismatched cards from the previous turn
        if len(self.revealed_cards_indices) == 2:
            idx1, idx2 = self.revealed_cards_indices
            card1 = self.grid[idx1[1]][idx1[0]]
            card2 = self.grid[idx2[1]][idx2[0]]
            card1.state = 'hidden'
            card2.state = 'hidden'
            self.revealed_cards_indices = []

        # 2. Process player actions
        if movement != 0:
            if movement == 1: self.cursor_pos[1] -= 1 # Up
            elif movement == 2: self.cursor_pos[1] += 1 # Down
            elif movement == 3: self.cursor_pos[0] -= 1 # Left
            elif movement == 4: self.cursor_pos[0] += 1 # Right
            
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        if space_pressed:
            cx, cy = self.cursor_pos
            card = self.grid[cy][cx]

            if card.state == 'hidden' and len(self.revealed_cards_indices) < 2:
                # Sound: card_flip.wav
                card.state = 'revealed'
                card.reveal_anim_progress = 1.0
                self.moves_left -= 1
                self.revealed_cards_indices.append((cx, cy))

                # Calculate information-based reward
                partner_is_hidden = any(
                    other_card.shape_id == card.shape_id and other_card.state == 'hidden'
                    for r_idx, r in enumerate(self.grid) for c_idx, other_card in enumerate(r)
                    if not (r_idx == cy and c_idx == cx)
                )
                reward += 0.5 if partner_is_hidden else -0.1

        # 3. Check for new matches
        if len(self.revealed_cards_indices) == 2:
            idx1, idx2 = self.revealed_cards_indices
            card1 = self.grid[idx1[1]][idx1[0]]
            card2 = self.grid[idx2[1]][idx2[0]]

            if card1.shape_id == card2.shape_id:
                # Sound: match_success.wav
                card1.state = 'matched'
                card2.state = 'matched'
                self.matched_pairs += 1
                reward += 5.0
                self.last_match_pos = [idx1, idx2]
                self.revealed_cards_indices = []
            else:
                # Sound: mismatch_error.wav
                # Cards will be flipped back at the start of the next step
                pass
        
        self.score += reward

        # 4. Check for termination
        terminated = False
        if self.matched_pairs == 16:
            # Sound: game_win.wav
            reward += 50.0
            self.score += 50.0
            terminated = True
            self.game_over = True
        elif self.moves_left <= 0 and len(self.revealed_cards_indices) != 2:
            # Sound: game_lose.wav
            reward -= 50.0
            self.score -= 50.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "matched_pairs": self.matched_pairs,
        }

    def _render_game(self):
        # Draw cards
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                card = self.grid[r][c]
                card_x = self.GRID_MARGIN_X + c * (self.CARD_WIDTH + self.GRID_MARGIN_X)
                card_y = self.GRID_TOP_OFFSET + self.GRID_MARGIN_Y + r * (self.CARD_HEIGHT + self.GRID_MARGIN_Y)
                rect = pygame.Rect(card_x, card_y, self.CARD_WIDTH, self.CARD_HEIGHT)

                if card.state == 'hidden':
                    pygame.draw.rect(self.screen, self.COLOR_CARD_BACK, rect, border_radius=5)
                    pygame.draw.rect(self.screen, self.COLOR_CARD_BORDER, rect, width=2, border_radius=5)
                elif card.state == 'revealed':
                    self._draw_card_face(card, rect)

        # Draw match glow effect
        for pos in self.last_match_pos:
            c, r = pos
            card_x = self.GRID_MARGIN_X + c * (self.CARD_WIDTH + self.GRID_MARGIN_X)
            card_y = self.GRID_TOP_OFFSET + self.GRID_MARGIN_Y + r * (self.CARD_HEIGHT + self.GRID_MARGIN_Y)
            glow_rect = pygame.Rect(card_x - 5, card_y - 5, self.CARD_WIDTH + 10, self.CARD_HEIGHT + 10)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, self.COLOR_MATCH_GLOW, glow_surf.get_rect(), border_radius=10)
            self.screen.blit(glow_surf, glow_rect.topleft)

        # Draw cursor
        cursor_x = self.GRID_MARGIN_X + self.cursor_pos[0] * (self.CARD_WIDTH + self.GRID_MARGIN_X) - 4
        cursor_y = self.GRID_TOP_OFFSET + self.GRID_MARGIN_Y + self.cursor_pos[1] * (self.CARD_HEIGHT + self.GRID_MARGIN_Y) - 4
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.CARD_WIDTH + 8, self.CARD_HEIGHT + 8)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=4, border_radius=8)

    def _draw_card_face(self, card, rect):
        pygame.draw.rect(self.screen, card.color, rect, border_radius=5)
        shape_color = (255, 255, 255)
        center = rect.center
        radius = min(rect.width, rect.height) * 0.35
        
        if card.shape == 'circle':
            pygame.gfxdraw.aacircle(self.screen, int(center[0]), int(center[1]), int(radius), shape_color)
            pygame.gfxdraw.filled_circle(self.screen, int(center[0]), int(center[1]), int(radius), shape_color)
        elif card.shape == 'square':
            size = radius * 1.7
            pygame.draw.rect(self.screen, shape_color, (center[0] - size/2, center[1] - size/2, size, size), border_radius=2)
        elif card.shape == 'triangle':
            points = [(center[0], center[1] - radius), (center[0] - radius, center[1] + radius * 0.7), (center[0] + radius, center[1] + radius * 0.7)]
            pygame.gfxdraw.aapolygon(self.screen, points, shape_color)
            pygame.gfxdraw.filled_polygon(self.screen, points, shape_color)
        elif card.shape == 'diamond':
            points = [(center[0], center[1] - radius * 1.2), (center[0] + radius, center[1]), (center[0], center[1] + radius * 1.2), (center[0] - radius, center[1])]
            pygame.gfxdraw.aapolygon(self.screen, points, shape_color)
            pygame.gfxdraw.filled_polygon(self.screen, points, shape_color)
        elif card.shape == 'star':
            self._draw_star(center, 5, radius, radius*0.4, shape_color)
        elif card.shape == 'hexagon':
            self._draw_polygon(center, 6, radius, shape_color, rotation=math.pi/2)
        elif card.shape == 'cross':
            w, h = radius * 0.5, radius * 1.8
            pygame.draw.rect(self.screen, shape_color, (center[0]-w/2, center[1]-h/2, w, h), border_radius=2)
            pygame.draw.rect(self.screen, shape_color, (center[0]-h/2, center[1]-w/2, h, w), border_radius=2)
        elif card.shape == 'pentagon':
            self._draw_polygon(center, 5, radius, shape_color)

    def _draw_polygon(self, center, n_sides, radius, color, rotation=0):
        points = []
        for i in range(n_sides):
            angle = 2 * math.pi * i / n_sides - math.pi / 2 + rotation
            points.append((center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)))
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_star(self, center, n_points, outer_radius, inner_radius, color):
        points = []
        for i in range(2 * n_points):
            angle = math.pi * i / n_points - math.pi / 2
            radius = outer_radius if i % 2 == 0 else inner_radius
            points.append((center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)))
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_ui(self):
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, (255, 255, 255))
        self.screen.blit(moves_text, (20, 15))

        pairs_text = self.font_ui.render(f"Pairs: {self.matched_pairs} / 16", True, (255, 255, 255))
        self.screen.blit(pairs_text, (self.SCREEN_WIDTH - pairs_text.get_width() - 20, 15))

        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, (255, 255, 255))
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 15))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.matched_pairs == 16 else "GAME OVER"
            end_color = self.COLOR_CURSOR if self.matched_pairs == 16 else (200, 50, 50)
            end_text = self.font_big.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

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
    env = GameEnv()
    obs, info = env.reset()
    game_running = True
    
    pygame.display.set_caption("Memory Match")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0.0
    episode_over = False

    while game_running:
        current_action = [0, 0, 0]
        action_taken = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_running = False
            if event.type == pygame.KEYDOWN:
                if episode_over:
                    obs, info = env.reset()
                    total_reward = 0
                    episode_over = False
                    print("\n--- NEW GAME ---")
                    continue

                action_taken = True
                if event.key == pygame.K_UP: current_action[0] = 1
                elif event.key == pygame.K_DOWN: current_action[0] = 2
                elif event.key == pygame.K_LEFT: current_action[0] = 3
                elif event.key == pygame.K_RIGHT: current_action[0] = 4
                elif event.key == pygame.K_SPACE: current_action[1] = 1
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    action_taken = False
                    print("\n--- GAME RESET ---")
        
        if action_taken:
            obs, reward, terminated, truncated, info = env.step(current_action)
            total_reward += reward
            print(f"Action: {current_action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Info: {info}")
            if terminated:
                print("--- EPISODE FINISHED --- (Press any key to restart)")
                episode_over = True
        
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    env.close()