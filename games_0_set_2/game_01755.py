
# Generated: 2025-08-28T02:37:15.530654
# Source Brief: brief_01755.md
# Brief Index: 1755

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (mapped to actions 1-4) to move on the grid. Each move costs 1 turn."
    )

    game_description = (
        "A strategic puzzle game. Collect 10 crystals before you run out of moves!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE_X, self.GRID_SIZE_Y = 10, 10
        self.TILE_WIDTH, self.TILE_HEIGHT = 60, 30
        self.TILE_W_HALF, self.TILE_H_HALF = self.TILE_WIDTH // 2, self.TILE_HEIGHT // 2
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 100

        # Game parameters
        self.MAX_MOVES = 20
        self.CRYSTALS_TO_WIN = 10
        self.NUM_CRYSTALS_ON_MAP = 15

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID_LINE = (40, 55, 71)
        self.COLOR_GRID_FILL = (33, 47, 60)
        self.COLOR_PLAYER = (236, 240, 241)
        self.COLOR_PLAYER_OUTLINE = (189, 195, 199)
        self.COLOR_SHADOW = (0, 0, 0, 50)
        self.CRYSTAL_COLORS = [
            ((46, 204, 113), (39, 174, 96)),  # Emerald
            ((52, 152, 219), (41, 128, 185)),  # Sapphire
            ((231, 76, 60), (192, 57, 43)),    # Ruby
            ((241, 196, 15), (243, 156, 18)),  # Gold
            ((155, 89, 182), (142, 68, 173)),  # Amethyst
        ]
        self.COLOR_UI_TEXT = (236, 240, 241)
        self.COLOR_WIN = (46, 204, 113)
        self.COLOR_LOSE = (231, 76, 60)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # State variables are initialized in reset()
        self.player_pos = [0, 0]
        self.crystals = []
        self.moves_left = 0
        self.crystals_collected = 0
        self.game_over = False
        self.win = False
        self.last_collection_effect = None
        self.np_random = None

    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_W_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_H_HALF
        return int(screen_x), int(screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.moves_left = self.MAX_MOVES
        self.crystals_collected = 0
        self.game_over = False
        self.win = False
        self.last_collection_effect = None

        self.player_pos = [self.GRID_SIZE_X // 2, self.GRID_SIZE_Y // 2]
        
        possible_coords = [
            (x, y) for x in range(self.GRID_SIZE_X) for y in range(self.GRID_SIZE_Y)
        ]
        possible_coords.remove(tuple(self.player_pos))
        
        crystal_indices = self.np_random.choice(
            len(possible_coords), self.NUM_CRYSTALS_ON_MAP, replace=False
        )
        
        self.crystals = []
        for i, idx in enumerate(crystal_indices):
            self.crystals.append({
                "pos": list(possible_coords[idx]),
                "color_idx": i % len(self.CRYSTAL_COLORS)
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.last_collection_effect = None

        movement = action[0]
        reward = 0

        self.moves_left -= 1
        
        px, py = self.player_pos
        nx, ny = px, py

        if movement == 1: ny -= 1
        elif movement == 2: ny += 1
        elif movement == 3: nx -= 1
        elif movement == 4: nx += 1

        if 0 <= nx < self.GRID_SIZE_X and 0 <= ny < self.GRID_SIZE_Y:
            self.player_pos = [nx, ny]
        
        crystal_to_remove = None
        for crystal in self.crystals:
            if crystal["pos"] == self.player_pos:
                crystal_to_remove = crystal
                break
        
        if crystal_to_remove:
            # SFX: Crystal collect sound
            self.crystals.remove(crystal_to_remove)
            self.crystals_collected += 1
            reward = 1
            self.last_collection_effect = {
                "pos": self.player_pos,
                "color": self.CRYSTAL_COLORS[crystal_to_remove["color_idx"]][0]
            }

        terminated = False
        if self.crystals_collected >= self.CRYSTALS_TO_WIN:
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100
        elif self.moves_left <= 0:
            self.game_over = True
            terminated = True
            if not self.win:
                reward -= 10

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        shadow_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)

        for y in range(self.GRID_SIZE_Y):
            for x in range(self.GRID_SIZE_X):
                tile_points = [
                    self._iso_to_screen(x, y), self._iso_to_screen(x + 1, y),
                    self._iso_to_screen(x + 1, y + 1), self._iso_to_screen(x, y + 1),
                ]
                pygame.gfxdraw.filled_polygon(self.screen, tile_points, self.COLOR_GRID_FILL)
                pygame.gfxdraw.aapolygon(self.screen, tile_points, self.COLOR_GRID_LINE)

                for crystal in self.crystals:
                    if crystal["pos"] == [x, y]:
                        self._draw_crystal(self.screen, shadow_surface, x, y, crystal["color_idx"])
                        break
                
                if self.player_pos == [x, y]:
                    self._draw_player(self.screen, shadow_surface, x, y)
        
        self.screen.blit(shadow_surface, (0, 0))

        if self.last_collection_effect:
            self._draw_burst_effect(self.screen, self.last_collection_effect['pos'], self.last_collection_effect['color'])

    def _draw_player(self, surface, shadow_surface, x, y):
        center_x, center_y = self._iso_to_screen(x + 0.5, y + 0.5)
        
        shadow_rect = pygame.Rect(0, 0, self.TILE_W_HALF * 0.8, self.TILE_H_HALF * 0.8)
        shadow_rect.center = (center_x, center_y + self.TILE_H_HALF * 0.6)
        pygame.draw.ellipse(shadow_surface, self.COLOR_SHADOW, shadow_rect)

        body_points = [
            (center_x, center_y - self.TILE_H_HALF * 0.8),
            (center_x + self.TILE_W_HALF * 0.4, center_y),
            (center_x, center_y + self.TILE_H_HALF * 0.8),
            (center_x - self.TILE_W_HALF * 0.4, center_y),
        ]
        pygame.gfxdraw.filled_polygon(surface, body_points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(surface, body_points, self.COLOR_PLAYER_OUTLINE)

    def _draw_crystal(self, surface, shadow_surface, x, y, color_idx):
        center_x, center_y = self._iso_to_screen(x + 0.5, y + 0.5)
        
        shadow_rect = pygame.Rect(0, 0, self.TILE_W_HALF * 0.7, self.TILE_H_HALF * 0.7)
        shadow_rect.center = (center_x, center_y + self.TILE_H_HALF * 0.4)
        pygame.draw.ellipse(shadow_surface, self.COLOR_SHADOW, shadow_rect)

        main_color, outline_color = self.CRYSTAL_COLORS[color_idx]
        h, w = self.TILE_H_HALF * 0.7, self.TILE_W_HALF * 0.35
        top_point, mid_left, mid_right, bottom_point = (center_x, center_y - h), (center_x - w, center_y), (center_x + w, center_y), (center_x, center_y + h)
        
        pygame.gfxdraw.filled_polygon(surface, [top_point, mid_left, (center_x, center_y)], main_color)
        pygame.gfxdraw.filled_polygon(surface, [top_point, mid_right, (center_x, center_y)], tuple(min(255, c+20) for c in main_color))
        pygame.gfxdraw.filled_polygon(surface, [bottom_point, mid_left, (center_x, center_y)], tuple(max(0, c-20) for c in main_color))
        pygame.gfxdraw.filled_polygon(surface, [bottom_point, mid_right, (center_x, center_y)], main_color)
        pygame.gfxdraw.aapolygon(surface, [top_point, mid_right, bottom_point, mid_left], outline_color)

    def _draw_burst_effect(self, surface, pos, color):
        center_x, center_y = self._iso_to_screen(pos[0] + 0.5, pos[1] + 0.5)
        for i in range(8):
            angle = i * (2 * math.pi / 8)
            start_point = (center_x + 15 * math.cos(angle), center_y + 15 * math.sin(angle))
            end_point = (center_x + 25 * math.cos(angle), center_y + 25 * math.sin(angle))
            pygame.draw.aaline(surface, color, start_point, end_point, 2)

    def _render_ui(self):
        moves_text = f"Moves: {self.moves_left}"
        text_surf = self.font_small.render(moves_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))

        crystals_text = f"Crystals: {self.crystals_collected} / {self.CRYSTALS_TO_WIN}"
        text_surf = self.font_small.render(crystals_text, True, self.COLOR_UI_TEXT)
        text_rect = text_surf.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(text_surf, text_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg, color = ("YOU WIN!", self.COLOR_WIN) if self.win else ("GAME OVER", self.COLOR_LOSE)
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {"score": self.crystals_collected, "steps": self.MAX_MOVES - self.moves_left}

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {obs.shape}, expected {(self.HEIGHT, self.WIDTH, 3)}"
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    env.validate_implementation()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Crystal Collector")
    
    obs, info = env.reset()
    done = False
    
    print("\n--- Manual Game Start ---")
    print(env.user_guide)
    
    running = True
    while running:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                moved = False
                if event.key == pygame.K_UP: action[0], moved = 1, True
                elif event.key == pygame.K_DOWN: action[0], moved = 2, True
                elif event.key == pygame.K_LEFT: action[0], moved = 3, True
                elif event.key == pygame.K_RIGHT: action[0], moved = 4, True
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    print("--- Game Reset ---")
                
                if moved:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action[0]}, Reward: {reward}, Info: {info}, Terminated: {terminated}")
                    if terminated:
                        print("--- Episode End --- (Press R to reset)")
        
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
            
    env.close()