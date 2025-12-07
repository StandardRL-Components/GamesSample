
# Generated: 2025-08-28T05:39:51.516366
# Source Brief: brief_05654.md
# Brief Index: 5654

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your character (yellow circle) around the grid. "
        "Collect the glowing crystals to score points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect shimmering crystals in an isometric world while dodging patrolling enemies. "
        "Collect 20 crystals to win, but be careful - touching an enemy ends the game!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 25
    GRID_HEIGHT = 25
    TILE_WIDTH_HALF = 12
    TILE_HEIGHT_HALF = 6
    MAX_STEPS = 1000
    CRYSTAL_TARGET = 20
    NUM_ENEMIES = 5

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (50, 55, 65)
    COLOR_PLAYER = (255, 220, 0)
    COLOR_PLAYER_OUTLINE = (255, 255, 255)
    COLOR_ENEMY = (138, 43, 226)
    COLOR_UI_TEXT = (240, 240, 240)
    CRYSTAL_COLORS = [
        (0, 255, 255),  # Cyan
        (124, 252, 0),  # Lawn Green
        (255, 20, 147), # Deep Pink
    ]

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)

        self.grid_center_x = self.SCREEN_WIDTH // 2
        self.grid_center_y = self.SCREEN_HEIGHT // 2 - 50

        self.enemy_paths = self._define_enemy_paths()
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.enemies = []
        self.crystals = []
        self.collected_crystals = 0
        self.np_random = None

        self.validate_implementation()

    def _define_enemy_paths(self):
        paths = []
        # Path 1: Horizontal line
        paths.append([[x, 5] for x in range(2, 23)] + [[x, 5] for x in range(22, 1, -1)])
        # Path 2: Vertical line
        paths.append([[20, y] for y in range(2, 23)] + [[20, y] for y in range(22, 1, -1)])
        # Path 3: Small square loop
        paths.append([[5, 15], [6, 15], [7, 15], [7, 16], [7, 17], [6, 17], [5, 17], [5, 16]])
        # Path 4: Large square loop
        paths.append([[x, 2] for x in range(2, 23)] + [[22, y] for y in range(3, 23)] + 
                     [[x, 22] for x in range(21, 1, -1)] + [[2, y] for y in range(21, 2, -1)])
        # Path 5: Figure-eight
        path5 = []
        for i in range(16): path5.append([10 + int(3 * math.cos(i * math.pi / 8)), 10 + int(3 * math.sin(i * math.pi / 4))])
        paths.append(path5)
        return paths

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.collected_crystals = 0

        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        # Generate crystals
        self.crystals = []
        possible_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        possible_coords.remove(tuple(self.player_pos))
        self.np_random.shuffle(possible_coords)
        for i in range(self.CRYSTAL_TARGET):
            self.crystals.append(list(possible_coords[i]))

        # Initialize enemies
        self.enemies = []
        for i in range(self.NUM_ENEMIES):
            path = self.enemy_paths[i % len(self.enemy_paths)]
            start_index = self.np_random.integers(0, len(path))
            self.enemies.append({
                "pos": list(path[start_index]),
                "path": path,
                "path_idx": start_index
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), self.score, True, False, self._get_info()
            
        movement = action[0]
        reward = 0.0

        # --- Reward for moving towards/away from nearest crystal ---
        if self.crystals:
            old_dist, _ = self._find_nearest_crystal(self.player_pos)
            
            potential_pos = self.player_pos.copy()
            self._move_player(movement, potential_pos)

            new_dist, _ = self._find_nearest_crystal(potential_pos)

            if new_dist < old_dist:
                reward += 0.1
            elif new_dist > old_dist:
                reward -= 0.1
        
        # --- Update Player State ---
        self._move_player(movement, self.player_pos)

        # --- Update Enemy State ---
        for enemy in self.enemies:
            enemy["path_idx"] = (enemy["path_idx"] + 1) % len(enemy["path"])
            enemy["pos"] = list(enemy["path"][enemy["path_idx"]])
            # # Sound placeholder:
            # if self.steps % 15 == 0: # play sound intermittently
            #     # play_sound('enemy_move.wav')

        # --- Check for Interactions & Update Rewards ---
        # Crystal Collection
        crystal_to_remove = None
        for crystal in self.crystals:
            if self.player_pos == crystal:
                crystal_to_remove = crystal
                break
        
        if crystal_to_remove:
            self.crystals.remove(crystal_to_remove)
            self.collected_crystals += 1
            self.score += 1
            reward += 1
            # # Sound placeholder: play_sound('crystal_collect.wav')

            # Bonus for collecting near an enemy
            if self._is_enemy_adjacent(self.player_pos):
                self.score += 2
                reward += 2

        # Enemy Collision
        for enemy in self.enemies:
            if self.player_pos == enemy["pos"]:
                self.game_over = True
                self.score -= 10
                reward -= 10
                # # Sound placeholder: play_sound('player_hit.wav')
                break

        # --- Update Game State & Check Termination ---
        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS

        if self.collected_crystals >= self.CRYSTAL_TARGET:
            terminated = True
            self.score += 100
            reward += 100
            # # Sound placeholder: play_sound('win_game.wav')

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_player(self, movement, pos):
        # 1=up (iso up-left), 2=down (iso down-right), 3=left (iso down-left), 4=right (iso up-right)
        if movement == 1: # Up (iso up-left)
            pos[0] -= 1
        elif movement == 2: # Down (iso down-right)
            pos[0] += 1
        elif movement == 3: # Left (iso down-left)
            pos[1] -= 1
        elif movement == 4: # Right (iso up-right)
            pos[1] += 1
        
        pos[0] = max(0, min(self.GRID_WIDTH - 1, pos[0]))
        pos[1] = max(0, min(self.GRID_HEIGHT - 1, pos[1]))

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.grid_center_x + (grid_x - grid_y) * self.TILE_WIDTH_HALF
        screen_y = self.grid_center_y + (grid_x + grid_y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _find_nearest_crystal(self, pos):
        if not self.crystals:
            return float('inf'), None
        
        min_dist = float('inf')
        nearest_crystal = None
        for crystal in self.crystals:
            dist = abs(pos[0] - crystal[0]) + abs(pos[1] - crystal[1])
            if dist < min_dist:
                min_dist = dist
                nearest_crystal = crystal
        return min_dist, nearest_crystal

    def _is_enemy_adjacent(self, pos):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                check_pos = [pos[0] + dx, pos[1] + dy]
                for enemy in self.enemies:
                    if enemy["pos"] == check_pos:
                        return True
        return False

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
            "crystals_collected": self.collected_crystals,
        }

    def _render_game(self):
        # Render grid
        for i in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(i, 0)
            end = self._iso_to_screen(i, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for i in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, i)
            end = self._iso_to_screen(self.GRID_WIDTH, i)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Render crystals
        for i, crystal_pos in enumerate(self.crystals):
            x, y = self._iso_to_screen(crystal_pos[0], crystal_pos[1])
            color = self.CRYSTAL_COLORS[i % len(self.CRYSTAL_COLORS)]
            
            # Glow effect
            glow_radius = self.TILE_WIDTH_HALF * 1.5
            glow_alpha = 60 + 30 * math.sin(self.steps * 0.2 + i)
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, glow_alpha), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (x - glow_radius, y - glow_radius))

            # Crystal shape (diamond)
            points = [
                (x, y - self.TILE_HEIGHT_HALF),
                (x + self.TILE_WIDTH_HALF * 0.8, y),
                (x, y + self.TILE_HEIGHT_HALF),
                (x - self.TILE_WIDTH_HALF * 0.8, y)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Render enemies
        for enemy in self.enemies:
            x, y = self._iso_to_screen(enemy["pos"][0], enemy["pos"][1])
            points = [
                (x, y - self.TILE_HEIGHT_HALF),
                (x + self.TILE_WIDTH_HALF * 0.7, y + self.TILE_HEIGHT_HALF * 0.7),
                (x - self.TILE_WIDTH_HALF * 0.7, y + self.TILE_HEIGHT_HALF * 0.7),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

        # Render player
        player_x, player_y = self._iso_to_screen(self.player_pos[0], self.player_pos[1])
        radius = self.TILE_WIDTH_HALF * 0.7
        # Outline
        pygame.gfxdraw.aacircle(self.screen, player_x, player_y, int(radius + 1), self.COLOR_PLAYER_OUTLINE)
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, int(radius + 1), self.COLOR_PLAYER_OUTLINE)
        # Main body
        pygame.gfxdraw.aacircle(self.screen, player_x, player_y, int(radius), self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, int(radius), self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        crystal_text = self.font_ui.render(f"CRYSTALS: {self.collected_crystals} / {self.CRYSTAL_TARGET}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crystal_text, (10, 35))

        if self.game_over:
            status_text_str = "GAME OVER"
            if self.collected_crystals >= self.CRYSTAL_TARGET:
                status_text_str = "YOU WIN!"

            status_font = pygame.font.SysFont("Consolas", 50, bold=True)
            status_text = status_font.render(status_text_str, True, self.COLOR_UI_TEXT)
            text_rect = status_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(status_text, text_rect)


    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It's a good way to test the environment
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # Since auto_advance is False, we only step on action
        action = [0, 0, 0] # Default action is no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    done = False
                
                if not done:
                    # Map keys to actions
                    if event.key == pygame.K_UP:
                        action[0] = 1 # up
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2 # down
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3 # left
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4 # right
                    
                    # Step the environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Done: {done}")

        # Pygame screen for human viewing
        # Note: The environment's observation is separate from this display
        render_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # If you have a display, you can show the game
        try:
            screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
            pygame.display.set_caption("Crystal Collector")
            screen.blit(render_surface, (0, 0))
            pygame.display.flip()
        except pygame.error:
            # Running in a headless environment
            pass

    env.close()