# Generated: 2025-08-27T15:39:14.917238
# Source Brief: brief_01036.md
# Brief Index: 1036

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your character on the grid. "
        "Avoid the red enemies and collect 20 flashing gems to win!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game. Navigate a grid to collect gems while dodging patrolling enemies. "
        "Lose a life if you're caught, but win by gathering all the gems!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 32
    GRID_HEIGHT = 20
    CELL_WIDTH = SCREEN_WIDTH // GRID_WIDTH
    CELL_HEIGHT = SCREEN_HEIGHT // GRID_HEIGHT
    
    MAX_STEPS = 1000
    GEM_TARGET = 20
    STARTING_LIVES = 3
    NUM_GEMS = 5
    GEM_RESPAWN_DELAY = 5
    NUM_ENEMIES = 4

    # --- Colors ---
    COLOR_BG = (15, 25, 40)
    COLOR_GRID = (30, 45, 60)
    COLOR_PLAYER = (255, 220, 0)
    COLOR_PLAYER_GLOW = (255, 220, 0, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BG = (25, 40, 55, 180)
    COLOR_GEM_BASE = (0, 255, 255) # Hue will be varied

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Game state variables are initialized in reset()
        self.player_pos = None
        self.enemies = []
        self.gems = []
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.gems_collected = 0
        self.enemy_speed = 0.0
        self.game_over = False
        self.win_message = ""

        # This call will fail if the implementation is wrong
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.lives = self.STARTING_LIVES
        self.gems_collected = 0
        self.game_over = False
        self.win_message = ""
        self.enemy_speed = 0.05 # 1 move every 20 steps

        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self._init_enemies()
        self._init_gems()

        return self._get_observation(), self._get_info()

    def _init_enemies(self):
        self.enemies = []
        patrol_starts = [
            [3, 3], [self.GRID_WIDTH - 4, 5], 
            [5, self.GRID_HEIGHT - 4], [self.GRID_WIDTH - 6, self.GRID_HEIGHT - 6]
        ]
        
        # Enemy 1: Horizontal Patrol
        self.enemies.append({
            "pos": list(patrol_starts[0]), "start": list(patrol_starts[0]), "end": [self.GRID_WIDTH - 4, 3],
            "dir": 1, "timer": 0.0, "type": "horizontal"
        })
        # Enemy 2: Vertical Patrol
        self.enemies.append({
            "pos": list(patrol_starts[1]), "start": list(patrol_starts[1]), "end": [self.GRID_WIDTH - 4, self.GRID_HEIGHT - 4],
            "dir": 1, "timer": 0.0, "type": "vertical"
        })
        # Enemy 3: Box Patrol
        self.enemies.append({
            "pos": list(patrol_starts[2]), "path": [
                (5, self.GRID_HEIGHT - 4), (12, self.GRID_HEIGHT - 4), (12, self.GRID_HEIGHT - 8), (5, self.GRID_HEIGHT - 8)
            ], "path_idx": 0, "timer": 0.0, "type": "box"
        })
        # Enemy 4: Diagonal Patrol
        self.enemies.append({
            "pos": list(patrol_starts[3]), "start": list(patrol_starts[3]), "end": [self.GRID_WIDTH - 12, self.GRID_HEIGHT - 12],
            "dir": 1, "timer": 0.0, "type": "diagonal"
        })


    def _init_gems(self):
        self.gems = []
        for i in range(self.NUM_GEMS):
            pos = self._get_random_empty_cell()
            self.gems.append({"pos": pos, "respawn_timer": 0, "hue": (i * 360 / self.NUM_GEMS)})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement = action[0]
        
        # 1. Update Player Position
        if movement == 0: # no-op
            reward -= 0.1
        else:
            reward -= 0.05
            if movement == 1: # Up
                self.player_pos[1] -= 1
            elif movement == 2: # Down
                self.player_pos[1] += 1
            elif movement == 3: # Left
                self.player_pos[0] -= 1
            elif movement == 4: # Right
                self.player_pos[0] += 1
        
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_WIDTH - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_HEIGHT - 1)

        # 2. Update Enemy Positions
        self._update_enemies()
        
        # 3. Handle Gem Respawns
        self._update_gems()

        # 4. Handle Interactions
        # Gem Collection
        for gem in self.gems:
            if gem["respawn_timer"] == 0 and self.player_pos == gem["pos"]:
                reward += 1.0
                self.score += 10
                self.gems_collected += 1
                gem["respawn_timer"] = self.GEM_RESPAWN_DELAY
                # Sound: gem_collect.wav

        # Enemy Collision
        player_hit = False
        for enemy in self.enemies:
            if self.player_pos == enemy["pos"]:
                player_hit = True
                break
        
        if player_hit:
            self.lives -= 1
            reward -= 10.0
            self.score -= 50
            # Sound: player_hit.wav
            if self.lives > 0:
                self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2] # Reset to center

        # 5. Update Difficulty
        if self.steps > 0 and self.steps % 200 == 0:
            self.enemy_speed = min(0.25, self.enemy_speed + 0.05)

        # 6. Check for Termination
        terminated = False
        if self.gems_collected >= self.GEM_TARGET:
            reward += 100
            self.score += 1000
            terminated = True
            self.game_over = True
            self.win_message = "YOU WIN!"
            # Sound: win_game.wav
        elif self.lives <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
            self.win_message = "GAME OVER"
            # Sound: lose_game.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win_message = "TIME UP"

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _update_enemies(self):
        for enemy in self.enemies:
            enemy["timer"] += self.enemy_speed
            if enemy["timer"] < 1.0:
                continue
            enemy["timer"] -= 1.0

            if enemy["type"] == "horizontal":
                enemy["pos"][0] += enemy["dir"]
                if enemy["pos"][0] >= enemy["end"][0] or enemy["pos"][0] <= enemy["start"][0]:
                    enemy["dir"] *= -1
            elif enemy["type"] == "vertical":
                enemy["pos"][1] += enemy["dir"]
                if enemy["pos"][1] >= enemy["end"][1] or enemy["pos"][1] <= enemy["start"][1]:
                    enemy["dir"] *= -1
            elif enemy["type"] == "box":
                target_pos = enemy["path"][enemy["path_idx"]]
                dx = np.sign(target_pos[0] - enemy["pos"][0])
                dy = np.sign(target_pos[1] - enemy["pos"][1])
                enemy["pos"][0] += dx
                enemy["pos"][1] += dy
                if tuple(enemy["pos"]) == target_pos:
                    enemy["path_idx"] = (enemy["path_idx"] + 1) % len(enemy["path"])
            elif enemy["type"] == "diagonal":
                enemy["pos"][0] += enemy["dir"]
                enemy["pos"][1] += enemy["dir"]
                if enemy["pos"][0] >= enemy["start"][0] or enemy["pos"][0] <= enemy["end"][0]:
                     enemy["dir"] *= -1

    def _update_gems(self):
        for gem in self.gems:
            if gem["respawn_timer"] > 0:
                gem["respawn_timer"] -= 1
                if gem["respawn_timer"] == 0:
                    gem["pos"] = self._get_random_empty_cell()

    def _get_random_empty_cell(self):
        occupied_cells = [tuple(self.player_pos)]
        occupied_cells.extend([tuple(e["pos"]) for e in self.enemies])
        occupied_cells.extend([tuple(g["pos"]) for g in self.gems if g["respawn_timer"] == 0])
        
        while True:
            pos = [self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)]
            if tuple(pos) not in occupied_cells:
                return pos

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_gems()
        self._render_enemies()
        self._render_player()
        self._render_ui()

        if self.game_over:
            self._render_game_over_message()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "gems_collected": self.gems_collected
        }

    def _world_to_screen(self, grid_pos):
        return (
            grid_pos[0] * self.CELL_WIDTH,
            grid_pos[1] * self.CELL_HEIGHT,
        )

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
    
    def _render_player(self):
        px, py = self._world_to_screen(self.player_pos)
        center_x, center_y = px + self.CELL_WIDTH // 2, py + self.CELL_HEIGHT // 2
        
        # Glow effect
        glow_radius = int(self.CELL_WIDTH * 0.8)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, glow_radius, self.COLOR_PLAYER_GLOW)
        
        # Player square
        player_rect = pygame.Rect(px, py, self.CELL_WIDTH, self.CELL_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-4, -4))

    def _render_enemies(self):
        for enemy in self.enemies:
            px, py = self._world_to_screen(enemy["pos"])
            center_x, center_y = px + self.CELL_WIDTH // 2, py + self.CELL_HEIGHT // 2
            
            # Bobbing animation
            bob = math.sin(self.steps * 0.3 + enemy["pos"][0]) * 2
            
            p1 = (center_x, py + 2 + bob)
            p2 = (px + 2, py + self.CELL_HEIGHT - 2 + bob)
            p3 = (px + self.CELL_WIDTH - 2, py + self.CELL_HEIGHT - 2 + bob)
            
            pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_ENEMY)
            pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_ENEMY)

    def _render_gems(self):
        for gem in self.gems:
            if gem["respawn_timer"] > 0:
                continue

            px, py = self._world_to_screen(gem["pos"])
            
            # Pulsing size and flashing color
            pulse = (math.sin(self.steps * 0.4 + gem["hue"]) + 1) / 2 # 0 to 1
            size_inflation = int(-8 + pulse * 6)
            
            gem_rect = pygame.Rect(px, py, self.CELL_WIDTH, self.CELL_HEIGHT).inflate(size_inflation, size_inflation)

            color = pygame.Color(0,0,0)
            brightness = 75 + pulse * 25 # 75 to 100
            color.hsva = (gem["hue"], 90, brightness, 100)

            pygame.draw.rect(self.screen, color, gem_rect)
            pygame.draw.rect(self.screen, tuple(min(255, c*1.2) for c in color[:3]), gem_rect, 1) # Border

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Lives
        for i in range(self.lives):
            self._draw_heart(self.SCREEN_WIDTH - 25 - i * 20, 15, self.COLOR_ENEMY)
        
        # Gems Collected
        gem_text = self.font_small.render(f"{self.gems_collected} / {self.GEM_TARGET}", True, self.COLOR_TEXT)
        text_rect = gem_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 15))
        self.screen.blit(gem_text, text_rect)
        
        # Gem icon
        gem_icon_rect = pygame.Rect(0, 0, self.CELL_WIDTH-4, self.CELL_HEIGHT-4)
        gem_icon_rect.center = (text_rect.left - 20, text_rect.centery)
        pygame.draw.rect(self.screen, self.COLOR_GEM_BASE, gem_icon_rect)

    def _draw_heart(self, x, y, color):
        points = [
            (x, y - 5), (x + 5, y - 10), (x + 10, y - 5),
            (x, y + 5),
            (x - 10, y - 5), (x - 5, y - 10)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _render_game_over_message(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0,0,0,180))
        self.screen.blit(s, (0,0))
        
        text = self.font_large.render(self.win_message, True, self.COLOR_TEXT)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text, text_rect)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset and observation space. reset() must be called to initialize state.
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # To play, you must have pygame installed and not be in a headless environment
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Cannot run in headless mode. Unset SDL_VIDEODRIVER to play.")
    else:
        env = GameEnv()
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Gem Collector")
        clock = pygame.time.Clock()

        running = True
        total_reward = 0
        
        print("\n" + "="*30)
        print(env.game_description)
        print(env.user_guide)
        print("="*30 + "\n")

        while running:
            movement, space, shift = 0, 0, 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]
            
            # Since auto_advance is False, we only step when there's an action (or no-op)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Render the observation from the environment to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
                pygame.time.wait(2000) # Pause for 2 seconds
                obs, info = env.reset()
                total_reward = 0

            # We are not using a fixed FPS, but this prevents the loop from running too fast
            clock.tick(10)

        env.close()