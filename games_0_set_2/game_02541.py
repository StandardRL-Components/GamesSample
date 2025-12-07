
# Generated: 2025-08-28T05:10:44.074083
# Source Brief: brief_02541.md
# Brief Index: 2541

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move on the isometric grid. "
        "Collect gems, avoid the triangles!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect shimmering gems while dodging cunning enemies in a vibrant, "
        "isometric arcade world. The closer an enemy is to a gem, the more points it's worth!"
    )

    # Frames auto-advance at 30fps.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Game parameters
    MAX_STEPS = 1500  # 50 seconds at 30 FPS
    WIN_GEMS = 50
    MAX_HITS = 3
    NUM_GEMS = 10
    NUM_ENEMIES = 4
    
    # Grid and Isometric Projection
    GRID_W = 18
    GRID_H = 18
    TILE_WIDTH = 48
    TILE_HEIGHT = 24
    TILE_WIDTH_HALF = TILE_WIDTH // 2
    TILE_HEIGHT_HALF = TILE_HEIGHT // 2
    
    # Colors
    COLOR_BG = (25, 20, 40)
    COLOR_GRID = (45, 40, 60)
    COLOR_PLAYER = (50, 255, 255)
    COLOR_PLAYER_OUTLINE = (200, 255, 255)
    COLOR_ENEMY_BASE = (255, 120, 0)
    COLOR_ENEMY_ALERT = (255, 50, 50)
    COLOR_GEM_LOW = (100, 150, 255)
    COLOR_GEM_HIGH = (255, 100, 150)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_WHITE = (255, 255, 255)

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 36, bold=True)

        # Isometric world origin
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT // 2 - (self.GRID_H * self.TILE_HEIGHT_HALF) // 2 + 30
        
        # Initialize state variables
        self.player = {}
        self.enemies = []
        self.gems = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.np_random = None

        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.particles = []
        
        # Player State
        self.player = {
            "pos": np.array([self.GRID_W // 2, self.GRID_H // 2]),
            "move_cooldown": 0,
            "invincible_timer": 0,
            "hits": 0,
            "gems_collected": 0,
        }
        
        # Enemy State
        self.enemies = []
        self.enemy_speed = 0.02 # Grid units per step
        for _ in range(self.NUM_ENEMIES):
            self._spawn_enemy()

        # Gem State
        self.gems = []
        for _ in range(self.NUM_GEMS):
            self._spawn_gem()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = -0.01 # Small penalty for existing, encourages faster play

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            
            # --- Update Game Logic ---
            self._update_player(movement)
            self._update_enemies()
            self._update_gems()
            self._update_particles()
            
            # --- Collisions and Rewards ---
            collected_gem_value = self._check_gem_collisions()
            if collected_gem_value > 0:
                reward += collected_gem_value
                self.score += collected_gem_value
            
            if self._check_enemy_collisions():
                reward -= 10 # Penalty for getting hit

            # --- Difficulty Scaling ---
            if self.steps > 0 and self.steps % 300 == 0: # Every 10 seconds
                self.enemy_speed = min(0.06, self.enemy_speed + 0.005)

        # --- Termination Check ---
        self.steps += 1
        terminated = False
        if self.player["gems_collected"] >= self.WIN_GEMS:
            terminated = True
            self.game_over = True
            self.game_won = True
            reward += 100
        elif self.player["hits"] >= self.MAX_HITS:
            terminated = True
            self.game_over = True
            reward -= 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    # --- Update Methods ---
    def _update_player(self, movement):
        # Cooldowns
        if self.player["move_cooldown"] > 0:
            self.player["move_cooldown"] -= 1
        if self.player["invincible_timer"] > 0:
            self.player["invincible_timer"] -= 1

        # Movement
        if self.player["move_cooldown"] == 0:
            moved = False
            pos = self.player["pos"]
            if movement == 1 and pos[1] > 0: # Up
                pos[1] -= 1; moved = True
            elif movement == 2 and pos[1] < self.GRID_H - 1: # Down
                pos[1] += 1; moved = True
            elif movement == 3 and pos[0] > 0: # Left
                pos[0] -= 1; moved = True
            elif movement == 4 and pos[0] < self.GRID_W - 1: # Right
                pos[0] += 1; moved = True
            
            if moved:
                self.player["move_cooldown"] = 4 # 4 frame cooldown

    def _update_enemies(self):
        for enemy in self.enemies:
            target_pos = enemy["path"][enemy["path_index"]]
            direction = target_pos - enemy["pos"]
            dist = np.linalg.norm(direction)

            if dist < self.enemy_speed:
                enemy["pos"] = target_pos.copy()
                enemy["path_index"] = (enemy["path_index"] + 1) % len(enemy["path"])
            else:
                enemy["pos"] += (direction / dist) * self.enemy_speed

    def _update_gems(self):
        for gem in self.gems:
            min_dist_to_enemy = 1000
            if self.enemies:
                for enemy in self.enemies:
                    dist = np.linalg.norm(gem["pos"] - enemy["pos"])
                    min_dist_to_enemy = min(min_dist_to_enemy, dist)
            
            # Value is higher when closer to an enemy (risk/reward)
            # Map distance (e.g., 0-10) to value (5-1)
            value_factor = 1.0 - min(1.0, max(0.0, (min_dist_to_enemy - 1.0) / 8.0))
            gem["value"] = 1 + int(value_factor * 4) # Value from 1 to 5
            gem["color"] = self._lerp_color(self.COLOR_GEM_LOW, self.COLOR_GEM_HIGH, value_factor)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

    # --- Collision and Spawning ---
    def _check_gem_collisions(self):
        collected_gem_index = -1
        for i, gem in enumerate(self.gems):
            if np.array_equal(self.player["pos"], gem["pos"]):
                collected_gem_index = i
                break
        
        if collected_gem_index != -1:
            gem = self.gems.pop(collected_gem_index)
            self.player["gems_collected"] += 1
            self._spawn_gem()
            self._create_particles(self._iso_to_screen(*gem["pos"]), gem["color"])
            # sfx: gem collect sound
            return gem["value"]
        return 0

    def _check_enemy_collisions(self):
        if self.player["invincible_timer"] > 0:
            return False
            
        for enemy in self.enemies:
            dist = np.linalg.norm(self.player["pos"] - enemy["pos"])
            if dist < 0.8: # Collision radius
                self.player["hits"] += 1
                self.player["invincible_timer"] = 60 # 2 seconds of invincibility
                # sfx: player hit sound
                return True
        return False

    def _spawn_gem(self):
        pos = self.np_random.integers(0, [self.GRID_W, self.GRID_H], size=2)
        # Ensure it doesn't spawn on the player or another gem
        while np.array_equal(pos, self.player["pos"]) or any(np.array_equal(pos, g["pos"]) for g in self.gems):
            pos = self.np_random.integers(0, [self.GRID_W, self.GRID_H], size=2)
        
        self.gems.append({"pos": pos, "value": 1, "color": self.COLOR_GEM_LOW})

    def _spawn_enemy(self):
        # Create a rectangular patrol path
        x, y = self.np_random.integers(1, [self.GRID_W - 2, self.GRID_H - 2])
        w, h = self.np_random.integers(2, 5, size=2)
        w = min(w, self.GRID_W - 1 - x)
        h = min(h, self.GRID_H - 1 - y)
        path = [
            np.array([x, y], dtype=float),
            np.array([x + w, y], dtype=float),
            np.array([x + w, y + h], dtype=float),
            np.array([x, y + h], dtype=float),
        ]
        
        start_pos_idx = self.np_random.integers(0, 4)
        start_pos = path[start_pos_idx].copy()
        
        self.enemies.append({
            "pos": start_pos,
            "path": path,
            "path_index": (start_pos_idx + 1) % 4
        })

    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_gems()
        self._render_enemies()
        self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for r in range(self.GRID_H + 1):
            start = self._iso_to_screen(0, r)
            end = self._iso_to_screen(self.GRID_W, r)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for c in range(self.GRID_W + 1):
            start = self._iso_to_screen(c, 0)
            end = self._iso_to_screen(c, self.GRID_H)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

    def _render_gems(self):
        for gem in self.gems:
            screen_pos = self._iso_to_screen(*gem["pos"])
            self._draw_hexagon(self.screen, gem["color"], screen_pos, self.TILE_HEIGHT_HALF * 0.8)

    def _render_enemies(self):
        for enemy in self.enemies:
            screen_pos = self._iso_to_screen(*enemy["pos"])
            
            # Color shifts based on proximity to player
            dist_to_player = np.linalg.norm(self.player["pos"] - enemy["pos"])
            alert_factor = 1.0 - min(1.0, max(0.0, (dist_to_player - 1.0) / 4.0))
            color = self._lerp_color(self.COLOR_ENEMY_BASE, self.COLOR_ENEMY_ALERT, alert_factor)
            
            # Triangle pointing up
            p1 = (screen_pos[0], screen_pos[1] - self.TILE_HEIGHT_HALF * 0.8)
            p2 = (screen_pos[0] - self.TILE_WIDTH_HALF * 0.6, screen_pos[1] + self.TILE_HEIGHT_HALF * 0.4)
            p3 = (screen_pos[0] + self.TILE_WIDTH_HALF * 0.6, screen_pos[1] + self.TILE_HEIGHT_HALF * 0.4)
            
            pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), color)
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), color)

    def _render_player(self):
        # Flash when invincible
        if self.player["invincible_timer"] > 0 and (self.steps // 3) % 2 == 0:
            return 

        pos = self.player["pos"]
        center = self._iso_to_screen(pos[0] + 0.5, pos[1] + 0.5)
        
        # Draw isometric square
        points = [
            self._iso_to_screen(pos[0], pos[1]),
            self._iso_to_screen(pos[0] + 1, pos[1]),
            self._iso_to_screen(pos[0] + 1, pos[1] + 1),
            self._iso_to_screen(pos[0], pos[1] + 1),
        ]
        
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER_OUTLINE)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0,0, p["size"], p["size"]))
            self.screen.blit(temp_surf, (int(p["pos"][0]), int(p["pos"][1])), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score and Gems
        score_text = f"SCORE: {int(self.score)}"
        gems_text = f"GEMS: {self.player['gems_collected']}/{self.WIN_GEMS}"
        
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        gems_surf = self.font_ui.render(gems_text, True, self.COLOR_UI_TEXT)
        
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))
        self.screen.blit(gems_surf, (10, 10))

        # Hits
        hits_text = f"HITS: {self.player['hits']}/{self.MAX_HITS}"
        hits_surf = self.font_ui.render(hits_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(hits_surf, (10, 30))
        
        # Game Over Message
        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            color = self.COLOR_PLAYER if self.game_won else self.COLOR_ENEMY_ALERT
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    # --- Helper Functions ---
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "hits": self.player["hits"],
            "gems_collected": self.player["gems_collected"],
        }
    
    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.origin_x + (grid_x - grid_y) * self.TILE_WIDTH_HALF
        screen_y = self.origin_y + (grid_x + grid_y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _lerp_color(self, c1, c2, t):
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t),
        )

    def _draw_hexagon(self, surface, color, center, radius):
        points = []
        for i in range(6):
            angle = math.pi / 3 * i + math.pi / 6 # Rotated to be flat on top/bottom
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": np.array(pos, dtype=float),
                "vel": vel,
                "life": life,
                "max_life": life,
                "color": color,
                "size": self.np_random.integers(2, 5),
            })
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
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

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Mapping from Pygame keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Use a separate screen for display
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Collector")

    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        for key, move_action in key_to_action.items():
            if keys[key]:
                action[0] = move_action
                break
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Wait 3 seconds
            obs, info = env.reset()
            total_reward = 0

    env.close()