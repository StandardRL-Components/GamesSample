
# Generated: 2025-08-27T12:38:01.342570
# Source Brief: brief_00110.md
# Brief Index: 110

        
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
        "Controls: Arrow keys to move your green avatar. Survive the monster onslaught for 60 seconds."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a 60-second onslaught of colorful, procedurally generated monsters in an isometric 2D arena by dodging and collecting power-ups."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRID_SIZE_X, self.GRID_SIZE_Y = 8, 8
        self.TILE_WIDTH, self.TILE_HEIGHT = 64, 32
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = self.HEIGHT // 2 - (self.GRID_SIZE_Y * self.TILE_HEIGHT // 4)
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_PLAYER = (0, 255, 127)
        self.COLOR_MONSTERS = [(255, 69, 0), (255, 165, 0), (30, 144, 255)]
        self.COLOR_SPEED_BOOST = (255, 255, 0)
        self.COLOR_INVULNERABILITY = (148, 0, 211)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_SHADOW = (0, 0, 0)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Game state variables (initialized in reset)
        self.player = None
        self.monsters = None
        self.powerups = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.timer = None
        self.stationary_counter = None
        self.powerup_spawn_timer = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_STEPS
        self.stationary_counter = 0
        self.powerup_spawn_timer = 5 * self.FPS # Spawn first power-up in 5s

        # Player state
        self.player = {
            "grid_pos": np.array([self.GRID_SIZE_X // 2, self.GRID_SIZE_Y // 2]),
            "screen_pos": self._iso_to_screen(self.GRID_SIZE_X // 2, self.GRID_SIZE_Y // 2),
            "speed_boost_timer": 0,
            "invulnerability_timer": 0,
            "size": 12
        }

        # Monster state
        self.monsters = []
        for i in range(3):
            path = self._generate_monster_path(i)
            start_pos = path[0]
            self.monsters.append({
                "grid_pos": np.array(start_pos),
                "screen_pos": self._iso_to_screen(start_pos[0], start_pos[1]),
                "path": path,
                "path_index": 0,
                "color": self.COLOR_MONSTERS[i],
                "size": 10
            })

        self.powerups = []
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0

        self._handle_input(movement)
        self._update_player()
        self._update_monsters()
        self._update_powerups()
        self._update_particles()
        
        # Calculate rewards and check for game events
        reward += 0.1  # Survival reward
        self.score += 0.1

        if self.stationary_counter > 5:
            reward -= 0.2

        collision_reward, terminated_by_collision = self._check_collisions()
        reward += collision_reward
        
        self.timer -= 1
        self.steps += 1
        
        # Check termination conditions
        win = self.timer <= 0
        lose = terminated_by_collision
        terminated = win or lose

        if win:
            reward += 100
            self.score += 100
        if lose:
            reward -= 50
            self.score -= 50
        
        if terminated:
            self.game_over = True

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
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer / self.FPS,
            "invulnerable": self.player["invulnerability_timer"] > 0,
            "speed_boost": self.player["speed_boost_timer"] > 0,
        }

    # --- Game Logic ---

    def _handle_input(self, movement):
        if movement == 0: # no-op
            self.stationary_counter += 1
            return
        
        self.stationary_counter = 0
        target_pos = self.player["grid_pos"].copy()
        
        if movement == 1:   # Up
            target_pos += np.array([-1, 0])
        elif movement == 2: # Down
            target_pos += np.array([1, 0])
        elif movement == 3: # Left
            target_pos += np.array([0, -1])
        elif movement == 4: # Right
            target_pos += np.array([0, 1])

        if 0 <= target_pos[0] < self.GRID_SIZE_X and 0 <= target_pos[1] < self.GRID_SIZE_Y:
            self.player["grid_pos"] = target_pos

    def _update_player(self):
        target_screen_pos = self._iso_to_screen(*self.player["grid_pos"])
        lerp_factor = 0.5 if self.player["speed_boost_timer"] > 0 else 0.25
        self.player["screen_pos"] = (
            self.player["screen_pos"][0] + (target_screen_pos[0] - self.player["screen_pos"][0]) * lerp_factor,
            self.player["screen_pos"][1] + (target_screen_pos[1] - self.player["screen_pos"][1]) * lerp_factor
        )
        if self.player["speed_boost_timer"] > 0:
            self.player["speed_boost_timer"] -= 1
        if self.player["invulnerability_timer"] > 0:
            self.player["invulnerability_timer"] -= 1
            
    def _update_monsters(self):
        # Monsters move every 10 frames (3 times per second)
        if self.steps > 0 and self.steps % 10 == 0:
            for m in self.monsters:
                m["path_index"] = (m["path_index"] + 1) % len(m["path"])
                m["grid_pos"] = np.array(m["path"][m["path_index"]])

        for m in self.monsters:
            target_screen_pos = self._iso_to_screen(*m["grid_pos"])
            lerp_factor = 0.1
            m["screen_pos"] = (
                m["screen_pos"][0] + (target_screen_pos[0] - m["screen_pos"][0]) * lerp_factor,
                m["screen_pos"][1] + (target_screen_pos[1] - m["screen_pos"][1]) * lerp_factor
            )

    def _update_powerups(self):
        self.powerup_spawn_timer -= 1
        if self.powerup_spawn_timer <= 0:
            self._spawn_powerup()
            self.powerup_spawn_timer = 5 * self.FPS # Reset timer

        for p in self.powerups:
            p["lifetime"] -= 1
        self.powerups = [p for p in self.powerups if p["lifetime"] > 0]

    def _update_particles(self):
        for p in self.particles:
            p["pos"] = (p["pos"][0] + p["vel"][0], p["pos"][1] + p["vel"][1])
            p["lifetime"] -= 1
        self.particles = [p for p in self.particles if p["lifetime"] > 0]

    def _check_collisions(self):
        reward = 0
        terminated = False
        
        # Player-monster
        if self.player["invulnerability_timer"] <= 0:
            for m in self.monsters:
                dist = np.linalg.norm(np.array(self.player["screen_pos"]) - np.array(m["screen_pos"]))
                if dist < self.player["size"] + m["size"]:
                    terminated = True
                    self._create_particles(self.player["screen_pos"], self.COLOR_PLAYER, 30)
                    # sfx: player_hit
                    break
        
        # Player-powerup
        for p in self.powerups[:]:
            dist = np.linalg.norm(np.array(self.player["screen_pos"]) - np.array(p["screen_pos"]))
            if dist < self.player["size"] + p["size"]:
                reward += 5
                self.score += 5
                if p["type"] == "speed":
                    self.player["speed_boost_timer"] = 3 * self.FPS
                elif p["type"] == "invulnerability":
                    self.player["invulnerability_timer"] = 2 * self.FPS
                self._create_particles(p["screen_pos"], p["color"], 20)
                self.powerups.remove(p)
                # sfx: powerup_collect
        
        return reward, terminated
    
    # --- Spawning and Generation ---
    
    def _generate_monster_path(self, monster_index):
        path = []
        start_x = self.np_random.integers(0, self.GRID_SIZE_X)
        start_y = self.np_random.integers(0, self.GRID_SIZE_Y)
        
        path_type = monster_index % 3
        if path_type == 0: # Square path
            size = self.np_random.integers(2, 5)
            for i in range(size): path.append((min(start_x + i, 7), start_y))
            for i in range(size): path.append((start_x + size -1, min(start_y + i, 7)))
            for i in range(size): path.append((max(start_x + size - 1 - i, 0), start_y + size -1))
            for i in range(size): path.append((start_x, max(start_y + size - 1 - i, 0)))
        elif path_type == 1: # Horizontal path
            length = self.np_random.integers(3, 8)
            for i in range(length): path.append(((start_x + i) % self.GRID_SIZE_X, start_y))
            for i in range(length): path.append(((start_x + length - 1 - i) % self.GRID_SIZE_X, start_y))
        else: # Vertical path
            length = self.np_random.integers(3, 8)
            for i in range(length): path.append((start_x, (start_y + i) % self.GRID_SIZE_Y))
            for i in range(length): path.append((start_x, (start_y + length - 1 - i) % self.GRID_SIZE_Y))
        
        return path if path else [(start_x, start_y)]

    def _spawn_powerup(self):
        if len(self.powerups) >= 3: return
        
        occupied_cells = [tuple(self.player["grid_pos"])] + [tuple(m["grid_pos"]) for m in self.monsters] + [tuple(p["grid_pos"]) for p in self.powerups]
        
        gx, gy = self.np_random.integers(0, self.GRID_SIZE_X), self.np_random.integers(0, self.GRID_SIZE_Y)
        while (gx, gy) in occupied_cells:
            gx, gy = self.np_random.integers(0, self.GRID_SIZE_X), self.np_random.integers(0, self.GRID_SIZE_Y)
        
        ptype = self.np_random.choice(["speed", "invulnerability"])
        color = self.COLOR_SPEED_BOOST if ptype == "speed" else self.COLOR_INVULNERABILITY
        
        self.powerups.append({
            "grid_pos": (gx, gy),
            "screen_pos": self._iso_to_screen(gx, gy),
            "type": ptype,
            "color": color,
            "size": 8,
            "lifetime": 10 * self.FPS
        })
        
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifetime": self.np_random.integers(10, 20),
                "color": color,
                "size": self.np_random.integers(2, 4)
            })
            
    # --- Rendering ---

    def _iso_to_screen(self, gx, gy):
        sx = self.ORIGIN_X + (gx - gy) * (self.TILE_WIDTH // 2)
        sy = self.ORIGIN_Y + (gx + gy) * (self.TILE_HEIGHT // 2)
        return sx, sy

    def _render_game(self):
        self._render_grid()
        
        # Collect and sort all dynamic entities for correct draw order
        entities = []
        entities.append({"type": "player", "z": self.player["grid_pos"][0] + self.player["grid_pos"][1], "obj": self.player})
        for m in self.monsters:
            entities.append({"type": "monster", "z": m["grid_pos"][0] + m["grid_pos"][1], "obj": m})
        for p in self.powerups:
            entities.append({"type": "powerup", "z": p["grid_pos"][0] + p["grid_pos"][1], "obj": p})
            
        entities.sort(key=lambda e: e["z"])

        # Render shadows first, then entities
        for e in entities:
            self._render_shadow(e["obj"])
        for e in entities:
            if e["type"] == "player": self._render_player()
            elif e["type"] == "monster": self._render_monster(e["obj"])
            elif e["type"] == "powerup": self._render_powerup(e["obj"])
            
        self._render_particles()

    def _render_grid(self):
        for y in range(self.GRID_SIZE_Y):
            for x in range(self.GRID_SIZE_X):
                p1 = self._iso_to_screen(x, y)
                p2 = self._iso_to_screen(x + 1, y)
                p3 = self._iso_to_screen(x + 1, y + 1)
                p4 = self._iso_to_screen(x, y + 1)
                pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3, p4], self.COLOR_GRID)

    def _render_shadow(self, entity):
        pos = entity["screen_pos"]
        size = entity["size"]
        shadow_rect = pygame.Rect(0, 0, size * 1.8, size * 0.9)
        shadow_rect.center = (pos[0], pos[1] + size * 1.5)
        shadow_surface = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, (0, 0, 0, 80), (0, 0, *shadow_rect.size))
        self.screen.blit(shadow_surface, shadow_rect.topleft)

    def _render_player(self):
        pos = (int(self.player["screen_pos"][0]), int(self.player["screen_pos"][1]))
        size = self.player["size"]
        
        # Invulnerability shield
        if self.player["invulnerability_timer"] > 0:
            alpha = 100 + 50 * math.sin(self.steps * 0.5)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size + 5, (*self.COLOR_INVULNERABILITY, alpha))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size + 5, (*self.COLOR_INVULNERABILITY, alpha))

        # Speed boost trail
        if self.player["speed_boost_timer"] > 0:
            self._create_particles(pos, self.COLOR_SPEED_BOOST, 1)

        # Glow effect
        for i in range(size // 2, 0, -1):
            alpha = 80 - (i / (size // 2)) * 80
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size + i, (*self.COLOR_PLAYER, alpha))
        
        # Player body
        rect = pygame.Rect(0, 0, size * 1.5, size * 1.5)
        rect.center = pos
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=3)
        
    def _render_monster(self, monster):
        pos = (int(monster["screen_pos"][0]), int(monster["screen_pos"][1]))
        size = monster["size"]
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, monster["color"])
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, monster["color"])
        
    def _render_powerup(self, powerup):
        pos = (int(powerup["screen_pos"][0]), int(powerup["screen_pos"][1]))
        size = powerup["size"]
        
        # Pulsing glow
        pulse = size + 2 * math.sin(self.steps * 0.1)
        alpha = 150 + 50 * math.sin(self.steps * 0.1)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(pulse), (*powerup["color"], alpha/2))
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(pulse), (*powerup["color"], alpha))
        
        # Core
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, powerup["color"])
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, powerup["color"])
    
    def _render_particles(self):
        for p in self.particles:
            alpha = 255 * (p["lifetime"] / 20)
            color = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["size"]), color)

    def _render_ui(self):
        # Timer
        time_left = max(0, self.timer / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        timer_surf = self.font_small.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 10, 10))
        
        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 40))

        # Game Over / Win message
        if self.game_over:
            message = "YOU SURVIVED!" if self.timer <= 0 else "GAME OVER"
            msg_surf = self.font_large.render(message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            # Draw a semi-transparent background for readability
            bg_rect = msg_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, bg_rect.topleft)
            self.screen.blit(msg_surf, msg_rect)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Survivor")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, 0, 0] # Space and shift are not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Press 'R' to restart.")
        
        clock.tick(env.FPS)
        
    pygame.quit()