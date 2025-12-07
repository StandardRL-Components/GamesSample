
# Generated: 2025-08-28T03:47:17.742891
# Source Brief: brief_05036.md
# Brief Index: 5036

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to select jump direction. The ship will hop to the nearest platform in that direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop your spaceship between scrolling platforms to collect 5 coins. Don't fall off the bottom!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    WIN_COIN_COUNT = 5
    PLATFORM_COUNT = 10
    
    # --- Colors ---
    COLOR_BG = (10, 20, 40)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (100, 200, 255, 100)
    COLOR_PLATFORM = (100, 110, 120)
    COLOR_GOLD = (255, 215, 0)
    COLOR_SILVER = (192, 192, 192)
    COLOR_TEXT = (240, 240, 240)

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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.render_mode = render_mode
        self.np_random = None # Will be initialized in reset
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.coins_collected = 0
        self.game_over = False
        self.win = False

        self.base_platform_speed = 1.0

        # Player state
        self.player = {
            "x": self.SCREEN_WIDTH / 2,
            "y": self.SCREEN_HEIGHT / 2,
            "size": 10,
            "state": "IDLE",  # IDLE, JUMPING
            "on_platform_idx": None,
            "jump_start_pos": (0, 0),
            "jump_target_pos": (0, 0),
            "jump_progress": 0.0,
            "jump_duration": 20, # frames
        }

        # Game entities
        self.platforms = []
        self.coins = []
        self.stars = [
            {
                "x": self.np_random.integers(0, self.SCREEN_WIDTH),
                "y": self.np_random.integers(0, self.SCREEN_HEIGHT),
                "speed": self.np_random.uniform(0.1, 0.5),
                "size": self.np_random.integers(1, 3),
            }
            for _ in range(100)
        ]
        self.particles = []

        # Initial population
        initial_platform = self._create_platform(
            y=self.player["y"] + self.player["size"] * 2,
            x=self.player["x"] - 50 # Center the platform under the player
        )
        initial_platform['width'] = 100
        self.platforms.append(initial_platform)
        self.player["on_platform_idx"] = 0
        self.player["y"] = initial_platform['y'] - self.player['size'] / 2

        while len(self.platforms) < self.PLATFORM_COUNT:
            self._spawn_platform()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.02  # Time penalty
        
        if self.game_over:
            # If game is already over, do nothing but return final state
            final_reward = -10 if not self.win else 10
            return self._get_observation(), final_reward, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1  (not used)
        # shift_held = action[2] == 1 (not used)

        self._update_difficulty()
        self._update_player(movement)
        self._update_platforms_and_coins()
        self._update_stars()
        self._update_particles()
        
        collision_reward, landing_reward = self._handle_collisions()
        reward += collision_reward
        
        # Landing reward is separate because it can have a risk penalty
        if landing_reward > 0:
            reward += landing_reward
            # Risk penalty for landing near edge
            p = self.platforms[self.player['on_platform_idx']]
            dist_to_edge = min(self.player['x'] - p['x'], p['x'] + p['width'] - self.player['x'])
            if dist_to_edge < 20:
                reward -= 0.2 * max(1, self.score)

        self._cull_entities()
        self._populate_entities()

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.win:
                reward += 10 # Win bonus
            else:
                reward += -10 # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    # --- Update Methods ---

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.base_platform_speed = min(5.0, self.base_platform_speed + 0.1)

    def _update_player(self, movement):
        if self.player["state"] == "JUMPING":
            self.player["jump_progress"] += 1.0 / self.player["jump_duration"]
            
            progress = self.player["jump_progress"]
            start_x, start_y = self.player["jump_start_pos"]
            target_x, target_y = self.player["jump_target_pos"]
            
            self.player["x"] = start_x + (target_x - start_x) * progress
            arc_height = -40 * math.sin(min(1.0, progress) * math.pi)
            self.player["y"] = start_y + (target_y - start_y) * progress + arc_height

            if self.player["jump_progress"] >= 1.0:
                self.player["state"] = "IDLE"
                self.player["jump_progress"] = 0.0
        
        else: # state is "IDLE"
            if self.player["on_platform_idx"] is not None:
                try:
                    platform = self.platforms[self.player["on_platform_idx"]]
                    self.player["y"] += self.base_platform_speed * platform["speed_multiplier"]
                    self.player["x"] = max(platform['x'], min(platform['x'] + platform['width'], self.player['x']))
                except IndexError:
                    self.player["on_platform_idx"] = None
            else:
                self.player["y"] += self.base_platform_speed + 2
            
            if movement != 0:
                target_platform = self._find_target_platform(movement)
                if target_platform:
                    # SFX: Jump
                    self.player["state"] = "JUMPING"
                    self.player["on_platform_idx"] = None
                    self.player["jump_start_pos"] = (self.player["x"], self.player["y"])
                    
                    target_x = self.np_random.uniform(target_platform["x"], target_platform["x"] + target_platform["width"])
                    target_y = target_platform["y"] - self.player["size"] / 2
                    self.player["jump_target_pos"] = (target_x, target_y)

    def _find_target_platform(self, movement):
        px, py = self.player["x"], self.player["y"]
        candidates = []

        if movement == 1: # Up
            candidates = [p for p in self.platforms if p["y"] < py]
        elif movement == 2: # Down
            candidates = [p for p in self.platforms if p["y"] > py]
        elif movement == 3: # Left
            candidates = [p for p in self.platforms if p["x"] + p["width"] < px]
        elif movement == 4: # Right
            candidates = [p for p in self.platforms if p["x"] > px]
        
        if not candidates:
            return None
        
        best_platform = min(
            candidates,
            key=lambda p: math.hypot(p["x"] + p["width"] / 2 - px, p["y"] - py)
        )
        return best_platform

    def _update_platforms_and_coins(self):
        for p in self.platforms:
            p["y"] += self.base_platform_speed * p["speed_multiplier"]
        for c in self.coins:
            c["y"] += self.base_platform_speed * c["platform_speed_multiplier"]
            c["angle"] = (c["angle"] + 5) % 360

    def _update_stars(self):
        for star in self.stars:
            star["y"] += star["speed"] * self.base_platform_speed
            if star["y"] > self.SCREEN_HEIGHT:
                star["y"] = 0
                star["x"] = self.np_random.integers(0, self.SCREEN_WIDTH)

    def _update_particles(self):
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1

    def _handle_collisions(self):
        collision_reward = 0
        landing_reward = 0
        
        if self.player["state"] == "IDLE" and self.player["on_platform_idx"] is None:
            for i, p in enumerate(self.platforms):
                is_on_top = abs(self.player["y"] - (p["y"] - self.player["size"] / 2)) < 5
                is_horizontally_aligned = self.player["x"] > p["x"] and self.player["x"] < p["x"] + p["width"]
                if is_on_top and is_horizontally_aligned:
                    # SFX: Land
                    self.player["on_platform_idx"] = i
                    self.player["y"] = p["y"] - self.player["size"] / 2
                    landing_reward = 0.1 # Landing bonus
                    self._spawn_particles(self.player['x'], self.player['y'] + self.player['size']/2, self.COLOR_PLAYER, 5)
                    break 

        player_rect = pygame.Rect(self.player["x"] - self.player["size"], self.player["y"] - self.player["size"], self.player["size"]*2, self.player["size"]*2)
        for coin in self.coins[:]:
            coin_rect = pygame.Rect(coin["x"] - coin["size"], coin["y"] - coin["size"], coin["size"]*2, coin["size"]*2)
            if player_rect.colliderect(coin_rect):
                # SFX: Coin collect
                self.score += coin["value"]
                self.coins_collected += 1
                collision_reward += coin["value"]
                color = self.COLOR_GOLD if coin['value'] == 1 else self.COLOR_SILVER
                self._spawn_particles(coin['x'], coin['y'], color, 10)
                self.coins.remove(coin)
                break
        
        return collision_reward, landing_reward

    # --- Entity Management ---

    def _cull_entities(self):
        original_platforms = self.platforms
        self.platforms = [p for p in self.platforms if p["y"] < self.SCREEN_HEIGHT + 50]
        
        # Adjust player's platform index after culling
        if self.player['on_platform_idx'] is not None:
            try:
                platform_obj = original_platforms[self.player['on_platform_idx']]
                if platform_obj in self.platforms:
                    self.player['on_platform_idx'] = self.platforms.index(platform_obj)
                else:
                    self.player['on_platform_idx'] = None
            except IndexError:
                self.player['on_platform_idx'] = None

        self.coins = [c for c in self.coins if c["y"] < self.SCREEN_HEIGHT + 50]
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _populate_entities(self):
        while len(self.platforms) < self.PLATFORM_COUNT:
            self._spawn_platform()

    def _create_platform(self, y=None, x=None):
        if y is None:
            y = -20
        if x is None:
            x = self.np_random.integers(0, self.SCREEN_WIDTH - 80)
        
        platform = {
            "x": x,
            "y": y,
            "width": self.np_random.integers(40, 120),
            "speed_multiplier": self.np_random.choice([1.0, 1.2, 1.5], p=[0.6, 0.3, 0.1]),
        }
        return platform

    def _spawn_platform(self):
        y = -20
        if self.platforms:
            y = min(p['y'] for p in self.platforms) - self.np_random.integers(50, 100)

        platform = self._create_platform(y=y)
        self.platforms.append(platform)
        
        if self.np_random.random() < 0.6:
            self._spawn_coin(platform)

    def _spawn_coin(self, platform):
        is_silver = self.np_random.random() < 0.2
        self.coins.append({
            "x": platform["x"] + platform["width"] / 2,
            "y": platform["y"] - 20,
            "size": 8 if is_silver else 6,
            "value": 2 if is_silver else 1,
            "type": "silver" if is_silver else "gold",
            "platform_speed_multiplier": platform["speed_multiplier"],
            "angle": 0
        })

    def _spawn_particles(self, x, y, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "x": x, "y": y,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": self.np_random.integers(10, 20),
                "color": color
            })

    # --- Termination and Info ---

    def _check_termination(self):
        if self.player["y"] > self.SCREEN_HEIGHT + self.player['size']:
            self.game_over = True
            self.win = False
            return True
        if self.coins_collected >= self.WIN_COIN_COUNT:
            self.game_over = True
            self.win = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
            return True
        return False
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "coins_collected": self.coins_collected,
        }

    # --- Rendering ---

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_platforms()
        self._render_coins()
        self._render_player()
        self._render_particles()
        self._render_ui()
        if self.game_over:
            self._render_game_over()

    def _render_stars(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, (200, 200, 220), (int(star["x"]), int(star["y"])), star["size"])

    def _render_platforms(self):
        for p in self.platforms:
            pygame.draw.rect(
                self.screen,
                self.COLOR_PLATFORM,
                pygame.Rect(int(p["x"]), int(p["y"]), int(p["width"]), 10),
                border_radius=3
            )

    def _render_coins(self):
        for c in self.coins:
            color = self.COLOR_GOLD if c["type"] == "gold" else self.COLOR_SILVER
            size = c['size']
            pulse = abs(math.sin(self.steps * 0.1 + c['x'])) * 2
            
            rect_points = []
            for i in range(4):
                angle_rad = math.radians(c['angle'] + i * 90 + 45)
                x = c['x'] + (size + pulse) * math.cos(angle_rad)
                y = c['y'] + (size + pulse) * math.sin(angle_rad)
                rect_points.append((int(x), int(y)))
            pygame.gfxdraw.aapolygon(self.screen, rect_points, color)
            pygame.gfxdraw.filled_polygon(self.screen, rect_points, color)

    def _render_player(self):
        px, py, psize = self.player["x"], self.player["y"], self.player["size"]
        
        glow_surf = pygame.Surface((psize * 4, psize * 4), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, int(psize*2), int(psize*2), int(psize*1.5), self.COLOR_PLAYER_GLOW)
        self.screen.blit(glow_surf, (int(px - psize*2), int(py - psize*2)))

        p1 = (px, py - psize)
        p2 = (px - psize * 0.8, py + psize * 0.8)
        p3 = (px + psize * 0.8, py + psize * 0.8)
        
        if self.player['state'] == 'JUMPING':
            dx = self.player['jump_target_pos'][0] - self.player['jump_start_pos'][0]
            angle = np.clip(dx * -0.1, -20, 20)
            p1 = self._rotate_point(p1, (px, py), angle)
            p2 = self._rotate_point(p2, (px, py), angle)
            p3 = self._rotate_point(p3, (px, py), angle)
            
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)

    def _rotate_point(self, point, center, angle_deg):
        angle_rad = math.radians(angle_deg)
        x, y = point
        cx, cy = center
        new_x = cx + (x - cx) * math.cos(angle_rad) - (y - cy) * math.sin(angle_rad)
        new_y = cy + (x - cx) * math.sin(angle_rad) + (y - cy) * math.cos(angle_rad)
        return new_x, new_y

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / 20.0))
            color = p['color']
            # Create a temporary surface for alpha blending
            surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(surf, color + (alpha,), (2, 2), 2)
            self.screen.blit(surf, (int(p['x']) - 2, int(p['y']) - 2))

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        coins_text = self.font_ui.render(f"COINS: {self.coins_collected} / {self.WIN_COIN_COUNT}", True, self.COLOR_TEXT)
        self.screen.blit(coins_text, (10, 40))

    def _render_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0,0,0,180))
        self.screen.blit(s, (0,0))
        
        message = "YOU WIN!" if self.win else "GAME OVER"
        text_surf = self.font_game_over.render(message, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11'
    
    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.set_caption("Hopping Spaceship")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    action = [0, 0, 0]
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    action = [0,0,0]

        keys = pygame.key.get_pressed()
        
        current_movement = 0
        if keys[pygame.K_UP]: current_movement = 1
        elif keys[pygame.K_DOWN]: current_movement = 2
        elif keys[pygame.K_LEFT]: current_movement = 3
        elif keys[pygame.K_RIGHT]: current_movement = 4
        
        action = [current_movement, 0, 0]

        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

    env.close()