import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move on the isometric grid. Collect gems, avoid enemies."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect glittering gems while dodging cunning enemies in an isometric arcade world."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # Game constants
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 20
        self.MAX_STEPS = 1000
        self.WIN_GEMS = 50
        self.MAX_HEALTH = 100
        self.INITIAL_GEMS = 10

        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_GRID = (20, 40, 60)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_OUTLINE = (200, 255, 255)
        self.COLOR_GEM = (255, 223, 0)
        self.COLOR_ENEMY_PATROL = (255, 50, 50)
        self.COLOR_ENEMY_RANDOM = (200, 50, 200)
        self.COLOR_ENEMY_CHASER = (255, 100, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)
        self.COLOR_HEALTH_BAR_FG = (200, 0, 0)

        # Isometric projection constants
        self.TILE_W = 28
        self.TILE_H = 14
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80

        # Initialize state variables
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gems_collected = 0
        self.player_health = 0
        self.last_hit_timer = 0
        self.player_pos = [0, 0]
        self.gems = []
        self.enemies = []
        self.particles = []

        # This is a dummy call to initialize np_random before validation
        self.reset()
        # self.validate_implementation() # Commented out for submission, as it runs a step

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gems_collected = 0
        self.player_health = self.MAX_HEALTH
        self.last_hit_timer = 0

        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.gems = []
        self.enemies = []
        self.particles = []

        occupied_positions = {tuple(self.player_pos)}

        # Spawn enemies
        self.enemies.append({
            "type": "patrol", "pos": [3, 3],
            "path": [(3, 3), (3, 8), (8, 8), (8, 3)], "path_index": 0,
            "color": self.COLOR_ENEMY_PATROL
        })
        occupied_positions.add((3, 3))

        self.enemies.append({
            "type": "random", "pos": [15, 15],
            "zone": (13, 13, 17, 17),
            "color": self.COLOR_ENEMY_RANDOM
        })
        occupied_positions.add((15, 15))

        self.enemies.append({
            "type": "chaser", "pos": [15, 5], "radius": 7,
            "color": self.COLOR_ENEMY_CHASER
        })
        occupied_positions.add((15, 5))

        for _ in range(self.INITIAL_GEMS):
            self._spawn_gem(occupied_positions)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0

        dist_before = self._find_closest_gem_dist()
        self._move_player(movement)
        dist_after = self._find_closest_gem_dist()

        if dist_after < dist_before:
            reward += 1.0
        elif dist_after > dist_before:
            reward -= 0.1

        self._update_enemies()

        reward += self._handle_collisions_and_events()
        # The erroneous call to _update_particles was here. It's removed.
        # Particle updates are handled correctly within the rendering pipeline.
        if self.last_hit_timer > 0: self.last_hit_timer -= 1

        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            if self.gems_collected >= self.WIN_GEMS:
                reward += 100.0
                self.score += 100
            elif self.player_health <= 0:
                reward -= 100.0
                self.score -= 100
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _move_player(self, movement):
        prev_pos = tuple(self.player_pos)
        if movement == 1: self.player_pos[1] -= 1  # Up -> Up-Left
        elif movement == 2: self.player_pos[1] += 1  # Down -> Down-Right
        elif movement == 3: self.player_pos[0] -= 1  # Left -> Down-Left
        elif movement == 4: self.player_pos[0] += 1  # Right -> Up-Right

        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_WIDTH - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_HEIGHT - 1)

        if tuple(self.player_pos) != prev_pos:
            px, py = self._iso_to_cart(self.player_pos[0], self.player_pos[1])
            self._create_particles((px, py), self.COLOR_PLAYER, count=1, speed=0.5, life=10, size=1)

    def _handle_collisions_and_events(self):
        event_reward = 0.0

        gem_to_remove = next((g for g in self.gems if self.player_pos == g), None)
        if gem_to_remove:
            self.gems.remove(gem_to_remove)
            self.gems_collected += 1
            self.score += 10
            event_reward += 10.0

            is_risky = any(self._manhattan_distance(self.player_pos, e["pos"]) <= 1 for e in self.enemies)
            if is_risky:
                self.score += 2
                event_reward += 2.0

            # sfx: gem collect
            px, py = self._iso_to_cart(gem_to_remove[0], gem_to_remove[1])
            self._create_particles((px, py + self.TILE_H // 2), self.COLOR_GEM, count=20, speed=3, life=20, size=2)
            self._spawn_gem()

        for enemy in self.enemies:
            if self.player_pos == enemy["pos"]:
                if self.last_hit_timer <= 0: # Prevent rapid re-hits
                    self.player_health -= 25
                    self.score -= 5
                    event_reward -= 5.0
                    self.last_hit_timer = 10
                    # sfx: player hit
        
        return event_reward

    def _check_termination(self):
        return (
            self.gems_collected >= self.WIN_GEMS
            or self.player_health <= 0
            or self.steps >= self.MAX_STEPS
        )

    def _update_enemies(self):
        for enemy in self.enemies:
            if enemy["type"] == "patrol":
                target = enemy["path"][enemy["path_index"]]
                if enemy["pos"] == list(target):
                    enemy["path_index"] = (enemy["path_index"] + 1) % len(enemy["path"])
                    target = enemy["path"][enemy["path_index"]]
                enemy["pos"] = self._move_towards(enemy["pos"], target)
            elif enemy["type"] == "random":
                if self.np_random.random() < 0.2: # Move occasionally
                    x_min, y_min, x_max, y_max = enemy["zone"]
                    dx = self.np_random.integers(-1, 2)
                    dy = self.np_random.integers(-1, 2)
                    new_x = np.clip(enemy["pos"][0] + dx, x_min, x_max)
                    new_y = np.clip(enemy["pos"][1] + dy, y_min, y_max)
                    enemy["pos"] = [new_x, new_y]
            elif enemy["type"] == "chaser":
                dist = self._manhattan_distance(self.player_pos, enemy["pos"])
                if dist <= enemy["radius"]:
                    enemy["pos"] = self._move_towards(enemy["pos"], self.player_pos)
                elif self.np_random.random() < 0.1: # Wander if player is far
                    dx = self.np_random.integers(-1, 2)
                    dy = self.np_random.integers(-1, 2)
                    enemy["pos"][0] = np.clip(enemy["pos"][0] + dx, 0, self.GRID_WIDTH - 1)
                    enemy["pos"][1] = np.clip(enemy["pos"][1] + dy, 0, self.GRID_HEIGHT - 1)

    def _move_towards(self, start, end):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if abs(dx) > abs(dy):
            return [start[0] + int(np.sign(dx)), start[1]]
        elif abs(dy) > 0:
            return [start[0], start[1] + int(np.sign(dy))]
        return start

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()

        entities_to_draw = []
        for gem_pos in self.gems:
            entities_to_draw.append({"pos": gem_pos, "type": "gem", "y_sort": gem_pos[0] + gem_pos[1]})
        for enemy in self.enemies:
            entities_to_draw.append({"pos": enemy["pos"], "type": "enemy", "enemy_data": enemy, "y_sort": enemy["pos"][0] + enemy["pos"][1]})
        entities_to_draw.append({"pos": self.player_pos, "type": "player", "y_sort": self.player_pos[0] + self.player_pos[1] + 0.1}) # Player on top

        entities_to_draw.sort(key=lambda e: e["y_sort"])

        self._update_and_draw_particles()

        for entity in entities_to_draw:
            if entity["type"] == "gem": self._draw_gem(entity["pos"])
            elif entity["type"] == "enemy": self._draw_enemy(entity["enemy_data"])
            elif entity["type"] == "player": self._draw_player()

        if self.last_hit_timer > 0:
            hit_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            alpha = int(100 * (self.last_hit_timer / 10.0))
            hit_surface.fill((255, 0, 0, alpha))
            self.screen.blit(hit_surface, (0, 0))

    def _draw_grid(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                px, py = self._iso_to_cart(c, r)
                points = [
                    (px, py + self.TILE_H),
                    (px + self.TILE_W / 2, py + self.TILE_H / 2),
                    (px, py),
                    (px - self.TILE_W / 2, py + self.TILE_H / 2),
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

    def _iso_to_cart(self, iso_x, iso_y):
        cart_x = self.ORIGIN_X + (iso_x - iso_y) * (self.TILE_W / 2)
        cart_y = self.ORIGIN_Y + (iso_x + iso_y) * (self.TILE_H / 2)
        return int(cart_x), int(cart_y)

    def _draw_iso_poly(self, pos, color, height_offset=0):
        px, py_top = self._iso_to_cart(pos[0], pos[1])
        py_top += height_offset
        
        base_h = self.TILE_H
        points = [
            self._iso_to_cart(pos[0], pos[1] + 1),
            self._iso_to_cart(pos[0] + 1, pos[1] + 1),
            self._iso_to_cart(pos[0] + 1, pos[1]),
            self._iso_to_cart(pos[0], pos[1]),
        ]
        points = [(p[0], p[1] + height_offset) for p in points]
        
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_player(self):
        px, py_top = self._iso_to_cart(self.player_pos[0], self.player_pos[1])
        py_center = py_top + self.TILE_H / 2
        # Glow effect
        glow_radius = int(self.TILE_W * 0.6)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER, 30), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (px - glow_radius, py_center - glow_radius))

        self._draw_iso_poly(self.player_pos, self.COLOR_PLAYER_OUTLINE)
        self._draw_iso_poly(self.player_pos, self.COLOR_PLAYER, height_offset=-2)

    def _draw_gem(self, pos):
        pulse = abs(math.sin(self.steps * 0.2 + pos[0])) * 4
        self._draw_iso_poly(pos, self.COLOR_GEM, height_offset=-pulse - 2)
    
    def _draw_enemy(self, enemy):
        color = enemy["color"]
        if enemy["type"] == "chaser" and self._manhattan_distance(self.player_pos, enemy["pos"]) <= enemy["radius"]:
            color = (255, 255, 0) # Alert color
        self._draw_iso_poly(enemy["pos"], color)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 10))
        # Gems
        gem_text = self.font_small.render(f"GEMS: {self.gems_collected} / {self.WIN_GEMS}", True, self.COLOR_TEXT)
        self.screen.blit(gem_text, (10, 10))
        # Health Bar
        health_pct = max(0, self.player_health / self.MAX_HEALTH)
        bar_w, bar_h = 150, 20
        bar_x, bar_y = self.WIDTH - bar_w - 10, 10
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (bar_x, bar_y, int(bar_w * health_pct), bar_h))
        health_text = self.font_small.render("HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (bar_x + (bar_w - health_text.get_width()) // 2, bar_y + (bar_h - health_text.get_height()) // 2))

    def _get_info(self):
        return { "score": self.score, "steps": self.steps, "gems": self.gems_collected, "health": self.player_health }

    def _spawn_gem(self, occupied_positions=None):
        if occupied_positions is None:
            occupied_positions = {tuple(self.player_pos)} | {tuple(e["pos"]) for e in self.enemies} | {tuple(g) for g in self.gems}
        
        pos = None
        for _ in range(100): # Max attempts to find a spot
            candidate_pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
            if candidate_pos not in occupied_positions:
                pos = list(candidate_pos)
                break
        if pos is None: # Fallback if grid is full
            pos = [0, 0]
        self.gems.append(pos)
        if occupied_positions:
            occupied_positions.add(tuple(pos))

    def _create_particles(self, pos, color, count, speed, life, size=1):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_mag = self.np_random.uniform(speed * 0.5, speed * 1.5)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * vel_mag, math.sin(angle) * vel_mag],
                "life": self.np_random.integers(life // 2, life),
                "max_life": life,
                "color": color,
                "size": size
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p["life"] / p["max_life"]))
                color = (*p["color"], alpha)
                surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (p["size"], p["size"]), p["size"])
                self.screen.blit(surf, (int(p["pos"][0]-p["size"]), int(p["pos"][1]-p["size"])))

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _find_closest_gem_dist(self):
        if not self.gems: return float('inf')
        return min(self._manhattan_distance(self.player_pos, gem_pos) for gem_pos in self.gems)
        
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
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11"
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gem Collector")
    clock = pygame.time.Clock()

    action = np.array([0, 0, 0]) # No-op, no space, no shift

    print("\n" + "="*30)
    print("Gem Collector - Manual Control")
    print(env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # Map keyboard to MultiDiscrete action space
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

        if terminated:
            print(f"Game Over! Final Info: {info}")
            # Wait a bit before closing
            pygame.time.wait(2000)

    env.close()