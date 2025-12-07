
# Generated: 2025-08-28T03:02:20.661722
# Source Brief: brief_01889.md
# Brief Index: 1889

        
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
        "Controls: Use arrow keys to move the deployment cursor. "
        "Press Space to build a short-range Turret. "
        "Hold Shift to build a long-range Cannon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of descending aliens in this top-down, real-time strategy game. "
        "Place turrets and cannons strategically to survive as long as possible."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 64, 40
        self.CELL_SIZE = 10
        self.FPS = 30
        self.MAX_STEPS = 3000 # 100 seconds at 30 FPS

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 45, 60)
        self.COLOR_BASE = (0, 100, 0)
        self.COLOR_BASE_HEALTH = (0, 200, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_CURSOR = (255, 255, 255, 100)
        
        self.COLOR_ALIEN = (220, 50, 50)
        self.COLOR_TURRET = (50, 150, 250) # Type 1
        self.COLOR_CANNON = (250, 200, 50) # Type 2
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_EXPLOSION = (255, 150, 0)

        # --- Game Entity Properties ---
        self.MAX_HEALTH = 100
        self.INITIAL_SCORE = 20
        self.ALIEN_SPEED = 1.0
        self.ALIEN_DAMAGE = 10
        self.INITIAL_SPAWN_PROB = 1.0 / 50.0
        self.SPAWN_PROB_INCREASE = 0.001

        self.TURRET_COST = 5
        self.TURRET_RANGE = 7 * self.CELL_SIZE
        self.TURRET_COOLDOWN = 20 # frames
        self.TURRET_PROJ_SPEED = 8

        self.CANNON_COST = 10
        self.CANNON_RANGE = 14 * self.CELL_SIZE
        self.CANNON_COOLDOWN = 45 # frames
        self.CANNON_PROJ_SPEED = 12
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.base_health = 0
        self.game_over = False
        self.win = False
        self.cursor_pos = [0, 0]
        self.aliens = []
        self.defenses = []
        self.projectiles = []
        self.particles = []
        self.spawn_probability = 0.0
        self.occupied_cells = set()
        
        # Initialize state by calling reset
        self.reset()
        
        # self.validate_implementation() # For self-testing, commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = self.INITIAL_SCORE
        self.base_health = self.MAX_HEALTH
        self.game_over = False
        self.win = False
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        
        self.aliens.clear()
        self.defenses.clear()
        self.projectiles.clear()
        self.particles.clear()
        self.occupied_cells.clear()
        
        self.spawn_probability = self.INITIAL_SPAWN_PROB

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        self.clock.tick(self.FPS)
        self.steps += 1
        reward = 0.0
        
        self._handle_player_action(action)
        
        self._spawn_aliens()
        self._update_defenses()
        self._update_projectiles()
        self._update_aliens()
        self._update_particles()
        
        kill_reward, score_gain = self._handle_collisions()
        reward += kill_reward
        self.score += score_gain

        terminated = False
        if self.base_health <= 0:
            self.game_over = True
            self.win = False
            terminated = True
            reward -= 100
            # sfx: game_over_lose
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100
            # sfx: game_over_win
            
        if self.steps > 0 and self.steps % 100 == 0:
            self.spawn_probability = min(0.1, self.spawn_probability + self.SPAWN_PROB_INCREASE)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_player_action(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_H - 2, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_W - 1, self.cursor_pos[0] + 1)

        cursor_tuple = tuple(self.cursor_pos)
        is_valid_location = cursor_tuple not in self.occupied_cells and self.cursor_pos[1] < self.GRID_H - 1

        if shift_pressed and self.score >= self.CANNON_COST and is_valid_location:
            self.score -= self.CANNON_COST
            pos_x = (self.cursor_pos[0] + 0.5) * self.CELL_SIZE
            pos_y = (self.cursor_pos[1] + 0.5) * self.CELL_SIZE
            self.defenses.append({
                "type": "cannon", "pos": pygame.math.Vector2(pos_x, pos_y),
                "cooldown": 0, "range": self.CANNON_RANGE
            })
            self.occupied_cells.add(cursor_tuple)
            # sfx: deploy_cannon
        elif space_pressed and self.score >= self.TURRET_COST and is_valid_location:
            self.score -= self.TURRET_COST
            pos_x = (self.cursor_pos[0] + 0.5) * self.CELL_SIZE
            pos_y = (self.cursor_pos[1] + 0.5) * self.CELL_SIZE
            self.defenses.append({
                "type": "turret", "pos": pygame.math.Vector2(pos_x, pos_y),
                "cooldown": 0, "range": self.TURRET_RANGE
            })
            self.occupied_cells.add(cursor_tuple)
            # sfx: deploy_turret
            
    def _spawn_aliens(self):
        if self.np_random.random() < self.spawn_probability:
            spawn_x = self.np_random.integers(self.CELL_SIZE, self.WIDTH - self.CELL_SIZE)
            self.aliens.append(pygame.math.Vector2(spawn_x, -10))
            # sfx: alien_spawn
            
    def _update_defenses(self):
        for defense in self.defenses:
            defense["cooldown"] = max(0, defense["cooldown"] - 1)
            if defense["cooldown"] == 0:
                target = None
                min_dist = float('inf')
                for alien in self.aliens:
                    dist = defense["pos"].distance_to(alien)
                    if dist < defense["range"] and dist < min_dist:
                        min_dist = dist
                        target = alien
                
                if target:
                    direction = (target - defense["pos"]).normalize()
                    proj_speed = self.TURRET_PROJ_SPEED if defense["type"] == "turret" else self.CANNON_PROJ_SPEED
                    self.projectiles.append({
                        "pos": defense["pos"].copy(),
                        "vel": direction * proj_speed
                    })
                    defense["cooldown"] = self.TURRET_COOLDOWN if defense["type"] == "turret" else self.CANNON_COOLDOWN
                    # sfx: shoot
                    
    def _update_projectiles(self):
        self.projectiles[:] = [p for p in self.projectiles if 0 <= p["pos"].x < self.WIDTH and 0 <= p["pos"].y < self.HEIGHT]
        for p in self.projectiles:
            p["pos"] += p["vel"]

    def _update_aliens(self):
        aliens_to_keep = []
        base_y = self.HEIGHT - self.CELL_SIZE
        for alien in self.aliens:
            alien.y += self.ALIEN_SPEED
            if alien.y >= base_y:
                self.base_health = max(0, self.base_health - self.ALIEN_DAMAGE)
                self._create_explosion(alien, 15, self.COLOR_BASE_HEALTH)
                # sfx: base_hit
            else:
                aliens_to_keep.append(alien)
        self.aliens = aliens_to_keep
        
    def _handle_collisions(self):
        reward = 0.0
        score_gain = 0
        
        aliens_hit = set()
        projectiles_hit = set()
        
        for i, p in enumerate(self.projectiles):
            for j, a in enumerate(self.aliens):
                if j in aliens_hit: continue
                if p["pos"].distance_to(a) < self.CELL_SIZE:
                    aliens_hit.add(j)
                    projectiles_hit.add(i)
                    reward += 1.0
                    score_gain += 1
                    self._create_explosion(a, 10, self.COLOR_EXPLOSION)
                    # sfx: alien_explode
                    break
        
        if aliens_hit:
            self.aliens = [a for i, a in enumerate(self.aliens) if i not in aliens_hit]
        if projectiles_hit:
            self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_hit]
            
        return reward, score_gain

    def _update_particles(self):
        self.particles[:] = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["radius"] = max(0, p["radius"] * 0.95)

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(), "vel": vel,
                "life": self.np_random.integers(10, 25),
                "radius": self.np_random.random() * 4 + 2,
                "color": color
            })
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "base_health": self.base_health}

    def _render_game(self):
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        base_rect = pygame.Rect(0, self.HEIGHT - self.CELL_SIZE, self.WIDTH, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)

        for defense in self.defenses:
            pos = (int(defense["pos"].x), int(defense["pos"].y))
            if defense["type"] == "turret":
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.CELL_SIZE // 2, self.COLOR_TURRET)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.CELL_SIZE // 2, self.COLOR_TURRET)
            else:
                size = self.CELL_SIZE // 2
                rect = pygame.Rect(pos[0] - size, pos[1] - size, size*2, size*2)
                pygame.draw.rect(self.screen, self.COLOR_CANNON, rect)

        for alien in self.aliens:
            x, y = int(alien.x), int(alien.y)
            points = [(x, y + 5), (x - 5, y - 5), (x + 5, y - 5)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ALIEN)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ALIEN)

        for p in self.projectiles:
            start_pos = (int(p["pos"].x), int(p["pos"].y))
            end_pos = (int(p["pos"].x - p["vel"].x * 0.8), int(p["pos"].y - p["vel"].y * 0.8))
            pygame.draw.aaline(self.screen, self.COLOR_PROJECTILE, start_pos, end_pos, 2)
            
        for p in self.particles:
            pos = (int(p["pos"].x), int(p["pos"].y))
            radius = int(p["radius"])
            if radius > 0:
                alpha = int(255 * (p["life"] / 25))
                color = (*p["color"], alpha)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

        cursor_x = self.cursor_pos[0] * self.CELL_SIZE
        cursor_y = self.cursor_pos[1] * self.CELL_SIZE
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.CELL_SIZE, self.CELL_SIZE)
        
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, (cursor_x, cursor_y))
        pygame.draw.rect(self.screen, (255,255,255), cursor_rect, 1)

    def _render_ui(self):
        health_ratio = self.base_health / self.MAX_HEALTH
        health_bar_width = int(200 * health_ratio)
        pygame.draw.rect(self.screen, (50,0,0), (10, 10, 200, 20))
        if health_bar_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_BASE_HEALTH, (10, 10, health_bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (10, 10, 200, 20), 1)

        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        time_left = max(0, (self.MAX_STEPS - self.steps) // self.FPS)
        time_text = self.font_ui.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH // 2 - time_text.get_width() // 2, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
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
    obs, info = env.reset()
    
    pygame.display.set_caption("Arcade Tower Defense")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        movement, space_pressed, shift_pressed = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_pressed = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_pressed = 1
        
        action = [movement, space_pressed, shift_pressed]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

    env.close()