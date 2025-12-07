
# Generated: 2025-08-28T04:48:30.788865
# Source Brief: brief_05376.md
# Brief Index: 5376

        
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

    user_guide = (
        "Controls: Arrow keys to move the placement cursor. Press space to build a tower."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing defensive towers."
    )

    auto_advance = True

    # --- Helper Classes ---
    class Enemy:
        def __init__(self, speed, path):
            self.path = path
            self.path_index = 0
            self.pos = np.array(self.path[0], dtype=float)
            self.health = 100
            self.speed = speed
            self.radius = 8
            self.value = 1 # Reward for destroying

        def move(self):
            if self.path_index >= len(self.path) - 1:
                return True # Reached the end

            target_pos = np.array(self.path[self.path_index + 1], dtype=float)
            direction = target_pos - self.pos
            distance = np.linalg.norm(direction)

            if distance < self.speed:
                self.pos = target_pos
                self.path_index += 1
            else:
                self.pos += (direction / distance) * self.speed
            return False

    class Tower:
        def __init__(self, grid_pos, cell_size):
            self.grid_pos = grid_pos
            self.pos = (
                (grid_pos[0] + 0.5) * cell_size,
                (grid_pos[1] + 0.5) * cell_size
            )
            self.range = 100
            self.fire_rate = 20 # Lower is faster
            self.cooldown = 0
            self.size = 15

        def update(self, enemies, projectiles, cell_size):
            if self.cooldown > 0:
                self.cooldown -= 1
                return

            target = self.find_target(enemies)
            if target:
                self.fire(target, projectiles, cell_size)
                self.cooldown = self.fire_rate

        def find_target(self, enemies):
            closest_enemy = None
            min_dist = self.range
            for enemy in enemies:
                dist = np.linalg.norm(np.array(self.pos) - enemy.pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_enemy = enemy
            return closest_enemy

        def fire(self, target, projectiles, cell_size):
            # SFX: Pew!
            projectiles.append(GameEnv.Projectile(self.pos, target))

    class Projectile:
        def __init__(self, start_pos, target_enemy):
            self.pos = np.array(start_pos, dtype=float)
            direction = target_enemy.pos - self.pos
            distance = np.linalg.norm(direction)
            self.velocity = (direction / distance) * 15 if distance > 0 else np.array([0,0])
            self.radius = 3
            self.damage = 35

        def move(self):
            self.pos += self.velocity

    class Particle:
        def __init__(self, pos, color, min_vel=-2, max_vel=2, gravity=0.1, lifetime=20):
            self.pos = np.array(pos, dtype=float)
            self.vel = np.array([random.uniform(min_vel, max_vel), random.uniform(min_vel, max_vel)])
            self.color = color
            self.lifetime = lifetime + random.randint(-5, 5)
            self.gravity = gravity

        def update(self):
            self.pos += self.vel
            self.vel[1] += self.gravity
            self.lifetime -= 1
            return self.lifetime <= 0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 16, 10
        self.CELL_SIZE = self.WIDTH // self.GRID_COLS

        # Colors
        self.COLOR_BG = (15, 18, 28)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_PATH = (45, 50, 70)
        self.COLOR_BASE = (30, 180, 100)
        self.COLOR_BASE_DMG = (200, 50, 50)
        self.COLOR_ENEMY = (220, 40, 40)
        self.COLOR_TOWER = (60, 150, 255)
        self.COLOR_PROJECTILE = (255, 220, 100)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_UI_BG = (25, 30, 45, 200)

        # Game parameters
        self.MAX_BASE_HEALTH = 50
        self.MAX_WAVES = 5
        self.MAX_STEPS = 2000

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
        self.font_s = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.base_health = 0
        self.current_wave = 0
        self.enemies_to_spawn = 0
        self.spawn_timer = 0
        self.cursor_pos = [0, 0]
        self.space_pressed_last_frame = False
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self._define_path()
        self.reset()
        
        self.validate_implementation()

    def _define_path(self):
        self.path = [
            (-self.CELL_SIZE, 3.5 * self.CELL_SIZE),
            (2.5 * self.CELL_SIZE, 3.5 * self.CELL_SIZE),
            (2.5 * self.CELL_SIZE, 7.5 * self.CELL_SIZE),
            (10.5 * self.CELL_SIZE, 7.5 * self.CELL_SIZE),
            (10.5 * self.CELL_SIZE, 1.5 * self.CELL_SIZE),
            (13.5 * self.CELL_SIZE, 1.5 * self.CELL_SIZE),
            (13.5 * self.CELL_SIZE, 4.5 * self.CELL_SIZE),
            (self.WIDTH + self.CELL_SIZE, 4.5 * self.CELL_SIZE)
        ]
        self.base_pos = (15, 4) # Grid coordinates
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.base_health = self.MAX_BASE_HEALTH
        
        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.space_pressed_last_frame = False
        
        self.current_wave = 0
        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES:
            self.game_won = True
            return

        self.enemies_to_spawn = 2 + self.current_wave
        self.spawn_timer = 0
        self.enemy_speed = 1.0 + (self.current_wave - 1) * 0.2

    def step(self, action):
        self.step_reward = -0.01 # Small penalty for each step
        
        if not self.game_over:
            self._handle_input(action)
            
            # --- Game Logic Updates ---
            self._update_spawner()
            self._update_enemies()
            self._update_towers()
            self._update_projectiles()
            self._handle_collisions()
            self._update_particles()
            
            self._check_wave_status()
        
        self.steps += 1
        terminated = self._check_termination()
        
        reward = self.step_reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # Place tower
        if space_held and not self.space_pressed_last_frame:
            is_valid_spot = True
            # Cannot build on path or on other towers
            if self.cursor_pos[0] == self.base_pos[0] and self.cursor_pos[1] == self.base_pos[1]:
                is_valid_spot = False
            for tower in self.towers:
                if tower.grid_pos[0] == self.cursor_pos[0] and tower.grid_pos[1] == self.cursor_pos[1]:
                    is_valid_spot = False
                    break
            
            if is_valid_spot:
                # SFX: Build
                self.towers.append(self.Tower(self.cursor_pos.copy(), self.CELL_SIZE))

        self.space_pressed_last_frame = space_held
        
    def _update_spawner(self):
        if self.enemies_to_spawn > 0:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self.enemies.append(self.Enemy(self.enemy_speed, self.path))
                self.enemies_to_spawn -= 1
                self.spawn_timer = 45 # Delay between spawns

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            if enemy.move():
                self.base_health -= 10
                # SFX: Base Damage
                self._create_particles(enemy.pos, self.COLOR_BASE_DMG, 20)
                self.enemies.remove(enemy)
                if self.base_health < 0: self.base_health = 0

    def _update_towers(self):
        for tower in self.towers:
            tower.update(self.enemies, self.projectiles, self.CELL_SIZE)

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p.move()
            if not (0 <= p.pos[0] < self.WIDTH and 0 <= p.pos[1] < self.HEIGHT):
                self.projectiles.remove(p)

    def _handle_collisions(self):
        for p in self.projectiles[:]:
            for e in self.enemies[:]:
                if np.linalg.norm(p.pos - e.pos) < e.radius + p.radius:
                    # SFX: Hit
                    e.health -= p.damage
                    self.step_reward += 0.1
                    self._create_particles(p.pos, self.COLOR_PROJECTILE, 3, lifetime=5)
                    if p in self.projectiles: self.projectiles.remove(p)

                    if e.health <= 0:
                        # SFX: Explosion
                        self._create_particles(e.pos, self.COLOR_ENEMY, 30)
                        self.score += e.value
                        self.step_reward += e.value
                        if e in self.enemies: self.enemies.remove(e)
                    break

    def _check_wave_status(self):
        if not self.game_won and self.enemies_to_spawn == 0 and not self.enemies:
            self.step_reward += 10 # Wave survived reward
            self.score += 10
            self._start_next_wave()
            
    def _update_particles(self):
        self.particles = [p for p in self.particles if not p.update()]

    def _create_particles(self, pos, color, count, **kwargs):
        for _ in range(count):
            self.particles.append(self.Particle(pos, color, **kwargs))

    def _check_termination(self):
        if self.game_over:
            return True
            
        if self.base_health <= 0:
            self.game_over = True
            self.step_reward -= 50
            return True
        if self.game_won:
            self.game_over = True
            self.step_reward += 50
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
            
        return False
        
    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "base_health": self.base_health,
        }
        
    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_and_path()
        self._render_base()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        if self.game_over:
            self._render_game_over()

    def _render_grid_and_path(self):
        # Draw path
        if len(self.path) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, width=self.CELL_SIZE)
            for p in self.path:
                pygame.draw.circle(self.screen, self.COLOR_PATH, p, self.CELL_SIZE // 2)

        # Draw grid lines
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
    def _render_base(self):
        base_rect = pygame.Rect(
            self.base_pos[0] * self.CELL_SIZE,
            self.base_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        pygame.draw.rect(self.screen, tuple(c*0.7 for c in self.COLOR_BASE), base_rect, 3)

    def _render_towers(self):
        for tower in self.towers:
            pos_int = (int(tower.pos[0]), int(tower.pos[1]))
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], tower.size + 4, (*self.COLOR_TOWER, 50))
            # Main body
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], tower.size, self.COLOR_TOWER)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], tower.size, tuple(c*0.8 for c in self.COLOR_TOWER))

    def _render_enemies(self):
        for enemy in self.enemies:
            pos_int = (int(enemy.pos[0]), int(enemy.pos[1]))
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], enemy.radius + 3, (*self.COLOR_ENEMY, 80))
            # Main body
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], enemy.radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], enemy.radius, tuple(c*0.8 for c in self.COLOR_ENEMY))

    def _render_projectiles(self):
        for p in self.projectiles:
            pos_int = (int(p.pos[0]), int(p.pos[1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], p.radius + 2, (*self.COLOR_PROJECTILE, 100))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], p.radius, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], p.radius, self.COLOR_PROJECTILE)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p.lifetime / 20))))
            color = (*p.color, alpha)
            pos_int = (int(p.pos[0]), int(p.pos[1]))
            pygame.draw.circle(self.screen, color, pos_int, 2)

    def _render_cursor(self):
        if self.game_over: return
        
        rect = pygame.Rect(
            self.cursor_pos[0] * self.CELL_SIZE,
            self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        # Create a temporary surface for transparency
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        s.fill((*self.COLOR_CURSOR, 50))
        pygame.draw.rect(s, (*self.COLOR_CURSOR, 150), s.get_rect(), 2)
        self.screen.blit(s, rect.topleft)
        
    def _render_ui(self):
        # Health Bar
        health_ratio = self.base_health / self.MAX_BASE_HEALTH
        bar_width = 150
        bar_rect = pygame.Rect(10, 30, bar_width * health_ratio, 15)
        bar_bg_rect = pygame.Rect(10, 30, bar_width, 15)
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, bar_bg_rect)
        pygame.draw.rect(self.screen, self.COLOR_BASE, bar_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, bar_bg_rect, 1)

        # Text
        wave_text = self.font_s.render(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        score_text = self.font_s.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        health_text = self.font_s.render(f"BASE HEALTH", True, self.COLOR_TEXT)
        
        self.screen.blit(health_text, (10, 10))
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 30))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        message = "YOU WIN!" if self.game_won else "GAME OVER"
        text = self.font_l.render(message, True, self.COLOR_TEXT)
        text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text, text_rect)
        
    def validate_implementation(self):
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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="human") # Change to human for direct play
    obs, info = env.reset()
    
    # Override screen and render_mode for direct play
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    env.metadata["render_modes"].append("human")
    env.render_mode = "human"

    done = False
    total_reward = 0
    
    # Key mapping for human play
    key_map = {
        pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4,
    }

    while not done:
        # Construct action from keyboard input
        movement = 0
        space = 0
        shift = 0
        
        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render to the screen
        env._render_all()
        pygame.display.flip()
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(30) # 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()