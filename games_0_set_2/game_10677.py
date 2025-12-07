import gymnasium as gym
import os
import pygame
import math
from collections import deque
import numpy as np
import pygame.gfxdraw
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:48:38.160234
# Source Brief: brief_00677.md
# Brief Index: 677
# """import gymnasium as gym

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a cyberpunk fractal racing shooter.

    The player controls a ship navigating a procedurally generated fractal track,
    dodging and destroying enemy ships. The core mechanics include momentum-based
    movement, a speed boost, and projectiles that branch on impact.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a procedurally generated fractal track in this cyberpunk shooter. "
        "Dodge and destroy enemy ships using branching projectiles and a speed boost."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your ship. "
        "Press space to fire projectiles and hold shift to activate a speed boost."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 30

    # Colors
    COLOR_BG = (10, 0, 20)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_GLOW = (0, 100, 150)
    COLOR_PLAYER_BOOST = (255, 255, 255)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (150, 0, 0)
    COLOR_PLAYER_PROJ = (50, 255, 50)
    COLOR_ENEMY_PROJ = (255, 150, 0)
    COLOR_PARTICLE = (255, 200, 0)
    COLOR_FRACTAL = (80, 20, 120)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH_HIGH = (0, 255, 0)
    COLOR_HEALTH_LOW = (255, 0, 0)

    # Player
    PLAYER_SIZE = 15
    PLAYER_ACCEL = 0.8
    PLAYER_DAMPING = 0.92
    PLAYER_MAX_SPEED = 8.0
    PLAYER_MAX_HEALTH = 100
    PLAYER_SHOOT_COOLDOWN = 6 # frames
    PLAYER_BOOST_SPEED = 14.0
    PLAYER_BOOST_DURATION = 30 # frames
    PLAYER_BOOST_COOLDOWN = 120 # frames

    # Projectiles
    PROJ_SPEED = 12
    PROJ_BRANCH_ANGLE = math.pi / 8 # 22.5 degrees
    PROJ_MAX_GENERATION = 1

    # Enemies
    ENEMY_BASE_SPEED = 2.0
    ENEMY_SPEED_WAVE_INC = 0.1
    ENEMY_BASE_SHOOT_RATE = 120 # frames
    ENEMY_WAVE_COUNT_INC = 2 # every N waves

    # Game
    MAX_STEPS = 5000
    MAX_WAVES = 10

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
        try:
            self.font_small = pygame.font.SysFont("Consolas", 18)
            self.font_large = pygame.font.SysFont("Consolas", 32)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 40)
        
        self.render_mode = render_mode
        self._initialize_state_variables()

    def _initialize_state_variables(self):
        """Initializes all mutable state variables to a default value."""
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_wave = 0
        self.wave_clear_bonus_given = False
        self.unlocked_fractal_patterns = {0}

        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = 0
        self.player_shoot_timer = 0
        self.player_boost_timer = 0
        self.player_boost_cooldown_timer = 0
        
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.fractal_lines = deque()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state_variables()

        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.8)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = self.PLAYER_MAX_HEALTH
        
        self.fractal_lines.clear()
        for i in range(self.SCREEN_HEIGHT // 40 + 2):
            self._add_fractal_segment(self.SCREEN_HEIGHT - i * 40)
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Update Game Logic ---
        reward += self._update_player(movement, space_held, shift_held)
        self._update_projectiles()
        self._update_enemies()
        
        self._update_particles()
        self._update_fractal_track()

        # --- Collision Detection ---
        reward += self._handle_collisions()

        # --- Wave Management ---
        if not self.enemies and self.current_wave <= self.MAX_WAVES:
            if not self.wave_clear_bonus_given:
                reward += 1.0  # Wave clear bonus
                self.wave_clear_bonus_given = True
            self._spawn_wave()
        
        # --- Termination Conditions ---
        if self.player_health <= 0:
            reward = -10.0 # Terminal penalty for dying
            terminated = True
            self.game_over = True
            self._create_explosion(self.player_pos, 100)
        elif self.current_wave > self.MAX_WAVES:
            reward = 100.0 # Terminal bonus for winning
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self, movement, space_held, shift_held):
        # -- Cooldowns --
        if self.player_shoot_timer > 0: self.player_shoot_timer -= 1
        if self.player_boost_timer > 0: self.player_boost_timer -= 1
        if self.player_boost_cooldown_timer > 0: self.player_boost_cooldown_timer -= 1

        # -- Speed Boost --
        max_speed = self.PLAYER_MAX_SPEED
        if shift_held and self.player_boost_cooldown_timer == 0:
            self.player_boost_timer = self.PLAYER_BOOST_DURATION
            self.player_boost_cooldown_timer = self.PLAYER_BOOST_COOLDOWN
            # sfx: boost activate
        
        if self.player_boost_timer > 0:
            max_speed = self.PLAYER_BOOST_SPEED
            # Add boost trail particles
            if self.steps % 2 == 0:
                p_pos = self.player_pos + self.np_random.uniform(-5, 5, 2)
                p_vel = -self.player_vel * 0.1
                self.particles.append([p_pos, p_vel, 15, self.COLOR_PLAYER_BOOST])


        # -- Movement --
        accel_vec = pygame.Vector2(0, 0)
        if movement == 1: accel_vec.y = -1 # Up
        elif movement == 2: accel_vec.y = 1  # Down
        elif movement == 3: accel_vec.x = -1 # Left
        elif movement == 4: accel_vec.x = 1  # Right
        
        if accel_vec.length() > 0:
            accel_vec.scale_to_length(self.PLAYER_ACCEL)
        
        self.player_vel += accel_vec
        self.player_vel *= self.PLAYER_DAMPING
        if self.player_vel.length() > max_speed:
            self.player_vel.scale_to_length(max_speed)
        
        self.player_pos += self.player_vel

        # -- Boundaries --
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE, self.SCREEN_HEIGHT - self.PLAYER_SIZE)

        # -- Shooting --
        if space_held and self.player_shoot_timer == 0:
            # sfx: player shoot
            proj_vel = pygame.Vector2(0, -self.PROJ_SPEED)
            self.player_projectiles.append([self.player_pos.copy(), proj_vel, 0])
            self.player_shoot_timer = self.PLAYER_SHOOT_COOLDOWN
        
        return 0

    def _update_projectiles(self):
        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj[0] += proj[1]
            if not self.screen.get_rect().collidepoint(proj[0]):
                # Branch at screen edges
                if proj[2] < self.PROJ_MAX_GENERATION:
                    self._branch_projectile(proj, is_player=True)
                self.player_projectiles.remove(proj)

        # Enemy projectiles
        for proj in self.enemy_projectiles[:]:
            proj[0] += proj[1]
            if not self.screen.get_rect().collidepoint(proj[0]):
                self.enemy_projectiles.remove(proj)

    def _update_enemies(self):
        wave_speed = self.ENEMY_BASE_SPEED + self.current_wave * self.ENEMY_SPEED_WAVE_INC
        for enemy in self.enemies[:]:
            # Movement
            if enemy['type'] == 'sinusoidal':
                enemy['pos'].x = enemy['origin_x'] + math.sin(self.steps * 0.05 + enemy['phase']) * 100
                enemy['pos'].y += wave_speed * 0.5
            elif enemy['type'] == 'diagonal':
                enemy['pos'] += enemy['vel'] * wave_speed
            
            # Despawn if off-screen
            if enemy['pos'].y > self.SCREEN_HEIGHT + 20:
                self.enemies.remove(enemy)
                continue

            # Shooting
            enemy['shoot_timer'] -= 1
            if enemy['shoot_timer'] <= 0:
                # sfx: enemy shoot
                direction = (self.player_pos - enemy['pos']).normalize()
                self.enemy_projectiles.append([enemy['pos'].copy(), direction * self.PROJ_SPEED * 0.75])
                enemy['shoot_timer'] = self.ENEMY_BASE_SHOOT_RATE - self.current_wave * 5

        return 0

    def _handle_collisions(self):
        reward = 0
        # Player projectiles vs Enemies
        for proj in self.player_projectiles[:]:
            for enemy in self.enemies[:]:
                if proj[0].distance_to(enemy['pos']) < self.PLAYER_SIZE + 5:
                    # sfx: enemy hit/explode
                    self._create_explosion(enemy['pos'], 30)
                    self.enemies.remove(enemy)
                    reward += 0.1 # Reward for destroying an enemy
                    
                    if proj[2] < self.PROJ_MAX_GENERATION:
                        self._branch_projectile(proj, is_player=True)
                    
                    if proj in self.player_projectiles:
                        self.player_projectiles.remove(proj)
                    break 

        # Enemy projectiles vs Player
        for proj in self.enemy_projectiles[:]:
            if proj[0].distance_to(self.player_pos) < self.PLAYER_SIZE:
                # sfx: player hit
                self.enemy_projectiles.remove(proj)
                self.player_health -= 10
                reward -= 0.1 # Penalty for taking damage
                self._create_explosion(self.player_pos, 10, self.COLOR_PLAYER)
                break
        return reward
    
    def _branch_projectile(self, proj, is_player):
        pos, vel, gen = proj
        new_gen = gen + 1

        vel1 = vel.rotate_rad(self.PROJ_BRANCH_ANGLE)
        vel2 = vel.rotate_rad(-self.PROJ_BRANCH_ANGLE)

        if is_player:
            self.player_projectiles.append([pos.copy(), vel1, new_gen])
            self.player_projectiles.append([pos.copy(), vel2, new_gen])

    def _create_explosion(self, pos, num_particles, color=None):
        color = color or self.COLOR_PARTICLE
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            p_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            p_life = self.np_random.integers(15, 40)
            self.particles.append([pos.copy(), p_vel, p_life, color])

    def _update_particles(self):
        for p in self.particles[:]:
            p[0] += p[1] # pos += vel
            p[2] -= 1    # life -= 1
            p[1] *= 0.95 # vel damping
            if p[2] <= 0:
                self.particles.remove(p)

    def _update_fractal_track(self):
        scroll_speed = 1.5
        for line in self.fractal_lines:
            for point in line:
                point.y += scroll_speed
        
        if self.fractal_lines and self.fractal_lines[0][0].y > self.SCREEN_HEIGHT:
            self.fractal_lines.popleft()
            self._add_fractal_segment(self.fractal_lines[-1][0].y - 40)

    def _spawn_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES:
            return

        self.wave_clear_bonus_given = False
        num_enemies = 3 + (self.current_wave - 1) // self.ENEMY_WAVE_COUNT_INC
        
        for i in range(num_enemies):
            spawn_x = self.SCREEN_WIDTH * (i + 1) / (num_enemies + 1)
            spawn_y = self.np_random.uniform(-80, -20)
            enemy_type = self.np_random.choice(['sinusoidal', 'diagonal'])
            
            enemy = {
                'pos': pygame.Vector2(spawn_x, spawn_y),
                'health': 1,
                'type': enemy_type,
                'shoot_timer': self.np_random.integers(60, self.ENEMY_BASE_SHOOT_RATE),
                'origin_x': spawn_x,
                'phase': self.np_random.uniform(0, 2 * math.pi)
            }
            if enemy_type == 'diagonal':
                enemy['vel'] = pygame.Vector2(self.np_random.choice([-1, 1]), 0.5).normalize()
            
            self.enemies.append(enemy)
        
        # Unlock new fractal patterns
        if self.current_wave % 5 == 0:
            self.unlocked_fractal_patterns.add(self.current_wave // 5)

    def _add_fractal_segment(self, y_pos):
        pattern_index = self.np_random.choice(list(self.unlocked_fractal_patterns))
        roughness = 0.4 + pattern_index * 0.1
        
        left_points = self._generate_fractal_line(pygame.Vector2(0, y_pos), pygame.Vector2(self.SCREEN_WIDTH * 0.4, y_pos), 4, roughness)
        right_points = self._generate_fractal_line(pygame.Vector2(self.SCREEN_WIDTH * 0.6, y_pos), pygame.Vector2(self.SCREEN_WIDTH, y_pos), 4, roughness)
        
        self.fractal_lines.append(left_points)
        self.fractal_lines.append(right_points)

    def _generate_fractal_line(self, p1, p2, depth, roughness):
        if depth == 0:
            return [p1, p2]
        
        mid = (p1 + p2) / 2
        length = p1.distance_to(p2)
        offset = self.np_random.uniform(-length * roughness, length * roughness)
        mid.y += offset

        return self._generate_fractal_line(p1, mid, depth-1, roughness) + \
               self._generate_fractal_line(mid, p2, depth-1, roughness)[1:]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render fractal track
        for line in self.fractal_lines:
            pygame.draw.aalines(self.screen, self.COLOR_FRACTAL, False, line)

        # Render projectiles
        for pos, _, _ in self.player_projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PLAYER_PROJ, (int(pos.x), int(pos.y)), 3)
        for pos, _ in self.enemy_projectiles:
            pygame.draw.circle(self.screen, self.COLOR_ENEMY_PROJ, (int(pos.x), int(pos.y)), 4)

        # Render enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            points = [(pos[0], pos[1] - 12), (pos[0] - 10, pos[1] + 8), (pos[0] + 10, pos[1] + 8)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY_GLOW)

        # Render player
        if self.player_health > 0:
            pos = (int(self.player_pos.x), int(self.player_pos.y))
            points = [(pos[0], pos[1] - self.PLAYER_SIZE), (pos[0] - self.PLAYER_SIZE*0.8, pos[1] + self.PLAYER_SIZE*0.8), (pos[0] + self.PLAYER_SIZE*0.8, pos[1] + self.PLAYER_SIZE*0.8)]
            
            # Glow effect
            glow_size = self.PLAYER_SIZE * (2.0 + math.sin(self.steps * 0.1) * 0.2)
            if self.player_boost_timer > 0:
                glow_size *= 1.5
            
            glow_points = [(pos[0], pos[1] - glow_size), (pos[0] - glow_size*0.8, pos[1] + glow_size*0.8), (pos[0] + glow_size*0.8, pos[1] + glow_size*0.8)]
            pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.aapolygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)

            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)

        # Render particles
        for pos, _, life, color in self.particles:
            alpha = max(0, min(255, int(255 * (life / 40.0))))
            size = max(1, int(life/10))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*color, alpha), (size, size), size)
            self.screen.blit(temp_surf, (int(pos.x - size), int(pos.y - size)))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_str = f"WAVE: {self.current_wave}/{self.MAX_WAVES}" if self.current_wave <= self.MAX_WAVES else "LEVEL CLEAR"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))
        
        # Health Bar
        health_pct = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        health_color = (
            self.COLOR_HEALTH_LOW[0] + (self.COLOR_HEALTH_HIGH[0] - self.COLOR_HEALTH_LOW[0]) * health_pct,
            self.COLOR_HEALTH_LOW[1] + (self.COLOR_HEALTH_HIGH[1] - self.COLOR_HEALTH_LOW[1]) * health_pct,
            self.COLOR_HEALTH_LOW[2] + (self.COLOR_HEALTH_HIGH[2] - self.COLOR_HEALTH_LOW[2]) * health_pct,
        )
        bar_width = 200
        bar_height = 15
        pygame.draw.rect(self.screen, (50,50,50), (10, self.SCREEN_HEIGHT - bar_height - 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, health_color, (10, self.SCREEN_HEIGHT - bar_height - 10, bar_width * health_pct, bar_height))

        # Boost Cooldown
        boost_ready = self.player_boost_cooldown_timer == 0
        boost_color = self.COLOR_PLAYER_BOOST if boost_ready else (80, 80, 80)
        boost_text = self.font_small.render("BOOST", True, boost_color)
        self.screen.blit(boost_text, (bar_width + 20, self.SCREEN_HEIGHT - bar_height - 8))

        if self.game_over:
            msg = "LEVEL CLEAR" if self.current_wave > self.MAX_WAVES else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "player_health": self.player_health
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    
    pygame.display.set_caption("Fractal Fury")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not terminated and not truncated:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Transpose back for Pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.TARGET_FPS)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait a bit before closing
            pygame.time.wait(2000)

    env.close()