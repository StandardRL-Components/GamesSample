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
        "Controls: Use arrow keys to move. Press Space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down space shooter. Survive five waves of increasingly difficult alien attacks."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 10000
    TOTAL_WAVES = 5

    # Initialize pygame here to use pygame.Color
    pygame.init()

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_STAR = (200, 200, 220)
    COLOR_PLAYER = pygame.Color(0, 255, 150)
    COLOR_PLAYER_MID_HP = pygame.Color(255, 255, 0)
    COLOR_PLAYER_LOW_HP = pygame.Color(255, 50, 50)
    COLOR_ALIEN = (255, 60, 100)
    COLOR_PLAYER_PROJ = (100, 255, 255)
    COLOR_ENEMY_PROJ = (255, 200, 0)
    COLOR_PARTICLE = (255, 255, 255)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_LIVES = (0, 255, 150)

    # Player settings
    PLAYER_SIZE = 12
    PLAYER_MAX_SPEED = 6
    PLAYER_ACCELERATION = 0.6
    PLAYER_FRICTION = 0.92
    PLAYER_MAX_HEALTH = 3
    PLAYER_LIVES = 3
    PLAYER_FIRE_COOLDOWN = 6  # frames

    # Alien settings
    ALIEN_BASE_SIZE = 14
    ALIEN_BASE_SPEED = 1.0
    ALIEN_BASE_FIRE_RATE = 0.01 # projectiles per frame

    # Projectile settings
    PROJ_SPEED = 10
    PROJ_SIZE = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.player_pos = None
        self.player_vel = None
        self.player_health = 0
        self.player_lives = 0
        self.player_fire_timer = 0
        
        self.aliens = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.stars = []
        
        self.steps = 0
        self.score = 0
        self.wave_number = 0
        self.game_over = False
        self.win = False

        # reset() is called here, which calls _get_observation(), which needs the env to be initialized.
        # No need to call it again, __init__ is the setup.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_lives = self.PLAYER_LIVES
        self.player_fire_timer = 0
        
        self.aliens = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.wave_number = 0
        self.game_over = False
        self.win = False

        self.stars = [
            (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT), self.np_random.integers(1, 3))
            for _ in range(150)
        ]

        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        reward = -0.01  # Time penalty

        if not self.game_over:
            # --- Player Logic ---
            self._handle_player_input(movement, space_held)
            
            # --- Update Game Objects ---
            self._update_player()
            self._update_projectiles()
            self._update_aliens()
            self._update_particles()
            
            # --- Handle Collisions & Rewards ---
            reward += self._handle_collisions()
            
            # --- Wave Logic ---
            if not self.aliens and self.wave_number <= self.TOTAL_WAVES:
                reward += 10 # Wave clear bonus
                self._start_next_wave()

            # --- Defensive play penalty ---
            if len(self.aliens) > 0 and not space_held:
                reward -= 0.2

        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated and not self.win:
            reward -= 100 # Penalty for losing
        elif terminated and self.win:
            reward += 100 # Bonus for winning

        self.steps += 1
        
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_player_input(self, movement, space_held):
        # Movement
        if movement == 1: self.player_vel.y -= self.PLAYER_ACCELERATION
        if movement == 2: self.player_vel.y += self.PLAYER_ACCELERATION
        if movement == 3: self.player_vel.x -= self.PLAYER_ACCELERATION
        if movement == 4: self.player_vel.x += self.PLAYER_ACCELERATION

        # Firing
        if space_held and self.player_fire_timer <= 0:
            # sfx: player_shoot.wav
            self.player_projectiles.append(pygame.math.Vector2(self.player_pos.x, self.player_pos.y - self.PLAYER_SIZE))
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN

    def _update_player(self):
        # Apply friction and limit speed
        self.player_vel *= self.PLAYER_FRICTION
        if self.player_vel.length() > self.PLAYER_MAX_SPEED:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)
        
        self.player_pos += self.player_vel
        
        # Boundary checks
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE, self.SCREEN_HEIGHT - self.PLAYER_SIZE)

        if self.player_fire_timer > 0:
            self.player_fire_timer -= 1

    def _update_projectiles(self):
        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj.y -= self.PROJ_SPEED
            if proj.y < 0:
                self.player_projectiles.remove(proj)
        
        # Enemy projectiles
        for proj in self.enemy_projectiles[:]:
            proj.y += self.PROJ_SPEED / 2
            if proj.y > self.SCREEN_HEIGHT:
                self.enemy_projectiles.remove(proj)

    def _update_aliens(self):
        for alien in self.aliens[:]:
            speed = self.ALIEN_BASE_SPEED + (self.wave_number - 1) * 0.2
            
            # Movement patterns
            if alien['pattern'] == 'sine':
                alien['pos'].y += speed / 2
                alien['pos'].x = alien['center_x'] + math.sin(alien['pos'].y / 50) * 100
            elif alien['pattern'] == 'diag':
                alien['pos'] += alien['vel'] * speed
                if alien['pos'].x < self.ALIEN_BASE_SIZE or alien['pos'].x > self.SCREEN_WIDTH - self.ALIEN_BASE_SIZE:
                    alien['vel'].x *= -1
            else: # Default 'down'
                alien['pos'].y += speed

            # Firing
            fire_rate = self.ALIEN_BASE_FIRE_RATE + (self.wave_number - 1) * 0.005
            if self.np_random.random() < fire_rate:
                # sfx: enemy_shoot.wav
                self.enemy_projectiles.append(pygame.math.Vector2(alien['pos']))

            # Remove if off-screen
            if alien['pos'].y > self.SCREEN_HEIGHT + self.ALIEN_BASE_SIZE:
                self.aliens.remove(alien)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Aliens
        for proj in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                if proj.distance_to(alien['pos']) < self.ALIEN_BASE_SIZE + self.PROJ_SIZE:
                    # sfx: explosion.wav
                    self._create_explosion(alien['pos'])
                    self.aliens.remove(alien)
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    self.score += 1
                    reward += 1
                    break
        
        # Enemy projectiles vs Player
        for proj in self.enemy_projectiles[:]:
            if proj.distance_to(self.player_pos) < self.PLAYER_SIZE + self.PROJ_SIZE:
                # sfx: player_hit.wav
                self.enemy_projectiles.remove(proj)
                self.player_health -= 1
                if self.player_health <= 0:
                    self._player_death()
                break
        
        return reward

    def _player_death(self):
        self.player_lives -= 1
        self._create_explosion(self.player_pos, count=50)
        if self.player_lives > 0:
            self.player_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50)
            self.player_vel = pygame.math.Vector2(0, 0)
            self.player_health = self.PLAYER_MAX_HEALTH
        else:
            self.game_over = True
            self.win = False

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.TOTAL_WAVES:
            self.game_over = True
            self.win = True
            return

        num_aliens = 5 + (self.wave_number - 1) * 5
        for i in range(num_aliens):
            pattern = 'down'
            if self.wave_number >= 2: pattern = 'sine' if i % 2 == 0 else 'down'
            if self.wave_number >= 4: pattern = 'diag' if i % 3 == 0 else pattern
            
            x_pos = (self.SCREEN_WIDTH / (num_aliens + 1)) * (i + 1)
            y_pos = self.np_random.integers(-150, -50)

            alien = {
                'pos': pygame.math.Vector2(x_pos, y_pos),
                'pattern': pattern,
                'center_x': x_pos,
                'vel': pygame.math.Vector2(self.np_random.choice([-1, 1]), 0.5).normalize()
            }
            self.aliens.append(alien)
    
    def _check_termination(self):
        return self.game_over

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "lives": self.player_lives,
            "health": self.player_health
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, self.COLOR_STAR, (x, y, size, size))
        
        # Projectiles
        for proj in self.player_projectiles:
            self._draw_glowing_circle(self.screen, self.COLOR_PLAYER_PROJ, proj, self.PROJ_SIZE, 3)
        for proj in self.enemy_projectiles:
            self._draw_glowing_circle(self.screen, self.COLOR_ENEMY_PROJ, proj, self.PROJ_SIZE, 3)
        
        # Aliens
        for alien in self.aliens:
            self._draw_glowing_polygon(self.screen, self.COLOR_ALIEN, self._get_alien_points(alien['pos']), 3)

        # Player
        if self.player_lives > 0:
            health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
            player_color = self.COLOR_PLAYER_LOW_HP.lerp(self.COLOR_PLAYER_MID_HP, (health_ratio * 2)) if health_ratio < 0.5 else self.COLOR_PLAYER_MID_HP.lerp(self.COLOR_PLAYER, (health_ratio - 0.5) * 2)
            self._draw_glowing_polygon(self.screen, player_color, self._get_player_points(), 5)

        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), color)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_text = self.font_ui.render(f"WAVE: {min(self.wave_number, self.TOTAL_WAVES)} / {self.TOTAL_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH // 2 - wave_text.get_width() // 2, 10))

        # Lives
        for i in range(self.player_lives -1):
             points = self._get_player_points(size_mod=0.7, offset=pygame.math.Vector2(self.SCREEN_WIDTH - 30 - i * 25, 22))
             pygame.draw.polygon(self.screen, self.COLOR_UI_LIVES, points)

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_PLAYER if self.win else self.COLOR_ALIEN
            end_text = self.font_game_over.render(msg, True, color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH // 2 - end_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - end_text.get_height() // 2))

    def _get_player_points(self, size_mod=1.0, offset=None):
        pos = offset if offset is not None else self.player_pos
        s = self.PLAYER_SIZE * size_mod
        return [
            (pos.x, pos.y - s),
            (pos.x - s * 0.8, pos.y + s * 0.8),
            (pos.x + s * 0.8, pos.y + s * 0.8)
        ]

    def _get_alien_points(self, pos):
        s = self.ALIEN_BASE_SIZE
        return [
            (pos.x, pos.y - s),
            (pos.x - s, pos.y),
            (pos.x, pos.y + s),
            (pos.x + s, pos.y)
        ]

    def _create_explosion(self, position, count=30):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(15, 40)
            self.particles.append({
                'pos': pygame.math.Vector2(position),
                'vel': vel,
                'life': life,
                'max_life': life,
                'size': self.np_random.integers(1, 4),
                'color': random.choice([ (255,255,255), (255,200,0), (255,100,0) ])
            })

    def _draw_glowing_polygon(self, surface, color, points, glow_size):
        # pygame.Color can be indexed like a tuple, so this works fine
        for i in range(glow_size, 0, -1):
            alpha = 150 * (1 - (i / glow_size))
            glow_color = (color[0], color[1], color[2], alpha)
            scaled_points = self._scale_points(points, 1 + i * 0.1)
            pygame.gfxdraw.filled_polygon(surface, [tuple(map(int, p)) for p in scaled_points], glow_color)
        pygame.gfxdraw.filled_polygon(surface, [tuple(map(int, p)) for p in points], color)
        pygame.gfxdraw.aapolygon(surface, [tuple(map(int, p)) for p in points], color)

    def _draw_glowing_circle(self, surface, color, pos, radius, glow_size):
        pos_int = (int(pos.x), int(pos.y))
        for i in range(glow_size, 0, -1):
            alpha = 150 * (1 - (i / glow_size))
            glow_color = (color[0], color[1], color[2], alpha)
            pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], radius + i, glow_color)
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], radius, color)
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], radius, color)

    def _scale_points(self, points, scale):
        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)
        return [
            (center_x + (p[0] - center_x) * scale, center_y + (p[1] - center_y) * scale)
            for p in points
        ]

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will open a window for human interaction
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Space Wave Survivor")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
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
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Convert the observation (H, W, C) to a Pygame surface (W, H)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS
        
    env.close()