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

    user_guide = (
        "Controls: ↑↓←→ to move. Hold Shift to activate your shield. Press Space to fire your weapon."
    )

    game_description = (
        "Survive waves of descending aliens in this top-down arcade shooter. Last for 60 seconds to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = self.FPS * 60  # 60 seconds

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_SHIELD = (100, 150, 255, 100)
        self.COLOR_ENEMY_A = (255, 80, 80)
        self.COLOR_PROJECTILE = (255, 255, 100)
        self.COLOR_EXPLOSION = (255, 180, 50)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)
        self.COLOR_HEALTH_BAR = (0, 200, 0)

        # Player properties
        self.PLAYER_SPEED = 4
        self.PLAYER_SIZE = 12
        self.PLAYER_MAX_HEALTH = 100

        # Projectile properties
        self.PROJECTILE_SPEED = 8
        self.SHOOT_COOLDOWN = 15  # frames

        # Shield properties
        self.SHIELD_DURATION = 120  # frames (2 seconds)
        self.SHIELD_COOLDOWN = 300  # frames (5 seconds)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Internal state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = pygame.Vector2(0, 0)
        self.player_health = 0
        self.aliens = []
        self.projectiles = []
        self.particles = []
        self.stars = []

        self.shoot_cooldown_timer = 0
        self.shield_duration_timer = 0
        self.shield_cooldown_timer = 0

        self.prev_space_held = False
        self.prev_shift_held = False

        self.spawn_chance_per_frame = 0.0
        self.alien_base_speed = 0.0

        self.np_random = None

        # self.reset() is called by the wrapper, no need to call it here.
        # self.validate_implementation() # This is for testing and not needed in the final env

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.player_health = self.PLAYER_MAX_HEALTH

        self.aliens = []
        self.projectiles = []
        self.particles = []

        self.shoot_cooldown_timer = 0
        self.shield_duration_timer = 0
        self.shield_cooldown_timer = 0

        self.prev_space_held = False
        self.prev_shift_held = False

        # Difficulty scaling reset
        self.spawn_chance_per_frame = (0.5 / self.FPS)
        self.alien_base_speed = 1.0

        # Create a static starfield
        if not self.stars:
            for _ in range(150):
                self.stars.append(
                    (
                        self.np_random.integers(0, self.WIDTH),
                        self.np_random.integers(0, self.HEIGHT),
                        self.np_random.integers(1, 3)  # star size
                    )
                )

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(self.FPS)
        reward = -0.01  # Time penalty to encourage efficiency

        # 1. Process Actions
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)

        # 2. Update Game State
        self._update_player()
        self._update_projectiles()
        self._update_aliens()
        self._update_particles()
        self._update_cooldowns()
        self._spawn_aliens()

        # 3. Handle Collisions and calculate rewards
        reward += self._handle_collisions()

        # 4. Update Difficulty
        self._update_difficulty()

        # 5. Check Termination Conditions
        self.steps += 1
        terminated = False
        truncated = False
        if self.player_health <= 0:
            terminated = True
            reward = -100.0  # Large penalty for dying
            self._create_explosion(self.player_pos, 50, self.COLOR_PLAYER)
            # sfx: player_explosion.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Game ends successfully
            reward += 100.0  # Large reward for survival

        self.game_over = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Movement
        if movement == 1: self.player_pos.y -= self.PLAYER_SPEED
        if movement == 2: self.player_pos.y += self.PLAYER_SPEED
        if movement == 3: self.player_pos.x -= self.PLAYER_SPEED
        if movement == 4: self.player_pos.x += self.PLAYER_SPEED

        # Shooting (on key press, not hold)
        if space_held and not self.prev_space_held and self.shoot_cooldown_timer <= 0:
            self.projectiles.append(pygame.Vector2(self.player_pos))
            self.shoot_cooldown_timer = self.SHOOT_COOLDOWN
            # sfx: laser_shoot.wav

        # Shield (on key press, not hold)
        if shift_held and not self.prev_shift_held and self.shield_cooldown_timer <= 0:
            self.shield_duration_timer = self.SHIELD_DURATION
            self.shield_cooldown_timer = self.SHIELD_COOLDOWN + self.SHIELD_DURATION
            # sfx: shield_activate.wav

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_player(self):
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.HEIGHT)

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p.y -= self.PROJECTILE_SPEED
            if p.y < 0:
                self.projectiles.remove(p)

    def _update_aliens(self):
        for alien in self.aliens[:]:
            alien['pos'] += alien['vel']
            if alien['pos'].y > self.HEIGHT + alien['size']:
                self.aliens.remove(alien)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _update_cooldowns(self):
        if self.shoot_cooldown_timer > 0: self.shoot_cooldown_timer -= 1
        if self.shield_duration_timer > 0: self.shield_duration_timer -= 1
        if self.shield_cooldown_timer > 0: self.shield_cooldown_timer -= 1

    def _spawn_aliens(self):
        if self.np_random.random() < self.spawn_chance_per_frame:
            size = self.np_random.integers(10, 21)
            pos = pygame.Vector2(self.np_random.integers(size, self.WIDTH - size), -size)
            angle = self.np_random.uniform(math.pi * 0.4, math.pi * 0.6)  # Descend mostly downwards
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.alien_base_speed

            self.aliens.append({'pos': pos, 'vel': vel, 'size': size, 'health': size // 5})

    def _handle_collisions(self):
        reward = 0

        # Projectiles vs Aliens
        for proj in self.projectiles[:]:
            for alien in self.aliens[:]:
                dist = proj.distance_to(alien['pos'])
                if dist < alien['size']:
                    self.projectiles.remove(proj)
                    alien['health'] -= 1
                    # sfx: alien_hit.wav
                    if alien['health'] <= 0:
                        self.aliens.remove(alien)
                        self.score += 1
                        reward += 1
                        self._create_explosion(alien['pos'], alien['size'], self.COLOR_EXPLOSION)
                        # sfx: alien_explosion.wav
                    break  # projectile can only hit one alien

        # Player vs Aliens
        is_shielded = self.shield_duration_timer > 0
        if not is_shielded:
            player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2,
                                      self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)
            for alien in self.aliens[:]:
                dist = self.player_pos.distance_to(alien['pos'])
                if dist < self.PLAYER_SIZE / 2 + alien['size']: # More accurate collision
                    self.aliens.remove(alien)
                    self.player_health -= alien['size']  # More damage from bigger aliens
                    self.player_health = max(0, self.player_health)
                    reward -= 5  # Penalty for taking damage
                    self._create_explosion(self.player_pos, 15, self.COLOR_EXPLOSION)
                    # sfx: player_hit.wav

        return reward

    def _update_difficulty(self):
        # Increase spawn rate over time
        self.spawn_chance_per_frame += (0.001 / self.FPS) / self.FPS
        # Increase alien speed every 10 seconds
        if self.steps > 0 and self.steps % (self.FPS * 10) == 0:
            self.alien_base_speed += 0.1

    def _create_explosion(self, pos, size, color):
        num_particles = int(size)
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'lifetime': lifetime, 'color': color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)

        # Render static stars
        for x, y, size in self.stars:
            self.screen.set_at((x, y), (80, 80, 100))

        # Render aliens
        for alien in self.aliens:
            pos = (int(alien['pos'].x), int(alien['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], alien['size'], self.COLOR_ENEMY_A)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], alien['size'], self.COLOR_ENEMY_A)

        # Render projectiles
        for p in self.projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, (int(p.x) - 2, int(p.y), 4, 10))

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 30.0))
            color = (*p['color'], alpha)
            pos = (int(p['pos'].x), int(p['pos'].y))
            radius = int(p['lifetime'] * 0.3)
            if radius > 0:
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, color)
                self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius))


        # Render player
        if self.player_health > 0:
            # Shield effect
            if self.shield_duration_timer > 0:
                shield_radius = int(self.PLAYER_SIZE * 2)
                alpha = 100
                if self.shield_duration_timer < 30:  # Flicker when ending
                    alpha = int(100 * (self.shield_duration_timer % 10 / 10))

                temp_surf = pygame.Surface((shield_radius * 2, shield_radius * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, shield_radius, shield_radius, shield_radius,
                                              (*self.COLOR_SHIELD[:3], alpha))
                self.screen.blit(temp_surf,
                                 (int(self.player_pos.x - shield_radius), int(self.player_pos.y - shield_radius)))

            # Player ship (triangle)
            p1 = (self.player_pos.x, self.player_pos.y - self.PLAYER_SIZE)
            p2 = (self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y + self.PLAYER_SIZE / 2)
            p3 = (self.player_pos.x + self.PLAYER_SIZE / 2, self.player_pos.y + self.PLAYER_SIZE / 2)
            pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)

        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = self.font_small.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI)
        time_rect = time_text.get_rect(centerx=self.WIDTH / 2, top=10)
        self.screen.blit(time_text, time_rect)

        # Health Bar
        health_pct = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        bar_width = 150
        bar_height = 15
        bar_x = self.WIDTH - bar_width - 10
        bar_y = 15
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, int(bar_width * health_pct), bar_height))

        # Shield Cooldown Indicator
        shield_cd_pct = 1.0
        if self.shield_cooldown_timer > 0:
            shield_cd_pct = 1.0 - (self.shield_cooldown_timer / (self.SHIELD_COOLDOWN + self.SHIELD_DURATION))

        cd_bar_width = 150
        cd_bar_height = 5
        cd_bar_x = self.WIDTH - cd_bar_width - 10
        cd_bar_y = bar_y + bar_height + 5
        pygame.draw.rect(self.screen, (40, 40, 60), (cd_bar_x, cd_bar_y, cd_bar_width, cd_bar_height))
        if shield_cd_pct >= 1.0:
            pygame.draw.rect(self.screen, self.COLOR_SHIELD,
                             (cd_bar_x, cd_bar_y, int(cd_bar_width * shield_cd_pct), cd_bar_height))
        else:
            pygame.draw.rect(self.screen, (80, 80, 100),
                             (cd_bar_x, cd_bar_y, int(cd_bar_width * shield_cd_pct), cd_bar_height))

        # Game Over / Win Text
        if self.game_over:
            if self.player_health <= 0:
                msg = "GAME OVER"
            else:
                msg = "YOU SURVIVED!"
            end_text = self.font_large.render(msg, True, self.COLOR_UI)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "aliens_on_screen": len(self.aliens)
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset(seed=42)
    done = False

    # Pygame setup for human play
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Alien Survival")
    clock = pygame.time.Clock()

    total_reward = 0

    while not done:
        # Map keyboard inputs to actions
        keys = pygame.key.get_pressed()
        movement = 0  # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

    # Keep the window open for a few seconds to show the final screen
    pygame.time.wait(3000)

    env.close()