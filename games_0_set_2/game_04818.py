
# Generated: 2025-08-28T03:05:51.527115
# Source Brief: brief_04818.md
# Brief Index: 4818

        
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
    metadata = {"render_modes": ["rgb_array", "human"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to shoot in the direction you are facing."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of zombies in a top-down arena shooter. Clear 5 waves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    ARENA_SIZE = 400
    ARENA_RECT = pygame.Rect((WIDTH - ARENA_SIZE) // 2, (HEIGHT - ARENA_SIZE) // 2, ARENA_SIZE, ARENA_SIZE)

    COLOR_BG = (20, 20, 30)
    COLOR_ARENA = (50, 50, 60)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_ZOMBIE = (255, 50, 50)
    COLOR_BULLET = (255, 255, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)
    COLOR_HEALTH_BAR_FG = (0, 200, 0)

    PLAYER_RADIUS = 10
    PLAYER_SPEED = 4.0
    PLAYER_MAX_HEALTH = 100

    ZOMBIE_RADIUS = 8
    ZOMBIE_BASE_SPEED = 1.0
    ZOMBIE_SPEED_INCREMENT = 0.2
    ZOMBIE_HEALTH = 30
    ZOMBIE_DAMAGE = 1.0 # per frame of contact

    BULLET_RADIUS = 3
    BULLET_SPEED = 8.0
    BULLET_DAMAGE = 10
    SHOOT_COOLDOWN_FRAMES = 5 # 6 shots per second at 30fps

    MAX_WAVES = 5
    MAX_STEPS = 3000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        self.display = None
        if self.render_mode == "human":
            self.display = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Zombie Survival")

        self.np_random = None
        self.player_pos = None
        self.player_health = None
        self.player_facing_direction = None
        self.zombies = []
        self.bullets = []
        self.particles = []
        self.wave = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.shoot_cooldown = 0
        self.screen_flash_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.player_pos = pygame.Vector2(self.ARENA_RECT.center)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_facing_direction = pygame.Vector2(0, -1) # Start facing up

        self.zombies = []
        self.bullets = []
        self.particles = []
        
        self.wave = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.shoot_cooldown = 0
        self.screen_flash_timer = 0
        
        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # In human mode, allow quitting even after game over
            if self.render_mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        if self.render_mode == "human":
            self.clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True # End the episode
        
        # --- Update Cooldowns ---
        if self.shoot_cooldown > 0: self.shoot_cooldown -= 1
        if self.screen_flash_timer > 0: self.screen_flash_timer -= 1

        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game Objects ---
        self._update_bullets()
        self._update_zombies()
        self._update_particles()

        # --- Handle Collisions and Damage ---
        reward += self._handle_collisions()

        # --- Check Game State ---
        if not self.zombies and self.wave <= self.MAX_WAVES and self.player_health > 0:
            reward += 10 # Wave clear bonus
            self._start_next_wave()
        
        terminated = self.player_health <= 0 or self.wave > self.MAX_WAVES or self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.player_health <= 0:
                reward -= 100 # Death penalty
            elif self.wave > self.MAX_WAVES:
                reward += 100 # Victory bonus
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        move_vector = pygame.Vector2(0, 0)
        if movement == 1: move_vector.y = -1
        elif movement == 2: move_vector.y = 1
        elif movement == 3: move_vector.x = -1
        elif movement == 4: move_vector.x = 1
        
        if move_vector.length() > 0:
            self.player_facing_direction = move_vector.normalize()
            self.player_pos += self.player_facing_direction * self.PLAYER_SPEED
        
        self.player_pos.x = max(self.ARENA_RECT.left + self.PLAYER_RADIUS, min(self.ARENA_RECT.right - self.PLAYER_RADIUS, self.player_pos.x))
        self.player_pos.y = max(self.ARENA_RECT.top + self.PLAYER_RADIUS, min(self.ARENA_RECT.bottom - self.PLAYER_RADIUS, self.player_pos.y))

        if space_held and self.shoot_cooldown == 0:
            # sfx: player_shoot.wav
            self._create_bullet()
            self._create_muzzle_flash()
            self.shoot_cooldown = self.SHOOT_COOLDOWN_FRAMES

    def _update_bullets(self):
        for bullet in self.bullets[:]:
            bullet['pos'] += bullet['vel']
            if not self.ARENA_RECT.collidepoint(bullet['pos']):
                self.bullets.remove(bullet)

    def _update_zombies(self):
        current_zombie_speed = self.ZOMBIE_BASE_SPEED + self.ZOMBIE_SPEED_INCREMENT * (self.wave - 1)
        for zombie in self.zombies:
            direction_to_player = self.player_pos - zombie['pos']
            if direction_to_player.length_squared() > 0:
                direction_to_player.normalize_ip()
            
            oscillation_offset = direction_to_player.rotate(90) * math.sin(self.steps * 0.2 + zombie['anim_offset']) * 0.5
            zombie['pos'] += (direction_to_player + oscillation_offset) * current_zombie_speed
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        
        for bullet in self.bullets[:]:
            for zombie in self.zombies[:]:
                if bullet['pos'].distance_to(zombie['pos']) < self.BULLET_RADIUS + self.ZOMBIE_RADIUS:
                    # sfx: zombie_hit.wav
                    zombie['health'] -= self.BULLET_DAMAGE
                    reward += 0.1
                    self._create_hit_sparks(bullet['pos'])
                    if bullet in self.bullets: self.bullets.remove(bullet)
                    
                    if zombie['health'] <= 0:
                        # sfx: zombie_die.wav
                        reward += 1.0
                        self._create_death_particles(zombie['pos'])
                        if zombie in self.zombies: self.zombies.remove(zombie)
                    break 

        for zombie in self.zombies:
            if self.player_pos.distance_to(zombie['pos']) < self.PLAYER_RADIUS + self.ZOMBIE_RADIUS:
                # sfx: player_damage.wav
                self.player_health -= self.ZOMBIE_DAMAGE
                reward -= 0.01
                self.screen_flash_timer = 3
        
        self.player_health = max(0, self.player_health)
        return reward

    def _start_next_wave(self):
        self.wave += 1
        if self.wave > self.MAX_WAVES:
            return

        num_zombies = 5 + (self.wave - 1) * 2
        for _ in range(num_zombies):
            self._spawn_zombie()

    def _spawn_zombie(self):
        while True:
            edge = self.np_random.integers(0, 4)
            if edge == 0: pos = pygame.Vector2(self.np_random.uniform(self.ARENA_RECT.left, self.ARENA_RECT.right), self.ARENA_RECT.top - 20)
            elif edge == 1: pos = pygame.Vector2(self.np_random.uniform(self.ARENA_RECT.left, self.ARENA_RECT.right), self.ARENA_RECT.bottom + 20)
            elif edge == 2: pos = pygame.Vector2(self.ARENA_RECT.left - 20, self.np_random.uniform(self.ARENA_RECT.top, self.ARENA_RECT.bottom))
            else: pos = pygame.Vector2(self.ARENA_RECT.right + 20, self.np_random.uniform(self.ARENA_RECT.top, self.ARENA_RECT.bottom))
            if pos.distance_to(self.player_pos) > 100: break
        
        self.zombies.append({
            'pos': pos, 'health': self.ZOMBIE_HEALTH, 'anim_offset': self.np_random.uniform(0, 2 * math.pi)
        })

    def _create_bullet(self):
        pos = self.player_pos + self.player_facing_direction * self.PLAYER_RADIUS
        vel = self.player_facing_direction * self.BULLET_SPEED
        self.bullets.append({'pos': pos, 'vel': vel})
        
    def _create_muzzle_flash(self):
        pos = self.player_pos + self.player_facing_direction * (self.PLAYER_RADIUS + 5)
        self.particles.append({'pos': pos, 'vel': pygame.Vector2(0,0), 'lifespan': 2, 'radius': 8, 'color': (255, 255, 200)})

    def _create_hit_sparks(self, pos):
        for _ in range(5):
            angle, speed = self.np_random.uniform(0, 2 * math.pi), self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifespan': self.np_random.integers(5, 10), 'radius': self.np_random.uniform(1, 3), 'color': self.COLOR_BULLET})
            
    def _create_death_particles(self, pos):
        for _ in range(20):
            angle, speed = self.np_random.uniform(0, 2 * math.pi), self.np_random.uniform(0.5, 3)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifespan': self.np_random.integers(15, 30), 'radius': self.np_random.uniform(2, 5), 'color': self.COLOR_ZOMBIE})

    def _render_frame(self):
        self.screen.fill(self.COLOR_BG)
        pygame.draw.rect(self.screen, self.COLOR_ARENA, self.ARENA_RECT)
        
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30)) if 'lifespan' in p and p['lifespan'] < 30 else 255
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))

        for bullet in self.bullets:
            pygame.draw.circle(self.screen, self.COLOR_BULLET, (int(bullet['pos'].x), int(bullet['pos'].y)), self.BULLET_RADIUS)

        for zombie in self.zombies:
            pos = (int(zombie['pos'].x), int(zombie['pos'].y))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ZOMBIE_RADIUS, self.COLOR_ZOMBIE)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ZOMBIE_RADIUS, self.COLOR_ZOMBIE)

        if self.player_health > 0:
            pos_int = (int(self.player_pos.x), int(self.player_pos.y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS + 3, (self.COLOR_PLAYER[0], self.COLOR_PLAYER[1], self.COLOR_PLAYER[2], 50))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

        health_bar_width, health_bar_height = 200, 20
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, health_bar_width, health_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, int(health_bar_width * health_ratio), health_bar_height))
        
        wave_text = self.font_small.render(f"Wave: {min(self.wave, self.MAX_WAVES)}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))
        
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, self.HEIGHT - 30))

        if self.game_over:
            end_text = self.font_large.render("VICTORY!" if self.player_health > 0 else "GAME OVER", True, self.COLOR_PLAYER if self.player_health > 0 else self.COLOR_ZOMBIE)
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - end_text.get_height() // 2))

        if self.screen_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((255, 0, 0, 80))
            self.screen.blit(flash_surface, (0, 0))

    def _get_observation(self):
        self._render_frame()
        if self.render_mode == "human":
            self.display.blit(self.screen, (0, 0))
            pygame.display.flip()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave, "player_health": self.player_health, "zombies_remaining": len(self.zombies)}
    
    def close(self):
        if self.display:
            pygame.display.quit()
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    terminated = False
    total_reward = 0
    
    while not terminated:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if info['steps'] % 30 == 0: # Print info once per second
            print(f"Step: {info['steps']}, Wave: {info['wave']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Terminated: {terminated}")

    print(f"Game Over. Final Score: {total_reward:.2f}")
    env.close()