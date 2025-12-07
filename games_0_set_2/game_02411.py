
# Generated: 2025-08-28T04:44:55.179032
# Source Brief: brief_02411.md
# Brief Index: 2411

        
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
        "Controls: Arrow keys to move. Your aim follows your last movement. "
        "Hold Space to shoot and Shift to reload."
    )

    game_description = (
        "Survive waves of zombies in a top-down horror arena. "
        "Manage your ammo, aim carefully, and stay alive for as long as you can."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.MAX_WAVES = 5
        self.PLAYER_SPEED = 3.0
        self.PLAYER_RADIUS = 10
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_MAX_AMMO = 20
        self.BULLET_SPEED = 12.0
        self.BULLET_RADIUS = 3
        self.INITIAL_ZOMBIE_COUNT = 20
        self.INITIAL_ZOMBIE_SPEED = 0.5
        self.ZOMBIE_SPEED_INCREMENT = 0.05
        self.ZOMBIE_RADIUS = 8
        self.ZOMBIE_DAMAGE = 0.5 # Damage per frame of contact
        self.ARENA_RADIUS = 180
        self.SHOOT_COOLDOWN_FRAMES = 6  # 5 shots per second at 30fps
        self.RELOAD_TIME_FRAMES = 60 # 2 seconds at 30fps

        # Colors
        self.COLOR_BG = (25, 20, 20)
        self.COLOR_ARENA_BG = (40, 35, 35)
        self.COLOR_ARENA_LINE = (80, 70, 70)
        self.COLOR_PLAYER = (50, 200, 255)
        self.COLOR_PLAYER_GLOW = (50, 200, 255, 50)
        self.COLOR_ZOMBIE = (100, 120, 80)
        self.COLOR_BULLET = (255, 255, 150)
        self.COLOR_BLOOD = (130, 20, 20)
        self.COLOR_MUZZLE_FLASH = (255, 220, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH = (220, 40, 40)
        self.COLOR_AMMO = (100, 200, 100)
        self.COLOR_RELOAD = (255, 150, 0)
        self.COLOR_UI_BG = (0, 0, 0, 128)

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_huge = pygame.font.SysFont("Consolas", 48, bold=True)
        self.center_pos = pygame.math.Vector2(self.WIDTH // 2, self.HEIGHT // 2)

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_ammo = None
        self.player_aim_vec = None
        self.zombies = None
        self.bullets = None
        self.particles = None
        self.wave_number = None
        self.zombie_speed = None
        self.zombies_to_spawn_this_wave = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win_game = None
        self.shoot_cooldown = None
        self.reload_timer = None
        
        self.reset()
        
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = self.center_pos.copy()
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_ammo = self.PLAYER_MAX_AMMO
        self.player_aim_vec = pygame.math.Vector2(1, 0)  # Start aiming right

        self.zombies = []
        self.bullets = []
        self.particles = []

        self.wave_number = 1
        self.zombie_speed = self.INITIAL_ZOMBIE_SPEED
        self.zombies_to_spawn_this_wave = self.INITIAL_ZOMBIE_COUNT
        self._spawn_wave()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_game = False

        self.shoot_cooldown = 0
        self.reload_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # Handle game over state
        if self.game_over or self.win_game:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # Unpack action and handle input
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward += self._handle_input(movement, space_held, shift_held)

        # Update game state
        self._update_player()
        reward += self._update_bullets()
        reward += self._update_zombies()
        self._update_particles()
        
        # Wave progression
        if not self.zombies and self.zombies_to_spawn_this_wave == 0:
            if self.wave_number == self.MAX_WAVES:
                self.win_game = True
                reward += 500
            else:
                reward += 100
                self.wave_number += 1
                self.zombie_speed += self.ZOMBIE_SPEED_INCREMENT
                self.zombies_to_spawn_this_wave = self.INITIAL_ZOMBIE_COUNT + (self.wave_number - 1) * 5
                self._spawn_wave()
        
        # Check termination conditions
        if self.player_health <= 0:
            self.game_over = True
            reward -= 100
        
        terminated = self.game_over or self.win_game or self.steps >= self.MAX_STEPS
        
        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        # Player Movement & Aiming
        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1 # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1 # Right
        
        if move_vec.length() > 0:
            self.player_aim_vec = move_vec.normalize()
            self.player_pos += self.player_aim_vec * self.PLAYER_SPEED
            
            # Clamp player position to arena
            dist_from_center = self.player_pos.distance_to(self.center_pos)
            if dist_from_center > self.ARENA_RADIUS - self.PLAYER_RADIUS:
                self.player_pos = self.center_pos + (self.player_pos - self.center_pos).normalize() * (self.ARENA_RADIUS - self.PLAYER_RADIUS)

        # Shooting
        if space_held and self.player_ammo > 0 and self.shoot_cooldown == 0 and self.reload_timer == 0:
            # sfx: player_shoot.wav
            bullet_pos = self.player_pos + self.player_aim_vec * (self.PLAYER_RADIUS + 5)
            bullet_vel = self.player_aim_vec * self.BULLET_SPEED
            self.bullets.append({'pos': bullet_pos, 'vel': bullet_vel})
            self.player_ammo -= 1
            self.shoot_cooldown = self.SHOOT_COOLDOWN_FRAMES
            self._create_muzzle_flash(bullet_pos, self.player_aim_vec)
        
        # Reloading
        if shift_held and self.reload_timer == 0 and self.player_ammo < self.PLAYER_MAX_AMMO:
            # sfx: reload_start.wav
            self.reload_timer = self.RELOAD_TIME_FRAMES
            
        return reward

    def _update_player(self):
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        
        if self.reload_timer > 0:
            self.reload_timer -= 1
            if self.reload_timer == 0:
                # sfx: reload_complete.wav
                self.player_ammo = self.PLAYER_MAX_AMMO

    def _update_bullets(self):
        reward = 0
        for bullet in self.bullets[:]:
            bullet['pos'] += bullet['vel']
            
            # Check for zombie collision
            hit = False
            for zombie in self.zombies[:]:
                if bullet['pos'].distance_to(zombie['pos']) < self.ZOMBIE_RADIUS + self.BULLET_RADIUS:
                    reward += 0.1 # Hit reward
                    reward += 1.0 # Kill reward
                    self.zombies.remove(zombie)
                    self._create_blood_splatter(zombie['pos'])
                    self.bullets.remove(bullet)
                    hit = True
                    # sfx: zombie_hit.wav
                    break
            if hit:
                continue

            # Check for out of bounds
            if not self.screen.get_rect().collidepoint(bullet['pos']):
                reward -= 0.01 # Miss penalty
                self.bullets.remove(bullet)
        return reward

    def _update_zombies(self):
        reward = 0
        for zombie in self.zombies:
            zombie['pos'].move_towards_ip(self.player_pos, self.zombie_speed)
            
            # Check for player collision
            if zombie['pos'].distance_to(self.player_pos) < self.ZOMBIE_RADIUS + self.PLAYER_RADIUS:
                self.player_health -= self.ZOMBIE_DAMAGE
                reward -= 1.0 * self.ZOMBIE_DAMAGE # Damage penalty
                self.player_health = max(0, self.player_health)
                # sfx: player_hurt.wav
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _spawn_wave(self):
        for _ in range(self.zombies_to_spawn_this_wave):
            self._spawn_zombie()
        self.zombies_to_spawn_this_wave = 0

    def _spawn_zombie(self):
        angle = self.np_random.uniform(0, 2 * math.pi)
        # Spawn just outside the arena
        spawn_radius = self.ARENA_RADIUS + 20
        x = self.center_pos.x + spawn_radius * math.cos(angle)
        y = self.center_pos.y + spawn_radius * math.sin(angle)
        self.zombies.append({'pos': pygame.math.Vector2(x, y)})

    def _create_blood_splatter(self, pos):
        # sfx: blood_splat.wav
        for _ in range(self.np_random.integers(15, 25)):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifetime': self.np_random.integers(10, 20),
                'color': self.COLOR_BLOOD,
                'size': self.np_random.integers(2, 5)
            })

    def _create_muzzle_flash(self, pos, direction):
        for _ in range(self.np_random.integers(5, 10)):
            angle_offset = self.np_random.uniform(-math.pi / 4, math.pi / 4)
            angle = math.atan2(direction.y, direction.x) + angle_offset
            speed = self.np_random.uniform(2, 6)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifetime': self.np_random.integers(2, 5),
                'color': self.COLOR_MUZZLE_FLASH,
                'size': self.np_random.integers(3, 6)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Arena
        pygame.gfxdraw.filled_circle(self.screen, int(self.center_pos.x), int(self.center_pos.y), self.ARENA_RADIUS, self.COLOR_ARENA_BG)
        pygame.gfxdraw.aacircle(self.screen, int(self.center_pos.x), int(self.center_pos.y), self.ARENA_RADIUS, self.COLOR_ARENA_LINE)

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), p['size'])

        # Zombies
        for z in self.zombies:
            pygame.gfxdraw.filled_circle(self.screen, int(z['pos'].x), int(z['pos'].y), self.ZOMBIE_RADIUS, self.COLOR_ZOMBIE)
            pygame.gfxdraw.aacircle(self.screen, int(z['pos'].x), int(z['pos'].y), self.ZOMBIE_RADIUS, tuple(int(c*0.8) for c in self.COLOR_ZOMBIE))

        # Player
        # Glow
        glow_radius = int(self.PLAYER_RADIUS * 1.5)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (int(self.player_pos.x - glow_radius), int(self.player_pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)
        # Body
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_RADIUS, self.COLOR_PLAYER)
        # Aim indicator
        aim_end_pos = self.player_pos + self.player_aim_vec * self.PLAYER_RADIUS
        pygame.draw.line(self.screen, self.COLOR_BG, (int(self.player_pos.x), int(self.player_pos.y)), (int(aim_end_pos.x), int(aim_end_pos.y)), 3)

        # Bullets
        for b in self.bullets:
            pygame.draw.circle(self.screen, self.COLOR_BULLET, (int(b['pos'].x), int(b['pos'].y)), self.BULLET_RADIUS)
    
    def _render_ui(self):
        # Health Bar
        health_pct = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, (50,50,50), (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (10, 10, int(bar_width * health_pct), bar_height))
        health_text = self.font_small.render(f"{int(self.player_health)}/{self.PLAYER_MAX_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Ammo Count
        ammo_text = self.font_large.render(f"{self.player_ammo}/{self.PLAYER_MAX_AMMO}", True, self.COLOR_AMMO)
        text_rect = ammo_text.get_rect(bottomright=(self.WIDTH - 15, self.HEIGHT - 10))
        self.screen.blit(ammo_text, text_rect)
        
        # Reload Indicator
        if self.reload_timer > 0:
            reload_pct = self.reload_timer / self.RELOAD_TIME_FRAMES
            reload_text = self.font_small.render("RELOADING", True, self.COLOR_RELOAD)
            text_rect = reload_text.get_rect(midbottom=(self.player_pos.x, self.player_pos.y - self.PLAYER_RADIUS - 15))
            self.screen.blit(reload_text, text_rect)
            
            # Reload progress bar
            bar_width = 40
            bar_height = 5
            bar_x = self.player_pos.x - bar_width / 2
            bar_y = self.player_pos.y - self.PLAYER_RADIUS - 10
            pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_RELOAD, (bar_x, bar_y, int(bar_width * (1-reload_pct)), bar_height))


        # Wave Number
        wave_text = self.font_large.render(f"WAVE {self.wave_number}", True, self.COLOR_TEXT)
        text_rect = wave_text.get_rect(midtop=(self.WIDTH // 2, 10))
        self.screen.blit(wave_text, text_rect)

        # Game Over / Win Message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            msg = self.font_huge.render("GAME OVER", True, self.COLOR_HEALTH)
            text_rect = msg.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg, text_rect)
        elif self.win_game:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            msg = self.font_huge.render("YOU SURVIVED", True, self.COLOR_PLAYER)
            text_rect = msg.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "health": self.player_health,
            "ammo": self.player_ammo,
            "zombies_remaining": len(self.zombies),
        }

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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    env.reset()
    
    # To test the environment with keyboard controls
    import pygame
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Zombie Survival")
    
    running = True
    terminated = False
    
    while running:
        # Action defaults
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space_held = 1
            
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift_held = 1
            
            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Draw the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        env.clock.tick(30) # Match the auto_advance rate
        
    env.close()