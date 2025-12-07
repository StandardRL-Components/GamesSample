
# Generated: 2025-08-27T16:21:07.322094
# Source Brief: brief_01196.md
# Brief Index: 1196

        
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
        "Controls: Arrow keys to move. Hold Space to shoot. Press Shift to reload."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of zombies in a top-down arena shooter. Clear waves to score points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 2000
    
    # Colors
    COLOR_BG = (25, 25, 30)
    COLOR_WALL = (60, 60, 70)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_ZOMBIE = (255, 50, 50)
    COLOR_BULLET = (255, 255, 100)
    COLOR_MUZZLE_FLASH = (255, 255, 200)
    COLOR_BLOOD = (180, 0, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_HEALTH_BAR = (50, 255, 50)
    COLOR_HEALTH_BAR_BG = (120, 0, 0)
    COLOR_RELOAD = (255, 150, 0)

    # Player
    PLAYER_SIZE = 20
    PLAYER_SPEED = 4.0
    PLAYER_HEALTH_MAX = 100
    
    # Weapon
    MAX_AMMO = 10
    SHOOT_COOLDOWN = 6 # frames
    RELOAD_TIME = 60 # frames
    BULLET_SPEED = 15.0
    BULLET_SIZE = (10, 4)

    # Zombie
    ZOMBIE_SIZE = 22
    ZOMBIE_HEALTH = 1
    ZOMBIE_BASE_SPEED = 1.0
    ZOMBIE_SPEED_VARIATION = 0.5
    ZOMBIE_SPAWN_WAVE_BASE = 5
    ZOMBIE_SPAWN_WAVE_INCREMENT = 5
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_aim_angle = None
        self.ammo = None
        self.shoot_cooldown_timer = None
        self.reload_timer = None
        self.zombies = None
        self.bullets = None
        self.particles = None
        self.wave = None
        self.steps = None
        self.score = None
        self.game_over = None
        
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_aim_angle = 0  # Pointing right
        
        self.ammo = self.MAX_AMMO
        self.shoot_cooldown_timer = 0
        self.reload_timer = 0
        
        self.zombies = []
        self.bullets = []
        self.particles = []
        
        self.wave = 1
        self._spawn_wave()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.steps += 1
        
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Update timers ---
        if self.shoot_cooldown_timer > 0:
            self.shoot_cooldown_timer -= 1
        if self.reload_timer > 0:
            self.reload_timer -= 1
            if self.reload_timer == 0:
                self.ammo = self.MAX_AMMO
                # sfx: reload_complete.wav
        
        # --- Handle player actions ---
        self._handle_movement(movement)
        
        if shift_held and self.ammo < self.MAX_AMMO and self.reload_timer == 0:
            self.reload_timer = self.RELOAD_TIME
            # sfx: reload_start.wav

        if space_held and self.reload_timer == 0:
            reward += self._handle_shooting()
        
        # --- Update game objects ---
        self._update_bullets()
        self._update_zombies()
        self._update_particles()
        
        # --- Handle collisions and damage ---
        reward += self._handle_collisions()
        
        # --- Check game state ---
        if not self.zombies:
            reward += 10.0 # Wave clear bonus
            self.wave += 1
            self._spawn_wave()
            # sfx: wave_cleared.wav

        # --- Termination conditions ---
        terminated = False
        if self.player_health <= 0:
            reward = -100.0 # Loss penalty
            terminated = True
            self.game_over = True
            # sfx: player_death.wav
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement_action):
        move_vec = pygame.Vector2(0, 0)
        if movement_action == 1: # Up
            move_vec.y = -1
        elif movement_action == 2: # Down
            move_vec.y = 1
        elif movement_action == 3: # Left
            move_vec.x = -1
        elif movement_action == 4: # Right
            move_vec.x = 1

        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED
            self.player_aim_angle = move_vec.angle_to(pygame.Vector2(1, 0))

        # Clamp player position
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE / 2, self.WIDTH - self.PLAYER_SIZE / 2)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE / 2, self.HEIGHT - self.PLAYER_SIZE / 2)

    def _handle_shooting(self):
        if self.ammo > 0 and self.shoot_cooldown_timer == 0:
            self.ammo -= 1
            self.shoot_cooldown_timer = self.SHOOT_COOLDOWN
            
            angle_rad = math.radians(-self.player_aim_angle)
            vel = pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * self.BULLET_SPEED
            
            # Spawn bullet slightly in front of the player
            spawn_offset = pygame.Vector2(self.PLAYER_SIZE / 2, 0).rotate(-self.player_aim_angle)
            bullet_pos = self.player_pos + spawn_offset
            
            self.bullets.append({'pos': bullet_pos, 'vel': vel, 'angle': self.player_aim_angle})
            
            # Muzzle flash particle
            flash_pos = self.player_pos + spawn_offset * 1.5
            self.particles.append({'pos': flash_pos, 'type': 'flash', 'lifespan': 3})
            # sfx: shoot.wav
            return 0 # No reward for just shooting
        elif self.ammo <= 0 and self.shoot_cooldown_timer == 0:
            self.shoot_cooldown_timer = self.SHOOT_COOLDOWN # Cooldown for empty click
            # sfx: empty_clip.wav
        return 0

    def _update_bullets(self):
        for bullet in self.bullets[:]:
            bullet['pos'] += bullet['vel']
            if not (0 < bullet['pos'].x < self.WIDTH and 0 < bullet['pos'].y < self.HEIGHT):
                self.bullets.remove(bullet)
                # This could be a small penalty for missing
                # self.score -= 0.01

    def _update_zombies(self):
        for zombie in self.zombies:
            direction = (self.player_pos - zombie['pos']).normalize()
            zombie['pos'] += direction * zombie['speed']

    def _update_particles(self):
        for p in self.particles[:]:
            p['lifespan'] -= 1
            if p['type'] == 'blood':
                p['pos'] += p['vel']
                p['vel'] *= 0.9 # friction
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        
        # Bullets and Zombies
        for bullet in self.bullets[:]:
            bullet_rect = pygame.Rect(bullet['pos'].x - self.BULLET_SIZE[0]/2, bullet['pos'].y - self.BULLET_SIZE[1]/2, self.BULLET_SIZE[0], self.BULLET_SIZE[1])
            for zombie in self.zombies[:]:
                zombie_rect = pygame.Rect(zombie['pos'].x - self.ZOMBIE_SIZE/2, zombie['pos'].y - self.ZOMBIE_SIZE/2, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
                if bullet_rect.colliderect(zombie_rect):
                    reward += 0.1 # Hit reward
                    self.zombies.remove(zombie)
                    reward += 1.0 # Kill reward
                    if bullet in self.bullets: self.bullets.remove(bullet)
                    self._create_blood_splatter(zombie['pos'])
                    # sfx: zombie_die.wav
                    break
        
        # Player and Zombies
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE/2, self.player_pos.y - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for zombie in self.zombies[:]:
            zombie_rect = pygame.Rect(zombie['pos'].x - self.ZOMBIE_SIZE/2, zombie['pos'].y - self.ZOMBIE_SIZE/2, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
            if player_rect.colliderect(zombie_rect):
                self.player_health -= 10
                self.player_health = max(0, self.player_health)
                # sfx: player_hit.wav
                # Knockback zombie to prevent instant multi-hits
                knockback_dir = (zombie['pos'] - self.player_pos).normalize()
                zombie['pos'] += knockback_dir * self.ZOMBIE_SIZE * 1.5
                zombie['pos'].x = np.clip(zombie['pos'].x, 0, self.WIDTH)
                zombie['pos'].y = np.clip(zombie['pos'].y, 0, self.HEIGHT)
        
        return reward
    
    def _spawn_wave(self):
        num_zombies = self.ZOMBIE_SPAWN_WAVE_BASE + (self.wave - 1) * self.ZOMBIE_SPAWN_WAVE_INCREMENT
        for _ in range(num_zombies):
            # Spawn on edges
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -self.ZOMBIE_SIZE)
            elif edge == 1: # Bottom
                pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_SIZE)
            elif edge == 2: # Left
                pos = pygame.Vector2(-self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT))
            else: # Right
                pos = pygame.Vector2(self.WIDTH + self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT))
            
            speed = self.ZOMBIE_BASE_SPEED + self.np_random.uniform(-self.ZOMBIE_SPEED_VARIATION, self.ZOMBIE_SPEED_VARIATION)
            self.zombies.append({'pos': pos, 'speed': max(0.5, speed)})
    
    def _create_blood_splatter(self, pos):
        for _ in range(self.np_random.integers(15, 25)):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(10, 30)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'type': 'blood', 'lifespan': lifespan, 'size': self.np_random.integers(1, 4)})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render particles (underneath other entities)
        for p in self.particles:
            if p['type'] == 'blood':
                pygame.draw.circle(self.screen, self.COLOR_BLOOD, (int(p['pos'].x), int(p['pos'].y)), p['size'])

        # Render zombies
        for zombie in self.zombies:
            rect = pygame.Rect(0, 0, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
            rect.center = (int(zombie['pos'].x), int(zombie['pos'].y))
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, rect)

        # Render player
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = (int(self.player_pos.x), int(self.player_pos.y))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)

        # Render player's gun barrel
        barrel_length = self.PLAYER_SIZE * 0.75
        barrel_end = self.player_pos + pygame.Vector2(barrel_length, 0).rotate(-self.player_aim_angle)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, self.player_pos, barrel_end, 3)

        # Render bullets
        for bullet in self.bullets:
            rotated_surf = pygame.Surface(self.BULLET_SIZE, pygame.SRCALPHA)
            rotated_surf.fill(self.COLOR_BULLET)
            rotated_surf = pygame.transform.rotate(rotated_surf, bullet['angle'])
            rect = rotated_surf.get_rect(center=bullet['pos'])
            self.screen.blit(rotated_surf, rect)

        # Render muzzle flash (on top)
        for p in self.particles:
            if p['type'] == 'flash':
                radius = int(15 * (p['lifespan'] / 3.0))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), radius, self.COLOR_MUZZLE_FLASH)
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'].x), int(p['pos'].y), radius, self.COLOR_MUZZLE_FLASH)

    def _render_ui(self):
        # Health bar
        health_ratio = self.player_health / self.PLAYER_HEALTH_MAX
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_ratio), bar_height))

        # Ammo count
        ammo_text = self.font_small.render(f"AMMO: {self.ammo}/{self.MAX_AMMO}", True, self.COLOR_TEXT)
        self.screen.blit(ammo_text, (self.WIDTH - ammo_text.get_width() - 10, 10))
        
        # Reload indicator
        if self.reload_timer > 0:
            reload_text = self.font_small.render("RELOADING...", True, self.COLOR_RELOAD)
            text_rect = reload_text.get_rect(center=(self.WIDTH/2, self.HEIGHT - 50))
            self.screen.blit(reload_text, text_rect)

        # Wave number
        wave_text = self.font_large.render(f"WAVE {self.wave}", True, self.COLOR_TEXT)
        text_rect = wave_text.get_rect(center=(self.WIDTH/2, self.HEIGHT - 25))
        self.screen.blit(wave_text, text_rect)

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            game_over_text = self.font_large.render("GAME OVER", True, self.COLOR_ZOMBIE)
            text_rect = game_over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "health": self.player_health,
            "ammo": self.ammo,
            "zombies_left": len(self.zombies),
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Zombie Arena")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        if terminated:
            # After game over, wait for a key press to reset
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    obs, info = env.reset()
                    terminated = False
        else:
            # --- Get human input ---
            movement = 0 # none
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Step environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            
            # --- Render to screen ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        clock.tick(30) # Run at 30 FPS

    env.close()